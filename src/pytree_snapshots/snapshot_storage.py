class SnapshotStorage:
    def __init__(self, max_snapshots=None, cmp_function=None):
        """
        Manages storage of PyTree snapshots.

        Args:
            max_snapshots (int, optional): Maximum number of snapshots to store. Defaults to None (no limit).
            cmp_function (callable, optional): Comparison function to order snapshots.
                                               Should take two snapshots and return:
                                               - Negative value if snapshot1 < snapshot2
                                               - 0 if snapshot1 == snapshot2
                                               - Positive value if snapshot1 > snapshot2
        """
        self.snapshots = {}
        self.snapshot_order = []  # List of snapshot IDs in insertion order
        self.ranked_snapshot_order = []  # List of snapshot IDs ordered by cmp_function
        self.max_snapshots = max_snapshots
        self.cmp_function = cmp_function

    def _insert_ranked(self, snapshot_id):
        """
        Insert a snapshot ID into the ranked_snapshot_order using the cmp_function for ranking.

        Args:
            snapshot_id (str): The ID of the snapshot to insert.
        """
        if not self.cmp_function:
            raise ValueError("A cmp_function must be provided to use ranked order.")

        # Perform manual binary search to find the correct position
        position = 0
        for idx, sid in enumerate(self.ranked_snapshot_order):
            if self.cmp_function(self.snapshots[snapshot_id], self.snapshots[sid]) > 0:
                position = idx
                break
            position += 1

        self.ranked_snapshot_order.insert(position, snapshot_id)

    def _remove_ranked(self, snapshot_id):
        """
        Remove a snapshot ID from the ranked_snapshot_order.

        Args:
            snapshot_id (str): The ID of the snapshot to remove.
        """
        self.ranked_snapshot_order.remove(snapshot_id)

    def get_ranked_snapshots(self):
        """
        Get the snapshots ordered by the cmp_function or by age if no cmp_function is provided.

        Returns:
            list: A list of snapshot IDs ordered based on the cmp_function or age.
        """
        if self.cmp_function:
            return self.ranked_snapshot_order.copy()
        return self.snapshot_order.copy()

    def add_snapshot(self, snapshot_id, snapshot, overwrite=False):
        """
        Adds a snapshot to storage, optionally overwriting an existing one.

        Args:
            snapshot_id (str): The ID of the snapshot.
            snapshot (Snapshot): The Snapshot object to store.
            overwrite (bool): Whether to overwrite an existing snapshot. Defaults to False.

        Raises:
            ValueError: If the snapshot ID already exists and `overwrite` is False.
        """
        if snapshot_id in self.snapshots:
            if not overwrite:
                raise ValueError(
                    f"Snapshot ID '{snapshot_id}' already exists. Use overwrite=True to update it."
                )
            # Overwrite existing snapshot
            self.snapshots[snapshot_id] = snapshot
            self._remove_ranked(snapshot_id)
            self._insert_ranked(snapshot_id)

            return True
        else:
            # Handle max_snapshots limit
            if (
                self.max_snapshots is not None
                and len(self.snapshots) >= self.max_snapshots
            ):
                # Get the lowest-ranked snapshot (last in ranked_snapshot_order)
                lowest_snapshot_id = (
                    self.ranked_snapshot_order[-1]
                    if self.cmp_function
                    else self.snapshot_order[0]
                )
                lowest_snapshot = self.snapshots[lowest_snapshot_id]

                # Compare the new snapshot to the lowest-ranked snapshot
                if (
                    self.cmp_function
                    and self.cmp_function(snapshot, lowest_snapshot) > 0
                ):
                    # New snapshot is better; remove the lowest one
                    self.remove_snapshot(lowest_snapshot_id)
                elif not self.cmp_function:
                    # If no cmp_function, enforce insertion order (remove oldest)
                    self.remove_snapshot(self.snapshot_order[0])
                else:
                    # New snapshot is not better; do not add
                    return False

            # Add the new snapshot
            self.snapshots[snapshot_id] = snapshot
            self.snapshot_order.append(snapshot_id)
            if self.cmp_function:
                self._insert_ranked(snapshot_id)

            return True

    def get_snapshot(self, snapshot_id):
        """
        Retrieve a snapshot by its ID.

        Args:
            snapshot_id (str): The ID of the snapshot to retrieve.

        Returns:
            Snapshot: The requested snapshot.
        """
        snapshot = self.snapshots.get(snapshot_id)
        if snapshot is None:
            raise KeyError(f"Snapshot with ID {snapshot_id} not found.")
        return snapshot

    def remove_snapshot(self, snapshot_id):
        """
        Removes a snapshot by its ID.

        Args:
            snapshot_id (str): The ID of the snapshot to remove.

        Returns:
            bool: True if the snapshot was removed, False if it did not exist.
        """
        if snapshot_id in self.snapshots:
            del self.snapshots[snapshot_id]
            self.snapshot_order.remove(snapshot_id)
            if self.cmp_function:
                self._remove_ranked(snapshot_id)
            return True
        return False

    def has_snapshot(self, snapshot_id):
        """
        Check if a snapshot exists in storage.

        Args:
            snapshot_id (str): The ID of the snapshot to check.

        Returns:
            bool: True if the snapshot exists, False otherwise.
        """
        return snapshot_id in self.snapshots
