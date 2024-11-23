class SnapshotStorage:
    def __init__(self, max_snapshots=None):
        """
        Manages storage of PyTree snapshots.

        Args:
            max_snapshots (int, optional): Maximum number of snapshots to store. Defaults to None (no limit).
        """
        self.snapshots = {}
        self.snapshot_order = []
        self.max_snapshots = max_snapshots

    def add_snapshot(self, snapshot_id, pytree_snapshot, overwrite=False):
        """
        Adds a Snapshot to storage, optionally overwriting an existing one.

        Args:
            snapshot_id (str): The ID of the snapshot.
            pytree_snapshot (Snapshot): The Snapshot object to store.
            overwrite (bool): Whether to overwrite an existing snapshot. Defaults to False.

        Raises:
            ValueError: If the snapshot ID already exists and `overwrite` is False.
        """
        if snapshot_id in self.snapshots:
            if not overwrite:
                raise ValueError(
                    f"Snapshot ID '{snapshot_id}' already exists. Use overwrite=True to update it."
                )
            # Overwrite logic
            self.snapshots[snapshot_id] = pytree_snapshot
        else:
            # Enforce max_snapshots limit
            if (
                self.max_snapshots is not None
                and len(self.snapshots) >= self.max_snapshots
            ):
                oldest_snapshot = self.snapshot_order.pop(0)
                del self.snapshots[oldest_snapshot]

            self.snapshots[snapshot_id] = pytree_snapshot
            self.snapshot_order.append(snapshot_id)

    def get_snapshot(self, snapshot_id):
        """
        Retrieves a Snapshot by its ID.

        Args:
            snapshot_id (str): The ID of the snapshot.

        Returns:
            Snapshot: The corresponding Snapshot object.

        Raises:
            ValueError: If the snapshot ID does not exist.
        """
        if snapshot_id not in self.snapshots:
            raise ValueError(f"Snapshot ID '{snapshot_id}' does not exist.")
        return self.snapshots[snapshot_id]

    def get_latest_snapshot_id(self):
        """
        Retrieve the ID of the most recent snapshot.

        Returns:
            str: The ID of the latest snapshot.

        Raises:
            IndexError: If no snapshots are available.
        """
        if not self.snapshot_order:
            raise IndexError("No snapshots available.")
        return self.snapshot_order[-1]

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
            return True
        return False
