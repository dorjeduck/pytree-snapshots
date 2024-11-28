from .utils import RankedList, RankedListAddResult


class SnapshotStorage:
    def __init__(self, max_snapshots=None, cmp=None):
        """
        Manages storage of PyTree snapshots.

        Args:
            max_snapshots (int, optional): Maximum number of snapshots to store. Defaults to None (no limit).
            cmp (callable, optional): Comparison function to order snapshots. Defaults to None.
        """
        self.max_snapshots = max_snapshots
        self.snapshots = {}  # Mapping of snapshot_id to snapshot objects
        self.insertion_order = []  # Maintains insertion order
        self.ranked_list = (
            None
            if cmp is None
            else RankedList(
                cmp=cmp,
                max_items=max_snapshots,
            )
        )

    def add_snapshot(self, snapshot, overwrite=False):
        """
        Adds a snapshot to storage, optionally overwriting an existing one.

        Args:
            snapshot (Snapshot): The snapshot to store.
            overwrite (bool): Whether to overwrite an existing snapshot. Defaults to False.

        Returns:
            bool: True if the snapshot was added, False otherwise.
        """

        # Handle overwrites
        if snapshot.id in self.snapshots:
            if not overwrite:
                raise ValueError(
                    f"Snapshot ID '{snapshot.id}' already exists. Use overwrite=True to update it."
                )
                # Overwrite snapshot
            self.snapshots[snapshot.id] = snapshot

            # Ensure ranked list reflects possible changes in ranking
            if self.ranked_list:
                self.ranked_list.sort_items()

            # Do not modify insertion_order on overwrite
            return True

        if self.ranked_list:
            result = self.ranked_list.add(snapshot)
            if result == RankedListAddResult.NOT_QUALIFIED:
                return False  # Snapshot doesn't meet ranking criteria

        # Add to snapshot storage
        self.snapshots[snapshot.id] = snapshot
        self.insertion_order.append(snapshot.id)

        # Maintain ordered list
        if (
            self.max_snapshots is not None
            and len(self.insertion_order) > self.max_snapshots
        ):
            oldest_id = self.insertion_order.pop(0)
            del self.snapshots[oldest_id]

        return True

    def remove_snapshot(self, snapshot_id):
        """
        Removes a snapshot by its ID.

        Args:
            snapshot_id (str): The ID of the snapshot to remove.

        Returns:
            bool: True if the snapshot was removed, False if it did not exist.
        """
        snapshot = self.snapshots.pop(snapshot_id, None)
        if snapshot is None:
            return False

        self.insertion_order.remove(snapshot_id)
        if self.ranked_list:
            self.ranked_list.remove(snapshot)

        return True

    def update_cmp(self, cmp):
        """
        Updates the cmp function and transitions to a ranked list.

        Args:
            cmp (callable): A new cmp function. If None, disables ranking.
        """
        self.cmp = cmp
        if cmp is None:
            # Disable ranking by setting ranked_list to None
            self.ranked_list = None
        else:
            # Re-sort the existing ranked list with the updated cmp

            self.ranked_list = RankedList(
                cmp=cmp,
                max_items=self.max_snapshots,
            )
            for snapshot in self.snapshots.values():
                self.ranked_list.add(snapshot)

    def update_max_snapshots(self, max_snapshots):
        """
        Updates the maximum number of snapshots and enforces the limit.

        Args:
            max_snapshots (int): The new maximum number of snapshots.
        """
        self.max_snapshots = max_snapshots
        if self.ranked_list:
            self.ranked_list.update_max_items(max_snapshots)
        else:
            while (
                self.max_snapshots is not None
                and len(self.insertion_order) > self.max_snapshots
            ):
                oldest_id = self.insertion_order.pop(0)
                del self.snapshots[oldest_id]

    def get_snapshot(self, snapshot_id):
        """
        Retrieves a snapshot by its ID.

        Args:
            snapshot_id (str): The ID of the snapshot to retrieve.

        Returns:
            Snapshot: The requested snapshot.
        """
        snapshot = self.snapshots.get(snapshot_id)
        if snapshot is None:
            raise KeyError(f"Snapshot with ID {snapshot_id} not found.")
        return snapshot

    def get_ids_by_rank(self):
        """
        Get the snapshots ordered by the cmp function.

        Returns:
            list: A list of snapshot IDs ordered by ranking or insertion order.
        """
        if self.ranked_list:
            return [snapshot.id for snapshot in self.ranked_list.get_items()]
        return self.insertion_order.copy()

    def get_ids_by_insertion_order(self):
        """
        Retrieve a list of snapshot IDs in the order they were added.

        This method returns the snapshot IDs in the exact sequence they were created or added
        to the manager. Unlike ranked snapshots, this order is based purely on the creation
        or insertion order, unaffected by any cmp or ranking logic.

        Returns:
            list[str]: A list of snapshot IDs ordered by their insertion sequence.
        """
        return self.insertion_order.copy()

    def has_snapshot(self, snapshot_id):
        """
        Check if a snapshot exists in storage.

        Args:
            snapshot_id (str): The ID of the snapshot to check.

        Returns:
            bool: True if the snapshot exists, False otherwise.
        """
        return snapshot_id in self.snapshots

    def resort(self):
        """
        Re-sorts the ranked list of snapshots.

        This method ensures the ranked list is updated and sorted according to the
        current comparator function. If no ranked list is maintained (i.e., cmp is None),
        this method has no effect.

        Returns:
            None
        """
        if self.ranked_list:
            self.ranked_list.sort_items()
