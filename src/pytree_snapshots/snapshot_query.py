from .snapshot_query_interface import SnapshotQueryInterface


class SnapshotQuery(SnapshotQueryInterface):
    def __init__(self, snapshots):
        """
        Initialize the query class.

        Args:
            snapshots (dict): A dictionary of snapshot_id to snapshot objects.
        """
        self.snapshots = snapshots

    def by_comparator(self, comparator):
        """
        Find a snapshot by comparing all snapshots using a custom comparator.

        Args:
            comparator (callable): A function that takes two snapshots and returns
                True if the first snapshot is "better" than the second.

        Returns:
            str: The ID of the snapshot that satisfies the criterion, or None if no snapshots exist.

        Raises:
            ValueError: If the comparator is not callable.

        Examples:
            Find the snapshot with the highest accuracy:
                selected_snapshot_id = query.by_comparator(
                    lambda s1, s2: s1.metadata["accuracy"] > s2.metadata["accuracy"]
                )
        """
        if not callable(comparator):
            raise ValueError("The comparator must be a callable function.")

        if not self.snapshots:
            return None  # No snapshots to compare

        selected_snapshot_id, selected_snapshot = None, None

        for snapshot_id, snapshot in self.snapshots.items():
            if selected_snapshot is None or comparator(snapshot, selected_snapshot):
                selected_snapshot_id = snapshot_id
                selected_snapshot = snapshot

        return selected_snapshot_id

    def by_metadata(self, key, value=None):
        return [
            snapshot_id
            for snapshot_id, snapshot in self.snapshots.items()
            if key in snapshot.metadata
            and (value is None or snapshot.metadata[key] == value)
        ]

    def by_tag(self, tag):
        return [
            snapshot_id
            for snapshot_id, snapshot in self.snapshots.items()
            if tag in snapshot.tags
        ]

    def by_time_range(self, start_time, end_time):
        return [
            snapshot_id
            for snapshot_id, snapshot in self.snapshots.items()
            if start_time <= snapshot.timestamp <= end_time
        ]

    def by_content(self, query_func):
        return [
            snapshot_id
            for snapshot_id, snapshot in self.snapshots.items()
            if query_func(snapshot.get_pytree())
        ]
