from .snapshot_query_interface import SnapshotQueryInterface
from .base_queries import (
    Query,
    ByMetadataQuery,
    ByTagQuery,
    ByTimeRangeQuery,
    ByContentQuery,
)


class SnapshotQuery(SnapshotQueryInterface):
    """
    Handles querying snapshots, including simple and logical queries.
    """

    def __init__(self, snapshots):
        """
        Initialize the SnapshotQuery object.

        Args:
            snapshots (dict): A dictionary of snapshot_id to snapshot objects.
        """
        self.snapshots = snapshots

    def by_metadata(self, key, value=None):
        """
        Find snapshots by metadata key and optionally a specific value.

        Args:
            key (str): Metadata key to search for.
            value (optional): Value to match for the given key.

        Returns:
            list: Snapshot IDs matching the criteria.
        """
        query = ByMetadataQuery(key, value)
        return self.evaluate(query)

    def by_tag(self, tag):
        """
        Find snapshots by a specific tag.

        Args:
            tag (str): Tag to search for.

        Returns:
            list: Snapshot IDs matching the criteria.
        """
        query = ByTagQuery(tag)
        return self.evaluate(query)

    def by_time_range(self, start_time, end_time):
        """
        Find snapshots within a specific time range.

        Args:
            start_time (float): Start of the time range (UNIX timestamp).
            end_time (float): End of the time range (UNIX timestamp).

        Returns:
            list: Snapshot IDs matching the criteria.
        """
        query = ByTimeRangeQuery(start_time, end_time)
        return self.evaluate(query)

    def by_comparator(self, comparator):
        """
        Find a snapshot by comparing all snapshots using a custom comparator.

        Args:
            comparator (callable): A function that takes two snapshots and returns
                True if the first snapshot is "better" than the second.

        Returns:
            str: The ID of the snapshot that satisfies the criterion, or None if no snapshots exist.
        """
        selected_snapshot_id = None

        for snapshot_id, snapshot in self.snapshots.items():
            if selected_snapshot_id is None or comparator(
                snapshot, self.snapshots[selected_snapshot_id]
            ):
                selected_snapshot_id = snapshot_id

        return selected_snapshot_id

    def by_content(self, query_func):
        """
        Find snapshots based on their content using a custom query function.

        Args:
            query_func (callable): A function that takes a snapshot's content (PyTree)
                                   and returns True if the snapshot matches the query.

        Returns:
            list: Snapshot IDs matching the criteria.
        """
        if not callable(query_func):
            raise ValueError("query_func must be a callable function.")

        query = ByContentQuery(query_func)
        return self.evaluate(query)

    def evaluate(self, query):
        """
        Evaluate a query against all snapshots.

        Args:
            query (Query): The query object to evaluate.

        Returns:
            list: Snapshot IDs matching the criteria.
        """
        return [
            snapshot_id
            for snapshot_id, snapshot in self.snapshots.items()
            if query.evaluate(snapshot)
        ]
