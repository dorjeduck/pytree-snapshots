from abc import ABC, abstractmethod


class SnapshotQueryInterface(ABC):
    """
    Abstract base class for snapshot query implementations.
    """

    @abstractmethod
    def by_metadata(self, key, value=None):
        """
        Find snapshots by metadata key and optionally a specific value.

        Args:
            key (str): Metadata key to search for.
            value (optional): Value to match for the given key.

        Returns:
            list: Snapshot IDs matching the criteria.
        """
        pass

    @abstractmethod
    def by_tag(self, tag):
        """
        Find snapshots by a specific tag.

        Args:
            tag (str): Tag to search for.

        Returns:
            list: Snapshot IDs matching the tag.
        """
        pass

    @abstractmethod
    def by_time_range(self, start_time, end_time):
        """
        Find snapshots within a specific time range.

        Args:
            start_time (float): Start of the time range (UNIX timestamp).
            end_time (float): End of the time range (UNIX timestamp).

        Returns:
            list: Snapshot IDs matching the criteria.
        """
        pass

    @abstractmethod
    def by_comparator(self, comparator):
        """
        Find a snapshot using a custom comparator.

        Args:
            comparator (callable): A function to compare snapshots.

        Returns:
            str: The ID of the snapshot that satisfies the comparator.
        """
        pass

    @abstractmethod
    def by_content(self, query_func):
        """
        Find snapshots by matching a condition in their content.

        Args:
            query_func (callable): A function that takes a PyTree and returns True if it matches the query.

        Returns:
            list: Snapshot IDs that match the condition.
        """
        pass

    @abstractmethod
    def evaluate(self, query):
        """
        Evaluate a complex query against all snapshots.

        Args:
            query: A query object to evaluate.

        Returns:
            list: Snapshot IDs matching the criteria.
        """
        pass
