from abc import ABC, abstractmethod


class Query(ABC):
    """
    Abstract base class for all query types.
    """

    @abstractmethod
    def evaluate(self, snapshot):
        """
        Evaluate the query against a single snapshot.

        Args:
            snapshot: The snapshot object to evaluate.

        Returns:
            bool: True if the snapshot satisfies the query, False otherwise.
        """
        pass


class ByMetadataQuery(Query):
    def __init__(self, key, value=None):
        self.key = key
        self.value = value

    def evaluate(self, snapshot):
        if self.key not in snapshot.metadata:
            return False
        return (
            snapshot.metadata[self.key] == self.value
            if self.value is not None
            else True
        )


class ByTagQuery(Query):
    def __init__(self, tag):
        self.tag = tag

    def evaluate(self, snapshot):
        return self.tag in snapshot.tags


class ByTimeRangeQuery(Query):
    """
    Query to filter snapshots based on their timestamp.

    Args:
        start_time (float): Start of the time range (inclusive).
        end_time (float): End of the time range (inclusive).
    """

    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time

    def evaluate(self, snapshot):
        """
        Evaluate whether a snapshot's timestamp falls within the range.

        Args:
            snapshot: The snapshot object to evaluate.

        Returns:
            bool: True if the snapshot's timestamp is within the range, False otherwise.
        """
        return self.start_time <= snapshot.timestamp <= self.end_time

class ByContentQuery(Query):
    """
    Query to filter snapshots based on their content.

    Args:
        query_func (callable): A function that takes a snapshot's content (PyTree) 
                               and returns True if the snapshot matches the query.
    """

    def __init__(self, query_func):
        if not callable(query_func):
            raise ValueError("query_func must be callable.")
        self.query_func = query_func

    def evaluate(self, snapshot):
        """
        Evaluate whether a snapshot's content satisfies the custom query function.

        Args:
            snapshot: The snapshot object to evaluate.

        Returns:
            bool: True if the snapshot's content matches the query function, False otherwise.
        """
        return self.query_func(snapshot.get_pytree())