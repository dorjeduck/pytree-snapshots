from abc import ABC, abstractmethod
from .snapshot_query_interface import SnapshotQueryInterface


class PyTreeSnapshotQueryInterface(SnapshotQueryInterface, ABC):
    """
    Abstract base class for PyTree-specific snapshot query implementations.
    Extends SnapshotQueryInterface with methods specific to PyTree queries.
    """

    @abstractmethod
    def by_leaf_value(self, condition):
        """
        Query snapshots based on a condition applied to PyTree leaves.

        Args:
            condition (callable): A function that takes a leaf value and returns True if it matches.

        Returns:
            list: Snapshot IDs matching the criteria.
        """
        pass
