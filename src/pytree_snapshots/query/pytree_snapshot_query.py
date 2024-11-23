from jax.tree_util import tree_flatten

from .snapshot_query import SnapshotQuery
from .pytree_snapshot_query_interface import PyTreeSnapshotQueryInterface
from .pytree_queries import ByLeafQuery
from ..pytree_snapshot import PyTreeSnapshot


class PyTreeSnapshotQuery(SnapshotQuery, PyTreeSnapshotQueryInterface):
    """
    Extends SnapshotQuery with PyTree-specific query capabilities.
    """

    def by_leaf_value(self, condition):
        """
        Create a query object to match snapshots based on a condition applied to PyTree leaves.

        Args:
            condition (callable): A function that takes a leaf value and returns True or False.

        Returns:
            PyTreeLeafQuery: A query object to evaluate the condition.
        """
        return ByLeafQuery(condition)

    def _leaf_matches(self, snapshot, condition):
        """
        Check if any leaf in the snapshot's PyTree matches the condition.

        Args:
            snapshot (Snapshot): The snapshot to evaluate.
            condition (callable): The condition to check for each leaf.

        Returns:
            bool: True if any leaf matches, False otherwise.
        """
        if not isinstance(snapshot, PyTreeSnapshot):
            return False
        pytree = snapshot.get_data(deepcopy=False)
        leaves, _ = tree_flatten(pytree)
        return any(condition(leaf) for leaf in leaves)
