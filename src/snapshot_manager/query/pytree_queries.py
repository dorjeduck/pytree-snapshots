from jax.tree_util import tree_flatten
from snapshot_manager.pytree_snapshot import PyTreeSnapshot
from .base_queries import Query

class ByLeafQuery(Query):
    def __init__(self, condition):
        self.condition = condition

    def evaluate(self, snapshot):
        if not isinstance(snapshot, PyTreeSnapshot):
            return False
        pytree = snapshot.get_data(deepcopy=False)
        leaves, _ = tree_flatten(pytree)
        return any(self.condition(leaf) for leaf in leaves)