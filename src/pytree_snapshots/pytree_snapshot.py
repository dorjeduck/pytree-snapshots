from jax.tree_util import tree_flatten, tree_map
from pytree_snapshots.snapshot import Snapshot


class PyTreeSnapshot(Snapshot):
    def __init__(self, pytree, metadata=None, tags=None, compress=False):
        """
        Initialize a PyTree-specific Snapshot.

        Args:
            pytree: A valid JAX PyTree.
            metadata (dict, optional): User-defined metadata.
            tags (list, optional): Tags associated with the snapshot.
            compress (bool, optional): Whether to compress the snapshot data.
        """
        self.validate_pytree(pytree)
        super().__init__(pytree, metadata, tags, compress)

    @staticmethod
    def validate_pytree(pytree):
        """
        Validate that the object is a valid JAX PyTree.
        Raises a ValueError if not valid.
        """
        try:
            tree_flatten(pytree)
        except Exception as e:
            raise ValueError(f"Invalid PyTree: {e}")

    def apply_leaf_transformation(self, func):
        """
        Apply a transformation function to each leaf of the PyTree.

        Args:
            func (callable): A function to apply to each leaf of the PyTree.

        Returns:
            A new PyTree with the transformation applied.
        """
        pytree = self.get_data(deepcopy=False)
        return tree_map(func, pytree)
