import jax
from snapshot_manager.snapshot import Snapshot


class PyTreeSnapshot(Snapshot):
    def __init__(self, pytree, metadata=None, tags=None):
        """
        Initialize a PyTree-specific Snapshot.

        Args:
            pytree: A valid JAX PyTree.
            metadata (dict, optional): User-defined metadata.
            tags (list, optional): Tags associated with the snapshot.
        """
        self.validate_pytree(pytree)
        super().__init__(pytree, metadata, tags)

    @staticmethod
    def validate_pytree(pytree):
        """
        Validate that the object is a valid JAX PyTree.
        Raises a ValueError if not valid.
        """
        try:
            jax.tree.flatten(pytree)
        except Exception as e:
            raise ValueError(f"Invalid PyTree: {e}")

    def update_leaf_nodes(self, func):
        """
        Apply a transformation function to each leaf of the PyTree.

        Args:
            func (callable): A function to apply to each leaf of the PyTree.
        """
        self.data = jax.tree.map(func, self.data)
