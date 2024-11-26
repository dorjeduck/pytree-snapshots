import copy
import jax
from snapshot_manager.snapshot import Snapshot


class PyTreeSnapshot(Snapshot):
    def __init__(
        self, pytree, metadata=None, tags=None, deepcopy=True, snapshot_id=None
    ):
        """
        Initialize a PyTree-specific Snapshot.

        Args:
            pytree: A valid JAX PyTree.
            metadata (dict, optional): User-defined metadata.
            tags (list, optional): Tags associated with the snapshot.
        """
        self.validate_pytree(pytree)
        super().__init__(pytree, metadata, tags, deepcopy, snapshot_id)

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

    def tree_map(self, func, is_leaf=None, new_id=None):
        """
        Apply a transformation function to each leaf of the PyTree and return a new PyTreeSnapshot.

        Args:
            func (callable): A function to apply to each leaf of the PyTree.
            is_leaf (Callable[[Any], bool], optional): Function to determine custom leaf nodes.
            new_id (str, optional): Optional custom ID for the new snapshot.

        Returns:
            PyTreeSnapshot: A new snapshot with the transformed PyTree.
        """
        # Transform the PyTree
        transformed_pytree = jax.tree.map(func, self.data, is_leaf=is_leaf)

        # Create a new PyTreeSnapshot with the transformed data
        new_snapshot = PyTreeSnapshot(
            transformed_pytree,
            metadata=copy.deepcopy(self.metadata),
            tags=copy.deepcopy(self.tags),
            deepcopy=False,
            snapshot_id=new_id,
        )

        return new_snapshot
