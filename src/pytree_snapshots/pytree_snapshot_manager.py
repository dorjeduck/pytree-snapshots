import uuid
import jax

from .snapshot_manager import SnapshotManager
from .pytree_snapshot import PyTreeSnapshot
from .query import PyTreeSnapshotQuery, PyTreeSnapshotQueryInterface

from .constants import DEFAULT


class PyTreeSnapshotManager(SnapshotManager):

    def __init__(self, *args, query_class=None, **kwargs):
        """
        Initialize the PyTreeSnapshotManager.

        Args:
            query_class (type, optional): A class implementing PyTreeSnapshotQueryInterface.
                                           Defaults to PyTreeSnapshotQuery.
        """
        # Default to PyTreeSnapshotQuery if no query class is provided
        query_class = query_class or PyTreeSnapshotQuery

        # Validate that the provided query class implements PyTreeSnapshotQueryInterface
        if not issubclass(query_class, PyTreeSnapshotQueryInterface):
            raise TypeError(
                f"The query_class must implement PyTreeSnapshotQueryInterface. "
                f"Received: {query_class.__name__}"
            )

        super().__init__(*args, query_class=query_class, **kwargs)

    def save_snapshot(
        self,
        pytree,
        snapshot_id=None,
        metadata=None,
        tags=None,
        overwrite=False,
    ):
        """
        Save a PyTree snapshot.

        Args:
            pytree: The PyTree to save.
            snapshot_id (str, optional): A unique identifier for the snapshot. If None, a new UUID is generated.
            metadata (dict, optional): Metadata to associate with the snapshot.
            tags (list, optional): Tags to associate with the snapshot.
            overwrite (bool): Whether to overwrite an existing snapshot.

        Returns:
            str: The ID of the saved snapshot.
        """
        snapshot_id = snapshot_id or str(uuid.uuid4())

        snapshot = PyTreeSnapshot(pytree, metadata, tags)
        self.storage.add_snapshot(snapshot_id, snapshot, overwrite=overwrite)
        return snapshot_id

    def update_leaf_nodes(self, snapshot_ids, func):
        """
        Apply a transformation function to one or more snapshots' PyTrees.

        Args:
            snapshot_ids (str or list): The ID(s) of the snapshot(s) to transform.
            func (callable): A function to apply to each leaf of the PyTree.

        Returns:
            Transformed PyTree or list of transformed PyTrees.
        """

        if isinstance(snapshot_ids, str):
            # Single snapshot case
            snapshot = self.storage.get_snapshot(snapshot_ids)
            if not isinstance(snapshot, PyTreeSnapshot):
                raise TypeError("Snapshot is not a PyTreeSnapshot.")
            snapshot.update_leaf_nodes(func)

        elif isinstance(snapshot_ids, list):
            # Multiple snapshots case
            for snapshot_id in snapshot_ids:
                snapshot = self.storage.get_snapshot(snapshot_id)
                if not isinstance(snapshot, PyTreeSnapshot):
                    raise TypeError(
                        f"Snapshot with ID {snapshot_id} is not a PyTreeSnapshot."
                    )
                snapshot.update_leaf_nodes(func)

        else:
            raise TypeError(
                "snapshot_ids must be a string (single ID) or a list of strings (multiple IDs)."
            )

    def update_all_leaf_nodes(self, func):
        """
        Apply a transformation function to all snapshots' PyTrees currently managed.

        Args:
            func (callable): A function to apply to each leaf of all PyTrees.

        Raises:
            TypeError: If any snapshot is not a PyTreeSnapshot.
        """
        for snapshot_id in self.storage.snapshots.keys():
            snapshot = self.storage.get_snapshot(snapshot_id)
            if not isinstance(snapshot, PyTreeSnapshot):
                raise TypeError(
                    f"Snapshot with ID {snapshot_id} is not a PyTreeSnapshot."
                )
            snapshot.update_leaf_nodes(func)

    def combine_snapshots(self, snapshot_ids=None, combine_fn=None):
        """
        Combine multiple snapshots into a single PyTree using a custom function.

        Args:
            snapshot_ids (list, optional): IDs of the snapshots to combine.
                                           If None, combine all snapshots.
            combine_fn (callable): A function that takes a list of leaves (one from each snapshot)
                                   and returns a single combined leaf.

        Returns:
            Combined PyTree with the same structure as the snapshots.

        Raises:
            TypeError: If snapshots are not PyTreeSnapshots or structures do not match.
            ValueError: If combine_fn is not provided.
        """
        if combine_fn is None:
            raise ValueError("A combine_fn must be provided to combine snapshots.")

        # Select snapshots to combine
        snapshot_ids = snapshot_ids or self.storage.snapshot_order
        pytree_list = [
            self.storage.get_snapshot(snapshot_id).data
            for snapshot_id in snapshot_ids
        ]

        # Ensure all PyTrees have the same structure
        flattened, structure = zip(
            *[jax.tree.flatten(pytree) for pytree in pytree_list]
        )
        if not all(structure[0] == s for s in structure):
            raise TypeError("All snapshots must have the same PyTree structure.")

        # Apply the combine function to each set of corresponding leaves
        combined_leaves = [combine_fn(leaves) for leaves in zip(*flattened)]

        # Reconstruct the combined PyTree
        combined_pytree = jax.tree.unflatten(structure[0], combined_leaves)
        return combined_pytree
