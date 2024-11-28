import uuid
import jax

from .snapshot_persistence import SnapshotPersistence
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

    def tree_map(self, func, snapshot_ids=None, is_leaf=None):
        """
        Apply a transformation function to the PyTree(s) of specified snapshots.

        Args:
            func (callable): A function to apply to each leaf of the PyTree.
                            The function should take a single argument (leaf) and return the transformed leaf.
            snapshot_ids (str, list, or None, optional): A single snapshot ID, a list of snapshot IDs, or None.
                                                        If None, all snapshots in storage will be affected.
            is_leaf (Callable[[Any], bool], optional): A function to determine custom leaf nodes. Defaults to None.

        Returns:
            PyTree or list: A transformed PyTree if a single snapshot_id is provided,
                            a list of transformed PyTrees otherwise.

        Raises:
            TypeError: If a snapshot is not a PyTreeSnapshot.
            ValueError: If no valid snapshots are found for the given snapshot_ids.
        """
        # Normalize snapshot_ids
        if isinstance(snapshot_ids, str):  # Single snapshot ID
            snapshot_ids = [snapshot_ids]
            single_return = True
        elif snapshot_ids is None:  # All snapshots
            snapshot_ids = self.storage.insertion_order
            single_return = False
        else:  # List of snapshot IDs
            single_return = False

        if not snapshot_ids:
            raise ValueError("No snapshot IDs provided or available.")

        transformed_pytrees = []

        for snap_id in snapshot_ids:
            # Retrieve the snapshot
            snapshot = self.storage.get_snapshot(snap_id)
            if not isinstance(snapshot, PyTreeSnapshot):
                raise TypeError(f"Snapshot {snap_id} is not a PyTreeSnapshot.")

            # Apply the tree_map transformation
            transformed_pytree = jax.tree_util.tree_map(func, snapshot.data, is_leaf=is_leaf)
            transformed_pytrees.append(transformed_pytree)

        if single_return:
            return transformed_pytrees[0]  # Return a single PyTree if only one snapshot ID was provided
        return transformed_pytrees  # Return a list of PyTrees otherwise

    def tree_replace(self, func, snapshot_ids=None, is_leaf=None):
        """
        Apply a transformation function to the PyTree(s) of specified snapshots and replace them in the storage.

        Args:
            func (callable): A function to apply to each leaf of the PyTree.
                            The function should take a single argument (leaf) and return the transformed leaf.
            snapshot_ids (str, list, or None, optional): A single snapshot ID, a list of snapshot IDs, or None.
                                                        If None, all snapshots in storage will be affected.
            is_leaf (Callable[[Any], bool], optional): A function to determine custom leaf nodes. Defaults to None.

        Raises:
            TypeError: If a snapshot is not a PyTreeSnapshot.

        Examples:
            Replace all snapshots with a transformation applied:
                manager.tree_replace(func=lambda x: x + 1)

            Replace a specific snapshot:
                manager.tree_replace(
                    func=lambda x: x * 2,
                    snapshot_ids="snapshot123"
                )

            Replace multiple snapshots with custom leaf detection:
                manager.tree_replace(
                    func=lambda x: x.lower(),
                    snapshot_ids=["snapshot123", "snapshot456"],
                    is_leaf=lambda x: isinstance(x, str)
                )
        """
        # Normalize snapshot_ids into a list
        if isinstance(snapshot_ids, str):
            snapshot_ids = [snapshot_ids]
        elif snapshot_ids is None:
            snapshot_ids = self.storage.insertion_order

        for snap_id in snapshot_ids:
            # Retrieve the snapshot
            snapshot = self.storage.get_snapshot(snap_id)
            if not isinstance(snapshot, PyTreeSnapshot):
                raise TypeError(f"Snapshot {snap_id} is not a PyTreeSnapshot.")

            # Replace the PyTree in the existing snapshot
            snapshot.data = jax.tree_util.tree_map(func, snapshot.data, is_leaf=is_leaf)

            # Update the snapshot in storage
            self.storage.add_snapshot(snapshot=snapshot, overwrite=True)

    def tree_combine(self, snapshot_ids=None, combine_fn=None):
        """
        Combine multiple snapshots' PyTrees into a single PyTree using a custom function,
        and returns it.

        Args:
            snapshot_ids (list or None, optional): A list of IDs of the snapshots to combine.
                                                If None, all snapshots in storage are used.
            combine_fn (callable): A function to combine corresponding leaves of the PyTrees.
                                This function takes a list of values (one from each snapshot)
                                and returns a single combined value.

        Returns:
            PyTree: returns the combined PyTree

        Raises:
            ValueError: If `combine_fn` is not provided.
            TypeError: If the snapshots are not PyTreeSnapshots or their structures do not match.

        Examples:
            Combine all snapshot pytrees with an addition operation:
                combined_pytree = manager.tree_combine(
                    combine_fn=lambda leaves: sum(leaves)
                )

            Combine specific snapshot pytrees:
                combined_pytree = manager.tree_combine(
                    snapshot_ids=["snapshot1", "snapshot2"],
                    combine_fn=lambda leaves: max(leaves)
                )

        """

        if combine_fn is None:
            raise ValueError("A combine_fn must be provided to combine PyTrees.")

        # Select snapshots to combine
        snapshot_ids = snapshot_ids or self.storage.insertion_order
        pytree_list = [
            self.storage.get_snapshot(snapshot_id).data for snapshot_id in snapshot_ids
        ]

        # Ensure all PyTrees have the same structure
        flattened, structure = zip(
            *[jax.tree_util.tree_flatten(pytree) for pytree in pytree_list]
        )
        if not all(structure[0] == s for s in structure):
            raise TypeError("All snapshots must have the same PyTree structure.")

        # Apply the combine function to each set of corresponding leaves
        combined_leaves = [combine_fn(leaves) for leaves in zip(*flattened)]

        # return the combined pytree
        return jax.tree_util.tree_unflatten(structure[0], combined_leaves)

    @staticmethod
    def load_from_file(file_path):
        """
        Load a PyTreeSnapshotManager state from a file.

        Args:
            file_path (str): The file path of the saved state to load.

        Returns:
            PyTreeSnapshotManager: An instance of `PyTreeSnapshotManager` with the loaded state.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            IOError: If there is an error reading the file.
            ValueError: If the file contents cannot be deserialized into the expected state.

        Examples:
            Load a manager state from a saved file:
                manager = PyTreeSnapshotManager.load_from_file("snapshots_state.json")

            Access snapshots from the loaded manager:
                snapshot = manager.get_snapshot("snapshot_id")
        """
        return SnapshotPersistence.load_from_file(
            file_path, PyTreeSnapshotManager, PyTreeSnapshot
        )

    def _create_snapshot(self, data, metadata, tags, deepcopy, snapshot_id):
        """
        Override the factory method to create a PyTreeSnapshot.

        Args:
            data: The data to store in the PyTreeSnapshot.
            metadata (dict): Metadata to associate with the snapshot.
            tags (list): Tags to associate with the snapshot.
            deepcopy (bool): Whether to deepcopy the data.
            snapshot_id (str): Identifier for the Snapshot.

        Returns:
            PyTreeSnapshot: The created PyTreeSnapshot instance.
        """
        return PyTreeSnapshot(
            data,
            metadata=metadata,
            tags=tags,
            deepcopy=deepcopy,
            snapshot_id=snapshot_id,
        )
