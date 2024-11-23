import uuid

from pytree_snapshots.snapshot_manager import SnapshotManager
from pytree_snapshots.pytree_snapshot import PyTreeSnapshot
from pytree_snapshots.constants import DEFAULT


class PyTreeSnapshotManager(SnapshotManager):
    def save_snapshot(
        self,
        pytree,
        snapshot_id=None,
        metadata=None,
        tags=None,
        compress=DEFAULT,
        overwrite=False,
    ):
        """
        Save a PyTree snapshot.

        Args:
            pytree: The PyTree to save.
            snapshot_id (str, optional): A unique identifier for the snapshot. If None, a new UUID is generated.
            metadata (dict, optional): Metadata to associate with the snapshot.
            tags (list, optional): Tags to associate with the snapshot.
            compress (bool): Whether to compress the snapshot data.
            overwrite (bool): Whether to overwrite an existing snapshot.

        Returns:
            str: The ID of the saved snapshot.
        """
        snapshot_id = snapshot_id or str(uuid.uuid4())
        compress = self.compress if compress is DEFAULT else compress
        snapshot = PyTreeSnapshot(pytree, metadata, tags, compress)
        self.storage.add_snapshot(snapshot_id, snapshot, overwrite=overwrite)
        return snapshot_id

    def apply_leaf_transformation(self, snapshot_ids, func, deepcopy=DEFAULT):
        """
        Apply a transformation function to one or more snapshots' PyTrees.

        Args:
            snapshot_ids (str or list): The ID(s) of the snapshot(s) to transform.
            func (callable): A function to apply to each leaf of the PyTree.
            deepcopy (bool): Whether to return a deep copy of the transformed PyTree(s). 
                            Defaults to the manager's deepcopy setting.

        Returns:
            Transformed PyTree or list of transformed PyTrees.
        """
        # Resolve DEFAULT to the manager's deepcopy setting
        deepcopy = self.deepcopy if deepcopy is DEFAULT else deepcopy

        if isinstance(snapshot_ids, str):
            # Single snapshot case
            snapshot = self.storage.get_snapshot(snapshot_ids, deepcopy=deepcopy)
            if not isinstance(snapshot, PyTreeSnapshot):
                raise TypeError("Snapshot is not a PyTreeSnapshot.")
            return snapshot.apply_leaf_transformation(func)

        elif isinstance(snapshot_ids, list):
            # Multiple snapshots case
            transformed_pytree_list = []
            for snapshot_id in snapshot_ids:
                snapshot = self.storage.get_snapshot(snapshot_id, deepcopy=deepcopy)
                if not isinstance(snapshot, PyTreeSnapshot):
                    raise TypeError(f"Snapshot with ID {snapshot_id} is not a PyTreeSnapshot.")
                transformed_pytree_list.append(snapshot.apply_leaf_transformation(func))
            return transformed_pytree_list

        else:
            raise TypeError("snapshot_ids must be a string (single ID) or a list of strings (multiple IDs).")