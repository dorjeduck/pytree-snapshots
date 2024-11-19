import uuid

from .snapshot import Snapshot
from .snapshot_storage import SnapshotStorage
from .snapshot_persistence import SnapshotPersistence
from .query import SnapshotQueryInterface, SnapshotQuery

from snapshot_manager.constants import DEFAULT


class SnapshotManager:
    """
    A manager for storing and managing data snapshots.
    """

    def __init__(
        self,
        deepcopy_on_save=True,
        deepcopy_on_retrieve=True,
        query_class=None,
        max_snapshots=None,
        cmp_function=None,
    ):
        """
        Initialize the SnapshotManager.

        Args:
            deepcopy_on_save (bool): Whether to deepcopy PyTrees when saving snapshots. Defaults to True.
            deepcopy_on_retrieve (bool): Whether to return deep copies of PyTrees when retrieving snapshots. Defaults to True.
            query_class (type, optional): A class implementing SnapshotQueryInterface. Defaults to SnapshotQuery.
            max_snapshots (int, optional): Maximum number of snapshots to store. Defaults to None (no limit).
            cmp_function (callable, optional): A comparison function to order snapshots, also used to decide which snapshot to remove
                                            when the storage limit is reached. Defaults to None.
        """
        if query_class and not issubclass(query_class, SnapshotQueryInterface):
            raise TypeError("query_class must implement SnapshotQueryInterface.")

        self.storage = SnapshotStorage(
            max_snapshots=max_snapshots, cmp_function=cmp_function
        )
        self.query = (query_class or SnapshotQuery)(self.storage.snapshots)
        self.deepcopy_on_save = deepcopy_on_save
        self.deepcopy_on_retrieve = deepcopy_on_retrieve

    def __getitem__(self, index, deepcopy=DEFAULT):
        """
        Retrieve a Snapshot by index or ID.

        Args:
            index (int or str):
                - If an integer, retrieves the Snapshot by its position in the order of creation.
                - If a string, retrieves the Snapshot by its ID.
            deepcopy (bool, optional): Whether to return a deep copy of the Snapshot's data. Defaults to the manager's deepcopy setting.

        Returns:
            The data of the Snapshot.

        Raises:
            ValueError: If the index or ID is invalid.
        """
        if isinstance(index, int):
            # Get snapshot ID by index using the storage layer
            try:
                snapshot_id = self.storage.get_snapshot_id_by_index(index)
            except IndexError:
                raise ValueError(f"Index '{index}' is out of range.")
        elif isinstance(index, str):
            # Verify the ID exists
            if not self.storage.has_snapshot(index):
                raise ValueError(f"Snapshot ID '{index}' does not exist.")
            snapshot_id = index
        else:
            raise ValueError(f"Invalid index type: {type(index)}. Must be int or str.")

        # Retrieve the snapshot and return its data
        snapshot = self.storage.get_snapshot(snapshot_id)

        return snapshot.get_data(self._should_deepcopy_on_retrieve(deepcopy))

    # Save, Retrieve, and Delete PytreeSnapshots

    def save_snapshot(
        self,
        data,
        snapshot_id=None,
        metadata=None,
        tags=None,
        overwrite=False,
        deepcopy=DEFAULT,
    ):
        """
        Save a new Snapshot or overwrite an existing one.

        Args:
            data: The data to store in the Snapshot.
            snapshot_id (str, optional): Identifier for the Snapshot. A unique ID is generated if not provided.
            metadata (dict, optional): Metadata to associate with the snapshot.
            tags (list, optional): Tags to associate with the snapshot.
            overwrite (bool): Whether to overwrite an existing snapshot.
            deepcopy (bool, optional): Whether to override the default deepcopy_on_save setting.


        Returns:
            str or bool: The ID of the saved snapshot if successfully added, otherwise False.
        """
        snapshot_id = snapshot_id or str(uuid.uuid4())

        deepcopy = deepcopy if deepcopy is not DEFAULT else self.deepcopy_on_save

        snapshot = Snapshot(data, metadata, tags, deepcopy=deepcopy)

        added = self.storage.add_snapshot(snapshot_id, snapshot, overwrite=overwrite)

        if not added:
            # Snapshot was not added due to space limit and comparison logic
            return False

        return snapshot_id

    def get_snapshot(self, snapshot_id, deepcopy=DEFAULT):
        """
        Retrieve the data from a Snapshot.

        Args:
            snapshot_id (str): The ID of the snapshot to retrieve.
            deepcopy (bool, optional): Whether to return a deep copy of the snapshot's data.

        Returns:
            The data stored in the snapshot.

        Raises:
            ValueError: If the snapshot ID does not exist.
        """

        snapshot = self.storage.get_snapshot(snapshot_id)
        return snapshot.get_data(self._should_deepcopy_on_retrieve(deepcopy))

    def get_latest_snapshot(self, deepcopy=DEFAULT):
        """
        Retrieve the most recent snapshot.

        Args:
            deepcopy (bool, optional): Whether to return a deep copy.

        Returns:
            The data of the most recent snapshot.

        Raises:
            IndexError: If no snapshots are available.
        """
        # Use storage to get the latest snapshot
        snapshot_id = self.storage.snapshot_order[0]
        snapshot = self.storage.get_snapshot(snapshot_id)
        return snapshot.get_data(self._should_deepcopy_on_retrieve(deepcopy))

    def get_ranked_snapshots(self):
        """
        Retrieve the list of snapshot IDs ordered by the comparison function if defined,
        otherwise by their age (insertion order).

        Returns:
            list: Ordered list of snapshot IDs.

        Examples:
            Get snapshots ranked by a custom comparison:
                ranked_snapshots = manager.get_ranked_snapshots()
        """

        return self.storage.get_ranked_snapshots()

    def remove_snapshot(self, snapshot_id):
        """
        Delete a snapshot by its ID.

        Args:
            snapshot_id (str): The ID of the snapshot to delete.

        Returns:
            bool: True if the snapshot was deleted, False otherwise.
        """
        return self.storage.remove_snapshot(snapshot_id)

    def clone_snapshot(self, snapshot_id, metadata=None):
        """
        Clone a snapshot with an optional update to metadata.

        Args:
            snapshot_id (str): The ID of the snapshot to clone.
            metadata (dict, optional): Metadata to add or update in the cloned snapshot.

        Returns:
            str: The ID of the cloned snapshot.

        Raises:
            ValueError: If the snapshot ID does not exist.
        """
        original_snapshot = self.storage.get_snapshot(snapshot_id)

        # Create a deep copy of the original data
        cloned_pytree = original_snapshot.get_data(True)

        # Merge metadata (if provided) with the original snapshot's metadata
        cloned_metadata = original_snapshot.metadata.copy()
        if metadata:
            cloned_metadata.update(metadata)

        # Save the cloned snapshot with a new ID
        cloned_snapshot_id = str(uuid.uuid4())
        self.save_snapshot(
            cloned_pytree, snapshot_id=cloned_snapshot_id, metadata=cloned_metadata
        )

        return cloned_snapshot_id
        # Metadata Management

    def get_metadata(self, snapshot_id):
        """
        Retrieve metadata for a specific Snapshot.

        Args:
            snapshot_id (str): The ID of the Snapshot.

        Returns:
            dict: The metadata associated with the Snapshot.

        Raises:
            ValueError: If the Snapshot ID does not exist.

        Examples:
            Retrieve metadata:
                metadata = manager.get_metadata("snap_id")
        """
        snapshot = self.storage.get_snapshot(snapshot_id)
        return snapshot.metadata

    def update_metadata(self, snapshot_id, new_metadata):
        """
        Update the metadata for a specific Snapshot.

        Args:
            snapshot_id (str): The ID of the Snapshot to update.
            new_metadata (dict): The new metadata to merge with the existing metadata.

        Raises:
            ValueError: If the Snapshot ID does not exist.

        Examples:
            Update metadata:
                manager.update_metadata("snap_id", {"new_key": "new_value"})
        """
        if not isinstance(new_metadata, dict):
            raise TypeError("new_metadata must be a dictionary.")

        snapshot = self.storage.get_snapshot(snapshot_id)
        snapshot.metadata.update(new_metadata)

    # Tag Management

    def add_tags(self, snapshot_id, tags):
        """
        Add tags to a Snapshot.

        Args:
            snapshot_id (str): The ID of the Snapshot.
            tags (list): Tags to add.

        Raises:
            ValueError: If the Snapshot ID does not exist.
        """
        if not isinstance(tags, list):
            raise TypeError("tags must be a list.")

        snapshot = self.storage.get_snapshot(snapshot_id)
        snapshot.add_tags(tags)

    def get_tags(self, snapshot_id):
        """
        Retrieve tags for a specific Snapshot.

        Args:
            snapshot_id (str): The ID of the Snapshot.

        Returns:
            list: The tags associated with the Snapshot.

        Raises:
            ValueError: If the Snapshot ID does not exist.

        Examples:
            Retrieve tags:
                tags = manager.get_tags("snapshot_id")
        """
        snapshot = self.storage.get_snapshot(snapshot_id)
        return snapshot.tags

    def remove_tags(self, snapshot_id, tags):
        """
        Remove tags from a Snapshot.

        Args:
            snapshot_id (str): The ID of the Snapshot.
            tags (list): Tags to remove.

        Raises:
            ValueError: If the Snapshot ID does not exist.

        Examples:
            Remove tags from a Snapshot:
                manager.remove_tags("snap_id", ["obsolete"])
        """
        snapshot = self.storage.get_snapshot(snapshot_id)
        snapshot.remove_tags(tags)

    # Querying and Listing PytreeSnapshots

    def list_snapshots(self):
        """
        List all Snapshot IDs in the order they were created.

        Returns:
            list: Snapshot IDs.

        Examples:
            List all Snapshot IDs:
                ids = manager.list_snapshots()
        """
        return self.storage.snapshot_order

    def get_snapshot_count(self):
        """
        Return the current number of PytreeSnapshots.

        Returns:
            int: The number of stored PytreeSnapshots.

        Examples:
            Get the number of PytreeSnapshots:
                count = manager.get_snapshot_count()
        """
        return len(self.storage.snapshots)

    def get_snapshot_by_index(self, index, deepcopy=DEFAULT):
        """
        Retrieve a Snapshot by its creation index.

        Args:
            index (int): Index of the Snapshot in the creation order.
            deepcopy (bool, optional): Whether to return a deep copy of the data.

        Returns:
            The data of the Snapshot at the specified index.

        Raises:
            IndexError: If the index is out of range.

        Examples:
            Retrieve a Snapshot by index:
                data = manager.get_snapshot_by_index(0)
        """
        if index < 0 or index >= len(self.snapshot_order):
            raise IndexError(f"Index '{index}' is out of range.")
        snapshot_id = self.snapshot_order[index]
        snapshot = self.storage.get_snapshot(snapshot_id)
        return snapshot.get_data(self._should_deepcopy_on_retrieve(deepcopy))

    def get_oldest_snapshot(self, deepcopy=DEFAULT):
        """
        Retrieve the oldest Snapshot.

        Args:
            deepcopy (bool, optional): Whether to return a deep copy of the data.

        Returns:
            The data of the oldest Snapshot.

        Raises:
            IndexError: If no PytreeSnapshots are available.

        Examples:
            Retrieve the oldest Snapshot:
                data = manager.get_oldest_snapshot()
        """
        if not self.storage.snapshot_order:
            raise IndexError("No PytreeSnapshots available.")

        # Get the oldest snapshot ID from the storage
        snapshot_id = self.storage.snapshot_order[0]
        snapshot = self.storage.get_snapshot(snapshot_id)
        return snapshot.get_data(self._should_deepcopy_on_retrieve(deepcopy))

    def list_snapshots_by_age(self, ascending=True):
        """
        List all Snapshot IDs sorted by their age (creation time).

        Args:
            ascending (bool): If True, lists PytreeSnapshots from oldest to newest. Otherwise, newest to oldest.

        Returns:
            list: Snapshot IDs sorted by age.

        Examples:
            List PytreeSnapshots from newest to oldest:
                ids = manager.list_snapshots_by_age(ascending=False)
        """
        return (
            self.storage.snapshot_order
            if ascending
            else list(reversed(self.storage.snapshot_order))
        )

    def save_state(self, file_path, compress=True):
        """
        Save the current state of the manager to a file.

        Args:
            file_path (str): Path to the file where the state should be saved.
            compress (bool): Whether to compress the saved state.

        Returns:
            None
        """
        SnapshotPersistence.save_state(self, file_path, compress)

    @staticmethod
    def load_state(file_path):
        """
        Load a SnapshotManager state from a file.

        Args:
            file_path (str): Path to the file containing the saved state.

        Returns:
            SnapshotManager: A new manager instance with the loaded state.
        """
        state = SnapshotPersistence.load_state(file_path)

        # Create a new manager with the loaded state
        manager = SnapshotManager(
            max_snapshots=state["max_snapshots"],
            deepcopy_on_save=state["deepcopy_on_save"],
            deepcopy_on_retrieve=state["deepcopy_on_retrieve"],
        )

        # Restore snapshots into the manager's storage
        for snapshot_id, snapshot_data in state["snapshots"].items():
            snapshot = Snapshot.from_dict(snapshot_data)
            manager.storage.add_snapshot(snapshot_id, snapshot)

        return manager

    # private

    def _should_deepcopy_on_retrieve(self, deepcopy):
        return deepcopy if deepcopy is not DEFAULT else self.deepcopy_on_retrieve
