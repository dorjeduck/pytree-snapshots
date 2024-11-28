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
        cmp=None,
    ):
        """
        Initialize the SnapshotManager.

        Args:
            deepcopy_on_save (bool): Whether to deepcopy PyTrees when saving snapshots. Defaults to True.
            deepcopy_on_retrieve (bool): Whether to return deep copies of PyTrees when retrieving snapshots. Defaults to True.
            query_class (type, optional): A class implementing SnapshotQueryInterface. Defaults to SnapshotQuery.
            max_snapshots (int, optional): Maximum number of snapshots to store. Defaults to None (no limit).
            cmp (callable, optional): A comparison function to order snapshots, also used to decide which snapshot to remove
                                            when the storage limit is reached. Defaults to None.
        """
        if query_class and not issubclass(query_class, SnapshotQueryInterface):
            raise TypeError("query_class must implement SnapshotQueryInterface.")

        self.storage = SnapshotStorage(max_snapshots=max_snapshots, cmp=cmp)
        self.query = (query_class or SnapshotQuery)(self.storage.snapshots)
        self.deepcopy_on_save = deepcopy_on_save
        self.deepcopy_on_retrieve = deepcopy_on_retrieve

    def __getitem__(self, index, deepcopy=DEFAULT):
        """
        Retrieve the data of a Snapshot by its index or ID.

        Args:
            index (int or str):
                - If an integer, retrieves the Snapshot by its position in the creation order.
                - If a string, retrieves the Snapshot by its unique ID.
            deepcopy (bool, optional): If True, returns a deep copy of the Snapshot's data.
                Defaults to the manager's `deepcopy_on_retrieve` setting.

        Returns:
            Any: The data of the requested Snapshot.

        Raises:
            ValueError: If `index` is neither an integer nor a string.
            IndexError: If the index is out of range when `index` is an integer.
            ValueError: If the Snapshot ID does not exist when `index` is a string.
        """

        if isinstance(index, int):
            # Retrieve Snapshot by creation index
            return self.get_snapshot_by_index(index, deepcopy=deepcopy)
        elif isinstance(index, str):
            # Retrieve Snapshot by ID
            return self.get_snapshot(index, deepcopy=deepcopy)
        else:
            raise ValueError(f"Invalid index type: {type(index)}. Must be int or str.")

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
            data (Any): The data to store in the Snapshot. Can be any serializable Python object or PyTree.
            snapshot_id (str, optional): Identifier for the Snapshot. A unique ID is generated if not provided.
            metadata (dict, optional): Additional metadata to associate with the Snapshot. Defaults to None.
            tags (list, optional): Tags to categorize or label the Snapshot. Defaults to None.
            overwrite (bool): Whether to overwrite an existing Snapshot with the same ID. Defaults to False.
            deepcopy (bool, optional): Whether to deepcopy the data before saving. If not specified,
                uses the manager's `deepcopy_on_save` setting.

        Returns:
            str: The ID of the saved Snapshot if successfully added.
            bool: False if the Snapshot was not saved due to storage constraints.

        Raises:
            ValueError: If attempting to save a Snapshot with an existing ID without `overwrite=True`.
            TypeError: If `tags` or `metadata` is not of the correct type.

        Examples:
            Save a simple data structure:
                manager.save_snapshot({"key": "value"}, tags=["example"], metadata={"desc": "test"})

            Overwrite an existing snapshot:
                manager.save_snapshot(data, snapshot_id="existing_id", overwrite=True)
        """

        if tags is not None and not isinstance(tags, list):
            raise TypeError("tags must be a list.")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary.")

        deepcopy = self.deepcopy_on_save if deepcopy is DEFAULT else deepcopy

        # Check if overwriting is not allowed
        if not overwrite and self.storage.has_snapshot(snapshot_id):
            raise ValueError(
                f"Snapshot ID '{snapshot_id}' already exists. Set `overwrite=True` to replace it."
            )

        # Create the snapshot using the factory method
        snapshot = self._create_snapshot(
            data=data,
            metadata=metadata,
            tags=tags,
            deepcopy=deepcopy,
            snapshot_id=snapshot_id,
        )

        # Add the snapshot to storage

        if not self.storage.add_snapshot(snapshot, overwrite=overwrite):
            # Snapshot was not added due to storage constraints / cmp policy
            return False

        return snapshot.id

    def get_snapshot(self, snapshot_id, deepcopy=DEFAULT):
        """
        Retrieve a Snapshot by its unique ID.

        Args:
            snapshot_id (str): The unique identifier of the Snapshot to retrieve.
            deepcopy (bool, optional): Whether to return a deep copy of the Snapshot's data.
                If not specified, defaults to the manager's `deepcopy_on_retrieve` setting.

        Returns:
            Snapshot: The Snapshot object associated with the given ID.

        Raises:
            ValueError: If the Snapshot ID does not exist in the storage.

        Examples:
            Retrieve a Snapshot with a deep copy:
                snapshot = manager.get_snapshot("example_id", deepcopy=True)

            Retrieve a Snapshot without a deep copy:
                snapshot = manager.get_snapshot("example_id", deepcopy=False)
        """
        snapshot = self.storage.get_snapshot(snapshot_id)

        deepcopy = self.deepcopy_on_retrieve if deepcopy is DEFAULT else deepcopy

        return snapshot.clone(snapshot.id) if deepcopy else snapshot

    def get_latest_snapshot(self, deepcopy=DEFAULT):
        """
        Retrieve the most recently added Snapshot.

        Args:
            deepcopy (bool, optional): Whether to return a deep copy of the Snapshot's data.
                If not specified, defaults to the manager's `deepcopy_on_retrieve` setting.

        Returns:
            Snapshot: The most recently added Snapshot object.

        Raises:
            IndexError: If no snapshots are stored.

        Examples:
            Retrieve the latest Snapshot with a deep copy:
                latest_snapshot = manager.get_latest_snapshot(deepcopy=True)

            Retrieve the latest Snapshot without a deep copy:
                latest_snapshot = manager.get_latest_snapshot(deepcopy=False)
        """
        # Use storage to get the latest snapshot

        if not self.storage.insertion_order:
            raise IndexError("No snapshots are available.")

        # Use storage to get the latest snapshot ID
        latest_snapshot_id = self.storage.insertion_order[0]

        # Retrieve the snapshot using the get_snapshot method
        return self.get_snapshot(latest_snapshot_id, deepcopy=deepcopy)

    def get_ids_by_rank(self):
        """
        Retrieve a list of snapshot IDs ranked according to the comparison function.

        If a custom comparison function (`cmp`) was provided during the initialization of
        the manager, the snapshots will be ranked based on that function. If no comparison function
        is defined, the snapshots are ranked by their creation time in insertion order.

        Returns:
            list[str]: A list of snapshot IDs ranked according to the defined criteria.

        Examples:
            Retrieve snapshot IDs ranked by a custom comparison:
                ranked_ids = manager.get_ids_by_rank()

            If no comparison function is defined:
                ranked_ids = manager.get_ids_by_rank()  # Returns IDs in insertion order
        """

        return self.storage.get_ids_by_rank()

    def resort(self):
        """
        Re-sort the snapshots in storage according to the current comparison function if available.

        Returns:
            None
        """
        self.storage.resort()

    def get_ids_by_insertion_order(self):
        """
        Retrieve a list of snapshot IDs in the order they were added.

        This method returns the snapshot IDs in the exact sequence they were created or added
        to the manager. Unlike ranked snapshots, this order is based purely on the creation
        or insertion order, unaffected by any cmp or ranking logic.

        Returns:
            list[str]: A list of snapshot IDs ordered by their insertion sequence.
        """
        return self.storage.get_ids_by_insertion_order()

    def remove_snapshot(self, snapshot_id):
        """
        Remove a Snapshot by its unique ID.

        Args:
            snapshot_id (str): The unique identifier of the Snapshot to remove.

        Returns:
            bool: True if the Snapshot was successfully removed, False if no Snapshot with the
                given ID exists in storage.

        Raises:
            ValueError: If the provided snapshot_id is not a string.

        Examples:
            Remove a Snapshot by ID:
                success = manager.remove_snapshot("example_id")
                if success:
                    print("Snapshot removed.")
                else:
                    print("Snapshot ID not found.")
        """
        if not isinstance(snapshot_id, str):
            raise ValueError(f"Snapshot ID must be a string, got {type(snapshot_id)}.")

        return self.storage.remove_snapshot(snapshot_id)

    def get_metadata(self, snapshot_id):
        """
        Retrieve the metadata associated with a specific Snapshot.

        This method fetches the metadata stored alongside the Snapshot identified by `snapshot_id`.
        Metadata typically contains additional descriptive information about the Snapshot.

        Args:
            snapshot_id (str): The unique ID of the Snapshot whose metadata is to be retrieved.

        Returns:
            dict: The metadata dictionary associated with the Snapshot.

        Raises:
            ValueError: If the Snapshot with the given `snapshot_id` does not exist.

        Examples:
            Retrieve metadata for a specific Snapshot:
                metadata = manager.get_metadata("example_snapshot_id")
                print(metadata)
        """

        snapshot = self.storage.get_snapshot(snapshot_id)
        return snapshot.metadata

    def update_metadata(self, snapshot_id, new_metadata):
        """
        Update the metadata for a specific Snapshot.

        This method merges the provided `new_metadata` dictionary with the existing metadata
        of the Snapshot identified by `snapshot_id`. Existing keys in the metadata will be
        updated, and new keys will be added.

        Args:
            snapshot_id (str): The unique ID of the Snapshot to update.
            new_metadata (dict): A dictionary of new metadata to merge with the existing metadata.

        Raises:
            ValueError: If the Snapshot ID does not exist in storage.
            TypeError: If `new_metadata` is not a dictionary.

        Examples:
            Update metadata for a snapshot:
                manager.update_metadata("snap_id", {"new_key": "new_value"})

            Merge additional metadata:
                manager.update_metadata("snap_id", {"updated_field": "new_value"})
        """
        if not isinstance(new_metadata, dict):
            raise TypeError("new_metadata must be a dictionary.")

        snapshot = self.storage.get_snapshot(snapshot_id)
        snapshot.metadata.update(new_metadata)

    # Tag Management

    def add_tags(self, snapshot_id, tags):
        """
        Add tags to a Snapshot.

        This method adds the provided tags to the Snapshot identified by `snapshot_id`.
        Duplicate tags will be ignored, ensuring that each tag appears only once in the Snapshot.

        Args:
            snapshot_id (str): The unique ID of the Snapshot to which tags should be added.
            tags (list[str]): A list of tags to add to the Snapshot. Each tag should be a string.

        Raises:
            ValueError: If the Snapshot ID does not exist in the storage.
            TypeError: If `tags` is not a list or contains non-string elements.

        Examples:
            Add new tags to a Snapshot:
                manager.add_tags("example_id", ["important", "review"])
        """

        if not isinstance(tags, list):
            raise TypeError("tags must be a list.")

        snapshot = self.storage.get_snapshot(snapshot_id)
        snapshot.add_tags(tags)

    def get_tags(self, snapshot_id):
        """
        Retrieve the tags associated with a specific Snapshot.

        Args:
            snapshot_id (str): The unique ID of the Snapshot whose tags are to be retrieved.

        Returns:
            list[str]: A list of tags associated with the Snapshot. If no tags are associated, returns an empty list.

        Raises:
            ValueError: If the Snapshot ID does not exist in the storage.

        Examples:
            Retrieve tags for a Snapshot:
                tags = manager.get_tags("snapshot_id")
                print(tags)
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
        return self.storage.insertion_order

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
            deepcopy (bool, optional): Whether to return a deep copy of the Snapshot object.

        Returns:
            Snapshot: The Snapshot object.

        Raises:
            IndexError: If the index is out of range.

        Examples:
            Retrieve a Snapshot by index:
                snapshot = manager.get_snapshot_by_index(0)
        """
        if index < 0 or index >= len(self.storage.insertion_order):
            raise IndexError(f"Index '{index}' is out of range.")

        # Get the snapshot ID by its index
        snapshot_id = self.storage.insertion_order[index]

        # Use the existing get_snapshot method to retrieve the snapshot
        return self.get_snapshot(snapshot_id, deepcopy=deepcopy)

    def get_oldest_snapshot(self, deepcopy=DEFAULT):
        """
        Retrieve the oldest Snapshot.

        Args:
            deepcopy (bool, optional): Whether to return a deep copy of the Snapshot object.

        Returns:
            Snapshot: The oldest Snapshot object.

        Raises:
            IndexError: If no Snapshots are available.

        Examples:
            Retrieve the oldest Snapshot:
                snapshot = manager.get_oldest_snapshot()
        """
        if not self.storage.insertion_order:
            raise IndexError("No Snapshots are available.")

        # Get the oldest snapshot ID from the storage
        oldest_snapshot_id = self.storage.insertion_order[-1]

        # Use the existing get_snapshot method to retrieve the snapshot
        return self.get_snapshot(oldest_snapshot_id, deepcopy=deepcopy)

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
            self.storage.insertion_order
            if ascending
            else list(reversed(self.storage.insertion_order))
        )

    def _create_snapshot(self, data, metadata, tags, deepcopy, snapshot_id):
        """
        Factory method to create a Snapshot. Can be overridden by subclasses.

        Args:
            data: The data to store in the Snapshot.
            metadata (dict): Metadata to associate with the snapshot.
            tags (list): Tags to associate with the snapshot.
            deepcopy (bool): Whether to deepcopy the data.
            snapshot_id (str): Identifier for the Snapshot.

        Returns:
            Snapshot: The created Snapshot instance.
        """

        return Snapshot(
            data,
            metadata=metadata,
            tags=tags,
            deepcopy=deepcopy,
            snapshot_id=snapshot_id,
        )

    def save_to_file(self, file_path, compress=True):
        """
        Save the current state of the SnapshotManager to a file.

        The state includes all stored snapshots, metadata, and configuration. This allows for
        persistent storage and later restoration of the manager's state.

        Args:
            file_path (str): The file path where the state should be saved.
            compress (bool): Whether to compress the saved state. Defaults to True.

        Returns:
            None

        Raises:
            IOError: If there is an issue writing to the file.

        Examples:
            Save the state of the manager to a file:
                manager.save_to_file("snapshot_manager_state.json")

            Save without compression:
                manager.save_to_file("snapshot_manager_state.json", compress=False)
        """

        SnapshotPersistence.save_to_file(self, file_path, compress)

    @staticmethod
    def load_from_file(file_path):
        """
        Load a SnapshotManager instance from a file.

        This method restores the state of a SnapshotManager from a file created using `save_to_file`.
        The loaded state includes all stored snapshots, metadata, and configuration.

        Args:
            file_path (str): The path to the file containing the saved state.

        Returns:
            SnapshotManager: A new instance of SnapshotManager initialized with the loaded state.

        Raises:
            IOError: If there is an issue reading from the file.
            ValueError: If the file does not contain valid SnapshotManager data.

        Examples:
            Load a SnapshotManager from a saved file:
                manager = SnapshotManager.load_from_file("snapshot_manager_state.json")
        """
        return SnapshotPersistence.load_from_file(file_path, SnapshotManager, Snapshot)

    def update_cmp(self, cmp):
        """
        Update the cmp function used for ranking snapshots.

        This method updates the comparison function used to determine the ranking of snapshots.
        The new cmp will be applied to the existing snapshots, and they will be re-ranked accordingly.

        Args:
            cmp (callable): A new cmp function that takes two snapshots as arguments and
                returns a negative, zero, or positive number to indicate their relative ranking.
        """
        self.storage.update_cmp(cmp)

    def remove_cmp(self):
        """
        Remove the cmp function and disable ranked snapshot management.

        This method removes the cmp function from the storage, disabling ranked snapshots.
        Snapshots will no longer be ranked or sorted based on the cmp, and only their creation
        order will be maintained.
        """
        self.storage.update_cmp(None)

    def update_max_snapshots(self, max_snapshots):
        """
        Update the maximum number of snapshots to store.

        This method adjusts the maximum limit for the number of snapshots stored by the manager.
        If the new limit is smaller than the current number of snapshots, excess snapshots will be removed
        according to the current ranking or creation order.

        Args:
            max_snapshots (int): The new maximum number of snapshots to retain. If None, no limit is enforced.
        """
        self.storage.update_max_snapshots(max_snapshots)
