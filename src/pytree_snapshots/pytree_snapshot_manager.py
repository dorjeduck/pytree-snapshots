import jax
import jax.numpy as jnp
import numpy as np
import uuid
import pickle
import zlib

from .pytree_snapshot import PytreeSnapshot


class PytreeSnapshotManager:
    """
    A manager for storing and managing PyTree snapshots.
    """

    # Initialization

    def __init__(self, max_snapshots=None, deepcopy=True):
        """
        Initialize the PytreeSnapshotManager.

        Args:
            max_snapshots (int, optional): Maximum number of snapshots to store. Defaults to None (no limit).
            deepcopy (bool): Whether to return deep copies of PyTrees by default. Defaults to True.
        """

        self.snapshots = {}
        self.snapshot_order = []
        self.max_snapshots = max_snapshots
        self.deepcopy = deepcopy

    def __getitem__(self, index, deepcopy=None):
        """
        Retrieve a PytreeSnapshot by index or ID.

        Args:
            index (int or str):
                - If an integer, retrieves the PytreeSnapshot by its position in the order of creation.
                - If a string, retrieves the PytreeSnapshot by its ID.
            deepcopy (bool, optional): Whether to return a deep copy of the PytreeSnapshot's PyTree. Defaults to the manager's deepcopy setting.

        Returns:
            The PyTree of the PytreeSnapshot.

        Raises:
            ValueError: If the index or ID is invalid.
        """
        if isinstance(index, int):
            if index < 0 or index >= len(self.snapshot_order):
                raise ValueError(f"Index '{index}' is out of range.")
            snapshot_id = self.snapshot_order[index]
        elif isinstance(index, str):
            if index not in self.snapshots:
                raise ValueError(f"Snapshot ID '{index}' does not exist.")
            snapshot_id = index
        else:
            raise ValueError(f"Invalid index type: {type(index)}. Must be int or str.")

        return self.snapshots[snapshot_id].get_pytree(
            deepcopy if deepcopy is not None else self.deepcopy
        )

    # Save, Retrieve, and Delete PytreeSnapshots

    def save_snapshot(
        self,
        pytree,
        snapshot_id=None,
        metadata=None,
        tags=None,
        compress=False,
        overwrite=False,
    ):
        """
        Save a new PytreeSnapshot or overwrite an existing one.

        Args:
            pytree: The PyTree to store in the PytreeSnapshot.
            snapshot_id (str, optional): Identifier for the PytreeSnapshot. A unique ID is generated if not provided.
            metadata (dict, optional): Additional metadata to associate with the PytreeSnapshot. Defaults to an empty dictionary.
            tags (list, optional): Tags to associate with the PytreeSnapshot for organization. Defaults to an empty list.
            compress (bool): Whether to compress the PytreeSnapshot. Defaults to False.
            overwrite (bool): Whether to overwrite an existing PytreeSnapshot with the same ID. Defaults to False.

        Returns:
            str: The ID of the saved PytreeSnapshot.

        Raises:
            ValueError: If the PytreeSnapshot ID already exists and `overwrite` is False.

        Examples:
            Save a new PytreeSnapshot:
                snapshot_id = manager.save_snapshot(pytree, metadata={"project": "test"}, tags=["experiment"])

            Overwrite an existing PytreeSnapshot:
                manager.save_snapshot(pytree, snapshot_id="existing_id", overwrite=True)
        """
        snapshot_id = snapshot_id or str(uuid.uuid4())

        if snapshot_id in self.snapshots and not overwrite:
            raise ValueError(
                f"PytreeSnapshot ID '{snapshot_id}' already exists. Use overwrite=True to update it."
            )

        if snapshot_id not in self.snapshots:
            # Enforce max_snapshots limit
            if (
                self.max_snapshots is not None
                and len(self.snapshots) >= self.max_snapshots
            ):
                oldest_snapshot = self.snapshot_order[0]
                self._remove_snapshot(oldest_snapshot)

            self.snapshot_order.append(snapshot_id)

        # Create and store the snapshot
        self.snapshots[snapshot_id] = PytreeSnapshot(pytree, metadata, tags, compress)
        return snapshot_id

    def get_snapshot(self, snapshot_id, deepcopy=None):
        """
        Retrieve the PyTree from a PytreeSnapshot.

        Args:
            snapshot_id (str): The ID of the PytreeSnapshot to retrieve.
            deepcopy (bool, optional): Whether to return a deep copy. Defaults to the manager's deepcopy setting.

        Returns:
            The PyTree of the PytreeSnapshot.

        Raises:
            ValueError: If the PytreeSnapshot ID does not exist.

        Examples:
            Retrieve a PytreeSnapshot:
                pytree = manager.get_snapshot("snap_id")
        """
        if snapshot_id not in self.snapshots:
            raise ValueError(f"PytreeSnapshot ID '{snapshot_id}' does not exist.")
        return self.snapshots[snapshot_id].get_pytree(
            deepcopy if deepcopy is not None else self.deepcopy
        )

    def get_latest_snapshot(self, deepcopy=None):
        """
        Retrieve the most recent snapshot.

        Args:
            deepcopy (bool, optional): Whether to return a deep copy.

        Returns:
            The PyTree of the most recent snapshot.

        Raises:
            IndexError: If no snapshots are available.
        """
        if not self.snapshot_order:
            raise IndexError("No snapshots available.")
        snapshot_id = self.snapshot_order[-1]
        return self.snapshots[snapshot_id].get_pytree(
            deepcopy if deepcopy is not None else self.deepcopy
        )

    def delete_snapshot(self, snapshot_id):
        """
        Delete a PytreeSnapshot.

        Args:
            snapshot_id (str): The ID of the PytreeSnapshot to delete.

        Returns:
            bool: True if the PytreeSnapshot was deleted, False if it didn't exist.

        Examples:
            Delete a PytreeSnapshot:
                success = manager.delete_snapshot("snap_id")
        """
        return self._remove_snapshot(snapshot_id)

    def clone_snapshot(self, snapshot_id, new_snapshot_id=None):
        """
        Clone an existing PytreeSnapshot with a new ID.

        Args:
            snapshot_id (str): The ID of the PytreeSnapshot to clone.
            new_snapshot_id (str, optional): The ID to assign to the cloned PytreeSnapshot. A unique ID is generated if not provided.

        Returns:
            str: The ID of the cloned PytreeSnapshot.

        Raises:
            ValueError: If the PytreeSnapshot ID to clone does not exist or the new PytreeSnapshot ID already exists.

        Examples:
            Clone a PytreeSnapshot with a new ID:
                cloned_id = manager.clone_snapshot("existing_id")
        """
        if snapshot_id not in self.snapshots:
            raise ValueError(f"PytreeSnapshot ID '{snapshot_id}' does not exist.")

        # Generate a new snapshot ID if none is provided
        new_snapshot_id = new_snapshot_id or str(uuid.uuid4())

        if new_snapshot_id in self.snapshots:
            raise ValueError(f"PytreeSnapshot ID '{new_snapshot_id}' already exists.")

        # Clone the snapshot
        original_snapshot = self.snapshots[snapshot_id]
        pytree = original_snapshot.get_pytree(deepcopy=True)
        metadata = original_snapshot.metadata.copy()
        tags = original_snapshot.tags.copy()
        compress = original_snapshot.compress

        # Save the cloned snapshot
        self.snapshots[new_snapshot_id] = PytreeSnapshot(
            pytree=pytree, metadata=metadata, tags=tags, compress=compress
        )
        self.snapshot_order.append(new_snapshot_id)

        return new_snapshot_id

    # Metadata Management

    def get_metadata(self, snapshot_id):
        """
        Retrieve metadata for a specific PytreeSnapshot.

        Args:
            snapshot_id (str): The ID of the PytreeSnapshot.

        Returns:
            dict: The metadata associated with the PytreeSnapshot.

        Raises:
            ValueError: If the PytreeSnapshot ID does not exist.

        Examples:
            Retrieve metadata:
                metadata = manager.get_metadata("snap_id")
        """
        if snapshot_id not in self.snapshots:
            raise ValueError(f"PytreeSnapshot ID '{snapshot_id}' does not exist.")
        return self.snapshots[snapshot_id].metadata

    def update_metadata(self, snapshot_id, new_metadata):
        """
        Update the metadata for a specific PytreeSnapshot.

        Args:
            snapshot_id (str): The ID of the PytreeSnapshot to update.
            new_metadata (dict): The new metadata to merge with the existing metadata.

        Raises:
            ValueError: If the PytreeSnapshot ID does not exist.

        Examples:
            Update metadata:
                manager.update_metadata("snap_id", {"new_key": "new_value"})
        """
        if snapshot_id not in self.snapshots:
            raise ValueError(f"PytreeSnapshot ID '{snapshot_id}' does not exist.")

        # Update the metadata
        self.snapshots[snapshot_id].metadata.update(new_metadata)

    # Tag Management

    def add_tags(self, snapshot_id, tags):
        """
        Add tags to a PytreeSnapshot.

        Args:
            snapshot_id (str): The ID of the PytreeSnapshot.
            tags (list): Tags to add.

        Raises:
            ValueError: If the PytreeSnapshot ID does not exist.

        Examples:
            Add tags to a PytreeSnapshot:
                manager.add_tags("snap_id", ["important", "experiment"])
        """
        if snapshot_id not in self.snapshots:
            raise ValueError(f"PytreeSnapshot ID '{snapshot_id}' does not exist.")
        self.snapshots[snapshot_id].add_tags(tags)

    def remove_tags(self, snapshot_id, tags):
        """
        Remove tags from a PytreeSnapshot.

        Args:
            snapshot_id (str): The ID of the PytreeSnapshot.
            tags (list): Tags to remove.

        Raises:
            ValueError: If the PytreeSnapshot ID does not exist.

        Examples:
            Remove tags from a PytreeSnapshot:
                manager.remove_tags("snap_id", ["obsolete"])
        """
        if snapshot_id not in self.snapshots:
            raise ValueError(f"PytreeSnapshot ID '{snapshot_id}' does not exist.")
        self.snapshots[snapshot_id].remove_tags(tags)

    # Querying and Listing PytreeSnapshots

    def list_snapshots(self):
        """
        List all PytreeSnapshot IDs in the order they were created.

        Returns:
            list: PytreeSnapshot IDs.

        Examples:
            List all PytreeSnapshot IDs:
                ids = manager.list_snapshots()
        """
        return self.snapshot_order.copy()

    def get_snapshot_count(self):
        """
        Return the current number of PytreeSnapshots.

        Returns:
            int: The number of stored PytreeSnapshots.

        Examples:
            Get the number of PytreeSnapshots:
                count = manager.get_snapshot_count()
        """
        return len(self.snapshots)

    def find_snapshots_by_metadata(self, key, value=None):
        """
        Find all PytreeSnapshots that contain a specific metadata key and optionally a specific value.

        Args:
            key (str): The metadata key to search for.
            value (optional): The value to match for the given metadata key. If None, matches any value for the key.

        Returns:
            list: PytreeSnapshot IDs that match the metadata key and value.

        Examples:
            Find PytreeSnapshots by metadata key:
                ids = manager.find_snapshots_by_metadata("project")
        """
        return [
            snapshot_id
            for snapshot_id, snapshot in self.snapshots.items()
            if key in snapshot.metadata
            and (value is None or snapshot.metadata[key] == value)
        ]

    def find_snapshots_by_tag(self, tag):
        """
        Find all PytreeSnapshots with a specific tag.

        Args:
            tag (str): The tag to search for.

        Returns:
            list: PytreeSnapshot IDs with the specified tag.

        Examples:
            Find PytreeSnapshots by tag:
                ids = manager.find_snapshots_by_tag("experiment")
        """
        return [
            snapshot_id
            for snapshot_id, snapshot in self.snapshots.items()
            if snapshot.has_tag(tag)
        ]

    def get_snapshots_by_time_range(self, start_time, end_time):
        """
        Retrieve PytreeSnapshots created within a specified time range.

        Args:
            start_time (float): Start of the time range (UNIX timestamp).
            end_time (float): End of the time range (UNIX timestamp).

        Returns:
            list: PytreeSnapshot IDs created within the specified time range.

        Examples:
            Find PytreeSnapshots in a time range:
                ids = manager.get_snapshots_by_time_range(1690000000.0, 1700000000.0)
        """
        return [
            snapshot_id
            for snapshot_id, snapshot in self.snapshots.items()
            if start_time <= snapshot.timestamp <= end_time
        ]

    def get_snapshot_by_index(self, index, deepcopy=None):
        """
        Retrieve a PytreeSnapshot by its creation index.

        Args:
            index (int): Index of the PytreeSnapshot in the creation order.
            deepcopy (bool, optional): Whether to return a deep copy of the PyTree.

        Returns:
            The PyTree of the PytreeSnapshot at the specified index.

        Raises:
            IndexError: If the index is out of range.

        Examples:
            Retrieve a PytreeSnapshot by index:
                pytree = manager.get_snapshot_by_index(0)
        """
        if index < 0 or index >= len(self.snapshot_order):
            raise IndexError(f"Index '{index}' is out of range.")
        snapshot_id = self.snapshot_order[index]
        return self.get_snapshot(snapshot_id, deepcopy=deepcopy)

    def get_oldest_snapshot(self, deepcopy=None):
        """
        Retrieve the oldest PytreeSnapshot.

        Args:
            deepcopy (bool, optional): Whether to return a deep copy of the PyTree.

        Returns:
            The PyTree of the oldest PytreeSnapshot.

        Raises:
            IndexError: If no PytreeSnapshots are available.

        Examples:
            Retrieve the oldest PytreeSnapshot:
                pytree = manager.get_oldest_snapshot()
        """
        if not self.snapshot_order:
            raise IndexError("No PytreeSnapshots available.")
        snapshot_id = self.snapshot_order[0]
        return self.get_snapshot(snapshot_id, deepcopy=deepcopy)

    def list_snapshots_by_age(self, ascending=True):
        """
        List all PytreeSnapshot IDs sorted by their age (creation time).

        Args:
            ascending (bool): If True, lists PytreeSnapshots from oldest to newest. Otherwise, newest to oldest.

        Returns:
            list: PytreeSnapshot IDs sorted by age.

        Examples:
            List PytreeSnapshots from newest to oldest:
                ids = manager.list_snapshots_by_age(ascending=False)
        """
        return self.snapshot_order if ascending else list(reversed(self.snapshot_order))

    # Comparison

    def compare_snapshots(
        self,
        snapshot_id1,
        snapshot_id2,
        custom_comparator=None,
        condition=None,
    ):
        """
        Compare two PytreeSnapshots and return their differences as a PyTree.

        Args:
            snapshot_id1 (str): The ID of the first PytreeSnapshot.
            snapshot_id2 (str): The ID of the second PytreeSnapshot.
            custom_comparator (callable, optional): A custom function to compare leaves. Defaults to None.
                - The function should take two arguments (leaf1, leaf2) and return:
                    - None: If the leaves are considered equal.
                    - A custom difference representation otherwise.
            condition (callable, optional): A function that takes a leaf value and returns True if it should be compared, False otherwise.
                - If the condition is False, the leaf is excluded from the comparison.
            return_paths (bool): If True, include paths to differing elements in the result. Defaults to True.

        Returns:
            dict: Differences between the two PytreeSnapshots.
                - Keys are paths (if `return_paths=True`) or an index-like representation.
                - Values are tuples: (value_in_snapshot1, value_in_snapshot2) or the custom comparison result.

        Raises:
            ValueError: If either PytreeSnapshot ID does not exist.

        Examples:
            Save two PytreeSnapshots with different PyTrees:
            >>> pytree1 = {"a": jnp.array([1, 2, 3]), "b": {"x": 42, "y": "hello"}}
            >>> pytree2 = {"a": jnp.array([1, 2, 4]), "b": {"x": 42, "y": "world"}}
            >>> manager.save_snapshot(pytree1, snapshot_id="id1")
            >>> manager.save_snapshot(pytree2, snapshot_id="id2")

            Compare the two PytreeSnapshots:
            >>> differences = manager.compare_snapshots(
                    "id1", "id2",
                    condition=lambda x: isinstance(x, jnp.ndarray)
                )
            >>> print(differences)

            Output:
            {
                "a": (Array([1, 2, 3], dtype=int32), Array([1, 2, 4], dtype=int32))
            }
        """
        if snapshot_id1 not in self.snapshots or snapshot_id2 not in self.snapshots:
            raise ValueError("Both PytreeSnapshot IDs must exist.")

        pytree1 = self.snapshots[snapshot_id1].get_pytree(deepcopy=False)
        pytree2 = self.snapshots[snapshot_id2].get_pytree(deepcopy=False)

        # Unify structures
        pytree1, pytree2 = self.unify_pytree_structures(pytree1, pytree2)

        def default_comparator(x, y):
            if isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):
                return None if jnp.array_equal(x, y) else (x, y)
            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                return None if np.array_equal(x, y) else (x, y)
            if isinstance(x, (int, float, str, bool)) and isinstance(
                y, (int, float, str, bool)
            ):
                return (x, y) if x != y else None
            if type(x) != type(y) or x != y:
                return (x, y)
            return None

        comparator = (
            (lambda x, y: None if custom_comparator(x, y) else (x, y))
            if custom_comparator
            else default_comparator
        )

        def collect_differences(x, y):
            # Apply condition
            if condition and not (condition(x) and condition(y)):
                return Ellipsis  # Represent excluded elements with Ellipsis
            result = comparator(x, y)
            return result if result is not None else None

        differences = jax.tree.map(collect_differences, pytree1, pytree2)

        return differences

    @staticmethod
    def unify_pytree_structures(pytree1, pytree2, placeholder=Ellipsis):
        """
        Align the structures of two PyTrees, filling in missing keys or elements with a placeholder.

        Args:
            pytree1: The first PyTree (can be a nested dict/list/tuple).
            pytree2: The second PyTree (can be a nested dict/list/tuple).
            placeholder: The value to use for missing elements. Defaults to Ellipsis.

        Returns:
            Tuple: Two PyTrees with unified structure.
        """
        if isinstance(pytree1, dict) and isinstance(pytree2, dict):
            # Union of all keys
            all_keys = set(pytree1.keys()).union(pytree2.keys())
            aligned1 = {}
            aligned2 = {}
            for key in all_keys:
                aligned1[key], aligned2[key] = PytreeSnapshotManager.unify_pytree_structures(
                    pytree1.get(key, placeholder),
                    pytree2.get(key, placeholder),
                    placeholder,
                )
            return aligned1, aligned2
        elif isinstance(pytree1, (list, tuple)) and isinstance(pytree2, (list, tuple)):
            # Align lists/tuples
            max_len = max(len(pytree1), len(pytree2))
            aligned1 = list(pytree1) + [placeholder] * (max_len - len(pytree1))
            aligned2 = list(pytree2) + [placeholder] * (max_len - len(pytree2))
            if isinstance(pytree1, tuple):
                aligned1 = tuple(aligned1)
            if isinstance(pytree2, tuple):
                aligned2 = tuple(aligned2)
            return aligned1, aligned2
        else:
            # For leaves or mismatched types, return as-is
            return pytree1, pytree2

    # State Persistence

    def save_state(self, file_path, compress=False):
        """
        Save the current state of the PytreeSnapshotManager to a file.

        Args:
            file_path (str): Path to the file where the state should be saved.
            compress (bool): Whether to compress the saved state. Defaults to False.

        Returns:
            None

        Examples:
            Save the manager's state:
                manager.save_state("state.pkl", compress=True)
        """
        # Prepare data for serialization
        state = {
            "snapshots": {
                snapshot_id: snapshot.to_dict()
                for snapshot_id, snapshot in self.snapshots.items()
            },
            "snapshot_order": self.snapshot_order,
            "max_snapshots": self.max_snapshots,
            "deepcopy": self.deepcopy,
        }

        # Serialize and compress if required
        serialized_data = pickle.dumps(state)
        if compress:
            serialized_data = zlib.compress(serialized_data)

        # Write to file
        with open(file_path, "wb") as file:
            file.write(serialized_data)

    @staticmethod
    def load_state(file_path, decompress=False):
        """
        Load the PytreeSnapshotManager state from a file.

        Args:
            file_path (str): Path to the file containing the saved state.
            decompress (bool): Whether the state file is compressed. Defaults to False.

        Returns:
            PytreeSnapshotManager: An instance of PytreeSnapshotManager with the loaded state.

        Raises:
            ValueError: If decompression or deserialization fails.
            FileNotFoundError: If the specified file does not exist.

        Examples:
            Load the manager's state:
                manager = PytreeSnapshotManager.load_state("state.pkl", decompress=True)
        """
        try:
            # Read the file
            with open(file_path, "rb") as file:
                serialized_data = file.read()

            # Decompress if needed
            if decompress:
                try:
                    serialized_data = zlib.decompress(serialized_data)
                except zlib.error as e:
                    raise ValueError(f"Decompression failed: {e}")

            # Deserialize the state
            state = pickle.loads(serialized_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        except (pickle.PickleError, EOFError) as e:
            raise ValueError(f"Deserialization failed: {e}")

        # Recreate PytreeSnapshotManager instance
        manager = PytreeSnapshotManager(
            max_snapshots=state["max_snapshots"],
            deepcopy=state["deepcopy"],
        )

        # Convert dictionaries back to PytreeSnapshot objects
        try:
            manager.snapshots = {
                snapshot_id: PytreeSnapshot.from_dict(snapshot_data)
                for snapshot_id, snapshot_data in state["snapshots"].items()
            }
        except Exception as e:
            raise ValueError(f"Failed to restore snapshots: {e}")

        manager.snapshot_order = state["snapshot_order"]
        return manager

    # private

    def _remove_snapshot(self, snapshot_id):
        """
        Remove a PytreeSnapshot from the manager, ensuring both `snapshots` and `snapshot_order` are updated.

        Args:
            snapshot_id (str): The ID of the PytreeSnapshot to remove.

        Returns:
            bool: True if the PytreeSnapshot was removed, False if it didn't exist.

        Examples:
            Remove a PytreeSnapshot:
                success = manager._remove_snapshot("snap_id")
        """
        if snapshot_id in self.snapshots:
            del self.snapshots[snapshot_id]
            self.snapshot_order.remove(snapshot_id)
            return True
        return False
