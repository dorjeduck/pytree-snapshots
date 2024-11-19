import pickle
import zlib


class SnapshotPersistence:
    """
    Handles saving and loading the state of SnapshotManager.
    """

    @staticmethod
    def save_state(manager, file_path, compress=True):
        """
        Save the state of a SnapshotManager to a file.

        Args:
            manager (SnapshotManager): The manager whose state is being saved.
            file_path (str): Path to the file where the state will be saved.
            compress (bool): Whether to compress the saved data.

        Returns:
            None
        """
        # Prepare the manager's state for serialization
        state = {
            "snapshots": {
                snapshot_id: snapshot.to_dict()
                for snapshot_id, snapshot in manager.storage.snapshots.items()
            },
            "snapshot_order": manager.storage.snapshot_order,
            "max_snapshots": manager.storage.max_snapshots,
            "deepcopy_on_save": manager.deepcopy_on_save,
            "deepcopy_on_retrieve": manager.deepcopy_on_retrieve,
        }

        # Serialize the state
        serialized_data = pickle.dumps(state)
        if compress:
            serialized_data = zlib.compress(serialized_data)

        # Write the serialized data to the file
        with open(file_path, "wb") as file:
            file.write(serialized_data)

    @staticmethod
    def load_state(file_path):
        """
        Load the state of a SnapshotManager from a file.

        Args:
            file_path (str): Path to the file containing the saved state.

        Returns:
            dict: The deserialized state.

        Raises:
            ValueError: If deserialization fails.
            FileNotFoundError: If the specified file does not exist.
        """
        try:
            # Read the serialized data from the file
            with open(file_path, "rb") as file:
                serialized_data = file.read()

            # Attempt to decompress
            try:
                serialized_data = zlib.decompress(serialized_data)
            except zlib.error:
                pass  # Data is not compressed, continue with raw serialized data

            # Deserialize the state
            state = pickle.loads(serialized_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        except (pickle.PickleError, zlib.error, EOFError) as e:
            raise ValueError(f"Failed to load state: {e}")

        return state