import pickle
import zlib


class SnapshotPersistence:
    """
    Handles saving and loading the state of SnapshotManager.
    """

    @staticmethod
    def save_state(manager, file_path, compress=False):
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
            "deepcopy": manager.deepcopy,
        }

        # Serialize the state
        serialized_data = pickle.dumps(state)
        if compress:
            serialized_data = zlib.compress(serialized_data)

        # Write the serialized data to the file
        with open(file_path, "wb") as file:
            file.write(serialized_data)

    @staticmethod
    def load_state(file_path, decompress=False):
        """
        Load the state of a SnapshotManager from a file.

        Args:
            file_path (str): Path to the file containing the saved state.
            decompress (bool): Whether the data is compressed.

        Returns:
            dict: The deserialized state.

        Raises:
            ValueError: If decompression or deserialization fails.
            FileNotFoundError: If the specified file does not exist.
        """
        try:
            # Read the serialized data from the file
            with open(file_path, "rb") as file:
                serialized_data = file.read()

            # Decompress if necessary
            if decompress:
                serialized_data = zlib.decompress(serialized_data)

            # Deserialize the state
            state = pickle.loads(serialized_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        except (pickle.PickleError, zlib.error, EOFError) as e:
            raise ValueError(f"Failed to load state: {e}")

        return state
