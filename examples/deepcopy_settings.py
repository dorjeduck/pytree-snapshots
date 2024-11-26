from snapshot_manager import SnapshotManager

# Initialize the manager
manager = SnapshotManager(deepcopy_on_save=False)  # Default behavior is deepcopy enabled

# Save a snapshot
snapshot_id = manager.save_snapshot({"a": 1, "b": [2, 3]})

# Retrieve a snapshot without deepcopy (shallow reference)
retrieved_reference = manager.get_snapshot(snapshot_id, deepcopy=False)

# Modify the retrieved snapshot
retrieved_reference.data["b"].append(4)

# Since deepcopy was disabled for this retrieval, the original snapshot is also modified
stored_snapshot = manager.get_snapshot(snapshot_id)
assert stored_snapshot.data["b"] == [2, 3, 4], "Deepcopy override failed: Original snapshot was not updated."