from snapshot_manager import SnapshotManager

manager = SnapshotManager()

# Save a snapshot with metadata
manager.save_snapshot({"a": 1}, snapshot_id="snap1", metadata={"experiment": "trial1"})

# Retrieve metadata
metadata = manager.get_metadata("snap1")
print("Metadata for snap1:", metadata)

# Update metadata
manager.update_metadata("snap1", {"status": "completed"})
print("Updated metadata for snap1:", manager.get_metadata("snap1"))
