from snapshot_manager import SnapshotManager

# Initialize the manager
manager = SnapshotManager()

# Save an initial snapshot
snapshot_id = manager.save_snapshot({"a": 1, "b": 2}, snapshot_id="snap1")

# Save a new version of the same snapshot (overwrite enabled)
manager.save_snapshot({"a": 1, "b": 3}, snapshot_id="snap1", overwrite=True)

# Retrieve and inspect the latest version
latest_snapshot = manager.get_snapshot("snap1")
print("Latest snapshot data:", latest_snapshot.data)
# Output: {'a': 1, 'b': 3}
