from pytree_snapshots import SnapshotManager

# Initialize the manager
manager = SnapshotManager()

# Save a snapshot
pytree = {"a": 1, "b": 2}
snapshot_id = manager.save_snapshot(pytree, metadata={"project": "example"})

# Retrieve the snapshot
retrieved = manager.get_snapshot(snapshot_id)
print("Retrieved snapshot:", retrieved)
