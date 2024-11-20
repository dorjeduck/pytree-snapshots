from pytree_snapshots import PytreeSnapshotManager

# Initialize the manager with a maximum of 3 snapshots
manager = PytreeSnapshotManager(max_snapshots=3)

# Add multiple snapshots
for i in range(5):
    pytree = {"value": i}
    manager.save_snapshot(pytree, snapshot_id=f"snap{i}")
    print(f"Added snapshot {i}: {pytree}")
    print(f"Current snapshots: {manager.list_snapshots()}")

# Verify that the number of snapshots does not exceed the limit
print("\nFinal list of snapshots (should not exceed 3):")
print(manager.list_snapshots())