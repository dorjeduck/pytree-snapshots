"""Example demonstrating snapshot limit in SnapshotManager.

This example shows:
1. Initializing manager with a maximum snapshot limit
2. Adding snapshots beyond the limit
3. Observing automatic removal of oldest snapshots
"""

from snapshot_manager import SnapshotManager

# Initialize the manager with a maximum of 3 snapshots
MAX_SNAPSHOTS = 3
print(f"\nInitializing SnapshotManager with max_snapshots={MAX_SNAPSHOTS}...")
manager = SnapshotManager(max_snapshots=MAX_SNAPSHOTS)

# Add multiple snapshots
print("\nAdding snapshots sequentially...")
for i in range(5):
    pytree = {"value": i}
    manager.save_snapshot(pytree, snapshot_id=f"snap{i}")
    print(f"\nAdded snapshot {i}: {pytree}")
    snapshots = manager.list_snapshots()
    print(f"Current snapshots ({len(snapshots)}/{MAX_SNAPSHOTS}):", snapshots)
    
    # Show which snapshots were removed
    if i >= MAX_SNAPSHOTS:
        print(f"Note: Oldest snapshot was automatically removed")

# Verify final state
print("\nFinal state:")
final_snapshots = manager.list_snapshots()
print(f"Total snapshots ({len(final_snapshots)}/{MAX_SNAPSHOTS}):", final_snapshots)

# Verify we can still retrieve the latest snapshots
print("\nVerifying latest snapshots:")
for snapshot_id in final_snapshots:
    snapshot = manager.get_snapshot(snapshot_id)
    print(f"Snapshot {snapshot_id}: {snapshot.data}")
