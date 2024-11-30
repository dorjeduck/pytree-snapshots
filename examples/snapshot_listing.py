"""Example demonstrating snapshot listing in SnapshotManager.

This example shows:
1. Saving multiple snapshots
2. Listing all saved snapshots
3. Displaying snapshot IDs and metadata
"""

# Save multiple snapshots
from snapshot_manager import SnapshotManager

# Create a manager and save a snapshot
print("\nCreating SnapshotManager and saving snapshots...")
manager = SnapshotManager()

manager.save_snapshot({"a": 1}, snapshot_id="snap1")
print("Saved snapshot 'snap1' with data: {a: 1}")

manager.save_snapshot({"b": 2}, snapshot_id="snap2")
print("Saved snapshot 'snap2' with data: {b: 2}")

manager.save_snapshot({"c": 3}, snapshot_id="snap3")
print("Saved snapshot 'snap3' with data: {c: 3}")

# List snapshots sorted by age (newest to oldest)
print("\nListing snapshots from newest to oldest...")
snapshots_newest_to_oldest = manager.list_snapshots_by_age(ascending=False)
print(f"Snapshots from newest to oldest: {snapshots_newest_to_oldest}")

# Display snapshot metadata
print("\nDisplaying snapshot metadata:")
for snapshot_id in snapshots_newest_to_oldest:
    snapshot = manager.get_snapshot(snapshot_id)
    print(f"Snapshot ID: {snapshot_id}, Data: {snapshot.data}")
