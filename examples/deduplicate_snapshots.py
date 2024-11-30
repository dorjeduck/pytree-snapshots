"""Example demonstrating snapshot deduplication in SnapshotManager.

This example shows:
1. Saving snapshots with duplicate content
2. Using ByContentQuery to find duplicates
3. Finding snapshots with identical content
"""

from snapshot_manager import SnapshotManager
from snapshot_manager.query.base_queries import ByContentQuery

# Initialize the manager
manager = SnapshotManager()

print("\nSaving test snapshots...")
# Save snapshots with duplicate content
data1 = {"a": 1, "b": 2}
data2 = {"a": 3, "b": 4}

manager.save_snapshot(data1, snapshot_id="snap1", metadata={"version": "1.0"})
print("Saved snap1:", data1)

manager.save_snapshot(data1, snapshot_id="snap2", metadata={"version": "1.1"})
print("Saved snap2 (duplicate of snap1):", data1)

manager.save_snapshot(data2, snapshot_id="snap3", metadata={"version": "2.0"})
print("Saved snap3 (different content):", data2)

# Query for snapshots with identical content as a reference snapshot
print("\nFinding duplicates of snap1...")
reference_snapshot_id = "snap1"
reference_snapshot = manager.get_snapshot(reference_snapshot_id)

# Define content comparison query
query = ByContentQuery(lambda pytree: pytree == reference_snapshot.data)
duplicates = manager.query.evaluate(query)

print("\nResults:")
print("Reference snapshot:", reference_snapshot_id)
print("Found duplicates:", duplicates)

# Show that duplicates have same content but different metadata
print("\nVerifying duplicates...")
for snapshot_id in duplicates:
    snapshot = manager.get_snapshot(snapshot_id)
    print(f"\nSnapshot: {snapshot_id}")
    print("Content:", snapshot.data)
    print("Metadata:", snapshot.metadata)