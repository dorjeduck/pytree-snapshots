from snapshot_manager import SnapshotManager
from snapshot_manager.query.base_queries import ByContentQuery

# Initialize the manager
manager = SnapshotManager()

# Save snapshots with duplicate content
manager.save_snapshot({"a": 1, "b": 2}, snapshot_id="snap1")
manager.save_snapshot({"a": 1, "b": 2}, snapshot_id="snap2")
manager.save_snapshot({"a": 3, "b": 4}, snapshot_id="snap3")

# Query for snapshots with identical content as a reference snapshot
reference_snapshot_id = "snap1"
reference_content = manager.get_snapshot(reference_snapshot_id)

query = ByContentQuery(lambda pytree: pytree == reference_content)
duplicates = manager.query.evaluate(query)

print("Duplicate snapshots:", duplicates)
# Output: ['snap1', 'snap2']