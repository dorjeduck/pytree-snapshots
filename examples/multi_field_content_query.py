from snapshot_manager import SnapshotManager
from snapshot_manager.query.base_queries import ByContentQuery

# Initialize the manager
manager = SnapshotManager()

# Save snapshots with structured content
manager.save_snapshot({"a": 1, "b": 2}, snapshot_id="snap1")
manager.save_snapshot({"a": 3, "b": 4}, snapshot_id="snap2")
manager.save_snapshot({"a": 1, "b": 4}, snapshot_id="snap3")

# Query snapshots where "a" == 1 and "b" == 4
query = ByContentQuery(lambda pytree: pytree.get("a") == 1 and pytree.get("b") == 4)
results = manager.query.evaluate(query)

print("Snapshots where a=1 and b=4:", results)
# Output: ['snap3']