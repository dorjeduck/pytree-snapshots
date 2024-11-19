from snapshot_manager import SnapshotManager
from snapshot_manager.query.base_queries import ByMetadataQuery

# Initialize the manager
manager = SnapshotManager()

# Save snapshots with nested metadata
manager.save_snapshot({"a": 1}, snapshot_id="snap1", metadata={"info": {"project": "example1", "owner": "Alice"}})
manager.save_snapshot({"b": 2}, snapshot_id="snap2", metadata={"info": {"project": "example2", "owner": "Bob"}})

# Custom metadata query: Find snapshots where "owner" is "Alice"
query = ByMetadataQuery("info.owner", "Alice")  # Assume `ByMetadataQuery` supports dot notation for nested keys
results = manager.query.evaluate(query)

print("Snapshots where owner is Alice:", results)
# Output: ['snap1']