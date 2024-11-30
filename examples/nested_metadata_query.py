"""Example demonstrating nested metadata queries in SnapshotManager.

This example shows:
1. Saving snapshots with nested metadata structure
2. Using dot notation to query nested metadata fields
3. Finding snapshots by nested metadata values
"""

from snapshot_manager import SnapshotManager
from snapshot_manager.query.base_queries import ByMetadataQuery

# Initialize the manager
manager = SnapshotManager()

# Save snapshots with nested metadata
print("\nSaving snapshots with nested metadata...")
nested_metadata1 = {
    "info": {
        "project": "example1",
        "owner": "Alice",
        "settings": {
            "version": "1.0",
            "status": "active"
        }
    }
}

nested_metadata2 = {
    "info": {
        "project": "example2",
        "owner": "Bob",
        "settings": {
            "version": "2.0",
            "status": "inactive"
        }
    }
}

# Save snapshots
manager.save_snapshot(
    {"a": 1}, 
    snapshot_id="snap1", 
    metadata=nested_metadata1
)
print("Saved snap1:", nested_metadata1)

manager.save_snapshot(
    {"b": 2}, 
    snapshot_id="snap2", 
    metadata=nested_metadata2
)
print("Saved snap2:", nested_metadata2)

# Query using dot notation for nested fields
print("\nQuerying snapshots...")

# Find snapshots where owner is Alice
print("\nFinding snapshots where info.owner = 'Alice'...")
owner_query = ByMetadataQuery("info.owner", "Alice")
owner_results = manager.query.evaluate(owner_query)
print("Results:", owner_results)

# Find snapshots with active status
print("\nFinding snapshots where info.settings.status = 'active'...")
status_query = ByMetadataQuery("info.settings.status", "active")
status_results = manager.query.evaluate(status_query)
print("Results:", status_results)

# Verify results
print("\nVerifying matches...")
for snapshot_id in owner_results:
    snapshot = manager.get_snapshot(snapshot_id)
    print(f"\nSnapshot: {snapshot_id}")
    print("Owner:", snapshot.metadata["info"]["owner"])
    print("Status:", snapshot.metadata["info"]["settings"]["status"])