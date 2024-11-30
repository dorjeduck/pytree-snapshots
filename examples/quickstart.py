"""Quickstart example for using SnapshotManager.

This example demonstrates:
1. Initializing a SnapshotManager
2. Saving and retrieving snapshots
3. Basic querying by metadata and tags
"""

"""Quickstart example demonstrating basic usage of SnapshotManager.

This example shows:
1. Creating a snapshot manager
2. Saving snapshots with metadata and tags
3. Retrieving snapshots
4. Basic querying
"""

from snapshot_manager import SnapshotManager
from snapshot_manager.query import ByMetadataQuery, ByTagQuery

# Initialize the manager with a maximum snapshot limit
manager = SnapshotManager(max_snapshots=10)

# Create some example data
pytree1 = {"model": {"weights": [1.0, 2.0], "bias": 0.5}}
pytree2 = {"model": {"weights": [1.5, 2.5], "bias": 0.7}}

# Save snapshots with metadata and tags
snapshot1 = manager.save_snapshot(
    pytree1,
    metadata={"epoch": 1, "accuracy": 0.85},
    tags=["checkpoint", "training"]
)

snapshot2 = manager.save_snapshot(
    pytree2,
    metadata={"epoch": 2, "accuracy": 0.90},
    tags=["checkpoint", "training"]
)

# Retrieve a specific snapshot
retrieved = manager.get_snapshot(snapshot1)
print("\nRetrieved snapshot data:", retrieved.data)
print("Metadata:", retrieved.metadata)
print("Tags:", manager.get_tags(snapshot1))

# Query snapshots by metadata
query = ByMetadataQuery("accuracy", 0.90)
results = manager.query.evaluate(query)
print("\nSnapshots with 90% accuracy:", results)

# Query snapshots by tag
query = ByTagQuery("checkpoint")
results = manager.query.evaluate(query)
print("\nAll checkpoint snapshots:", results)
