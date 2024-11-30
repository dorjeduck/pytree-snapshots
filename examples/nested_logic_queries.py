"""Example demonstrating nested logic queries in SnapshotManager.

This example shows:
1. Using logical operators (AND, OR, NOT)
2. Combining metadata and tag queries
3. Building complex nested query conditions
"""

from snapshot_manager import SnapshotManager
from snapshot_manager.query import AndQuery, OrQuery, NotQuery, ByMetadataQuery, ByTagQuery

# Initialize the manager
manager = SnapshotManager()

# Save snapshots with metadata and tags
print("\nSaving test snapshots...")
manager.save_snapshot(
    {"a": 1},
    snapshot_id="snap1",
    metadata={"project": "example1"},
    tags=["experiment", "baseline"],
)
print("Saved snap1: project=example1, tags=[experiment, baseline]")

manager.save_snapshot(
    {"b": 2}, 
    snapshot_id="snap2", 
    metadata={"project": "example2"}, 
    tags=["control"]
)
print("Saved snap2: project=example2, tags=[control]")

manager.save_snapshot(
    {"c": 3},
    snapshot_id="snap3",
    metadata={"project": "example1"},
    tags=["experiment", "published"],
)
print("Saved snap3: project=example1, tags=[experiment, published]")

# Build nested logical query
print("\nBuilding nested logical query:")
print("(project='example1' AND tag='experiment') OR (NOT tag='control')")

# First part: project='example1' AND tag='experiment'
condition1 = AndQuery(
    ByMetadataQuery("project", "example1"),
    ByTagQuery("experiment")
)

# Second part: NOT tag='control'
condition2 = NotQuery(ByTagQuery("control"))

# Combine conditions with OR
query = OrQuery(condition1, condition2)

# Evaluate the query
print("\nEvaluating query...")
results = manager.query.evaluate(query)
print("Matching snapshots:", results)

# Verify results
print("\nVerifying matches:")
for snapshot_id in results:
    snapshot = manager.get_snapshot(snapshot_id)
    print(f"\nSnapshot: {snapshot_id}")
    print("Project:", snapshot.metadata.get("project"))
    print("Tags:", snapshot.tags)
