from snapshot_manager import SnapshotManager
from snapshot_manager.query import AndQuery, OrQuery, NotQuery, ByMetadataQuery, ByTagQuery

# Initialize the manager
manager = SnapshotManager()

# Save snapshots with metadata and tags
manager.save_snapshot(
    {"a": 1},
    snapshot_id="snap1",
    metadata={"project": "example1"},
    tags=["experiment", "baseline"],
)
manager.save_snapshot(
    {"b": 2}, snapshot_id="snap2", metadata={"project": "example2"}, tags=["control"]
)
manager.save_snapshot(
    {"c": 3},
    snapshot_id="snap3",
    metadata={"project": "example1"},
    tags=["experiment", "published"],
)

# Logical Query: Find snapshots that are in project "example1" AND tagged with "experiment",
# OR snapshots that are NOT tagged with "control".
query = OrQuery(
    AndQuery(ByMetadataQuery("project", "example1"), ByTagQuery("experiment")),
    NotQuery(ByTagQuery("control")),
)

# Evaluate the query
results = manager.query.evaluate(query)
print("Snapshots matching the logical query:", results)
# Output:
# Snapshots matching the logical query: ['snap1', 'snap3']
