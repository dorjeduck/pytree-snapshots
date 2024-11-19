from snapshot_manager import SnapshotManager
from snapshot_manager.query import SnapshotQuery


class LoggingSnapshotQuery(SnapshotQuery):
    """
    A custom SnapshotQuery that logs all query operations.
    """

    def __init__(self, snapshots):
        self.snapshots = snapshots

    def by_metadata(self, key, value=None):
        print(f"Querying by metadata: {key} = {value}")
        return [
            snapshot_id
            for snapshot_id, snapshot in self.snapshots.items()
            if key in snapshot.metadata
            and (value is None or snapshot.metadata[key] == value)
        ]

    def by_tag(self, tag):
        print(f"Querying by tag: {tag}")
        return [
            snapshot_id
            for snapshot_id, snapshot in self.snapshots.items()
            if tag in snapshot.tags
        ]


# Inject the custom query class into SnapshotManager
manager = SnapshotManager(query_class=LoggingSnapshotQuery)

# Save some snapshots
manager.save_snapshot(
    {"a": 1, "b": 2},
    metadata={"project": "example1"},
    tags=["experiment"],
    snapshot_id="snap1",
)
manager.save_snapshot(
    {"x": 10, "y": 20},
    metadata={"project": "example2"},
    tags=["control"],
    snapshot_id="snap2",
)

# Perform queries

print("Metadata query results:", manager.query.by_metadata("project", "example1"))
# Output:
# Querying by metadata: project = example1
# Metadata query results: ['snap1']

print("Tag query results:", manager.query.by_tag("control"))
# Output:
# Querying by tag: control
# Tag query results: ['snap2']
