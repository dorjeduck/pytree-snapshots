"""Example demonstrating custom query class in SnapshotManager.

This example shows:
1. Creating a custom SnapshotQuery class
2. Adding logging to query operations
3. Injecting custom query class into SnapshotManager
4. Using the custom query methods
"""

from snapshot_manager import SnapshotManager
from snapshot_manager.query import SnapshotQuery


class LoggingSnapshotQuery(SnapshotQuery):
    """A custom SnapshotQuery that logs all query operations."""

    def __init__(self, snapshots):
        self.snapshots = snapshots

    def by_metadata(self, key, value=None):
        """Query snapshots by metadata with logging."""
        print(f"\nQuerying by metadata: {key} = {value}")
        matching_snapshots = [
            snapshot_id
            for snapshot_id, snapshot in self.snapshots.items()
            if key in snapshot.metadata
            and (value is None or snapshot.metadata[key] == value)
        ]
        print(f"Found {len(matching_snapshots)} matching snapshots")
        return matching_snapshots

    def by_tags(self, tag):
        """Query snapshots by tag with logging."""
        print(f"\nQuerying by tag: {tag}")
        matching_snapshots = [
            snapshot_id
            for snapshot_id, snapshot in self.snapshots.items()
            if tag in snapshot.tags
        ]
        print(f"Found {len(matching_snapshots)} matching snapshots")
        return matching_snapshots


# Initialize manager with custom query class
print("Initializing SnapshotManager with LoggingSnapshotQuery...")
manager = SnapshotManager(query_class=LoggingSnapshotQuery)

# Save snapshots with different metadata and tags
print("\nSaving test snapshots...")
manager.save_snapshot(
    {"a": 1, "b": 2},
    metadata={"project": "example1", "type": "experiment"},
    tags=["experiment", "v1"],
    snapshot_id="snap1",
)
manager.save_snapshot(
    {"x": 10, "y": 20},
    metadata={"project": "example2", "type": "control"},
    tags=["control", "v1"],
    snapshot_id="snap2",
)

# Demonstrate metadata queries
print("\nDemonstrating metadata queries...")
example1_snapshots = manager.query.by_metadata("project", "example1")
print("Snapshots from example1 project:", example1_snapshots)

# Demonstrate tag queries
print("\nDemonstrating tag queries...")
control_snapshots = manager.query.by_tags("control")
print("Control group snapshots:", control_snapshots)

# Show that both snapshots have the v1 tag
print("\nDemonstrating common tag query...")
v1_snapshots = manager.query.by_tags("v1")
print("Version 1 snapshots:", v1_snapshots)
