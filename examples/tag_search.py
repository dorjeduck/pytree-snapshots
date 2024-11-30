"""Example demonstrating tag operations and searching in SnapshotManager.

This example shows:
1. Saving snapshots with initial tags
2. Adding and removing tags from existing snapshots
3. Searching by single and multiple tags
4. Using tag-based queries with AND/OR logic
"""

from snapshot_manager import SnapshotManager
from snapshot_manager.query import ByTagQuery, AndQuery, OrQuery

# Initialize manager
manager = SnapshotManager()

# Save snapshots with initial tags
manager.save_snapshot(
    {"model": {"weights": [1.0, 2.0]}},
    snapshot_id="baseline",
    tags=["experiment", "baseline"],
    metadata={"accuracy": 0.85}
)

manager.save_snapshot(
    {"model": {"weights": [1.1, 2.1]}},
    snapshot_id="variant_a",
    tags=["experiment", "variant"],
    metadata={"accuracy": 0.87}
)

manager.save_snapshot(
    {"model": {"weights": [1.2, 2.2]}},
    snapshot_id="variant_b",
    tags=["experiment", "variant", "best"],
    metadata={"accuracy": 0.90}
)

# Add tags to existing snapshots
manager.add_tags("variant_b", ["published", "final"])

# Remove tags from snapshots
manager.remove_tags("variant_a", ["variant"])
manager.add_tags("variant_a", ["abandoned"])

# Search by single tag
experiment_snapshots = manager.query.evaluate(ByTagQuery("experiment"))
print("\nAll experiment snapshots:", experiment_snapshots)

# Search for snapshots with multiple tags (AND logic)
query = AndQuery(
    ByTagQuery("experiment"),
    ByTagQuery("variant")
)
variant_experiments = manager.query.evaluate(query)
print("\nExperiment variants:", variant_experiments)

# Search for snapshots with either tag (OR logic)
query = OrQuery(
    ByTagQuery("best"),
    ByTagQuery("baseline")
)
reference_snapshots = manager.query.evaluate(query)
print("\nReference snapshots (best or baseline):", reference_snapshots)

# Show all tags for a specific snapshot
variant_b_tags = manager.get_tags("variant_b")
print("\nTags for variant_b:", variant_b_tags)
