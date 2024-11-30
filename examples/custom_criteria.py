"""Example demonstrating custom comparison criteria in SnapshotManager.

This example shows:
1. Using custom comparison functions with by_cmp
2. Finding snapshots by comparing metadata values
3. Finding snapshots by comparing tag counts
4. Finding snapshots by comparing timestamps
"""

from snapshot_manager import SnapshotManager

# Initialize the manager
manager = SnapshotManager()

# Save snapshots with metadata and tags
print("\nSaving snapshots with different metadata and tags...")
manager.save_snapshot(
    {},
    snapshot_id="snap1",
    metadata={"accuracy": 0.85, "created_at": 1690000000.0},
    tags=["experiment", "draft"],
)
manager.save_snapshot(
    {},
    snapshot_id="snap2",
    metadata={"accuracy": 0.90, "created_at": 1695000000.0},
    tags=["draft"],
)
manager.save_snapshot(
    {},
    snapshot_id="snap3",
    metadata={"accuracy": 0.88, "created_at": 1790000000.0},
    tags=["final", "experiment", "published"],
)

# Find snapshot with the highest accuracy
print("\nFinding snapshot with highest accuracy...")
def accuracy_comparator(s1, s2):
    """Return True if s1's accuracy is higher than or equal to s2's"""
    return s1.metadata["accuracy"] >= s2.metadata["accuracy"]

snapshot_with_highest_accuracy = manager.query.by_cmp(accuracy_comparator)
print(f"Snapshot with highest accuracy: {snapshot_with_highest_accuracy}")
print(f"Accuracy value: {manager.get_metadata(snapshot_with_highest_accuracy)['accuracy']}")

# Find snapshot with the most tags
print("\nFinding snapshot with most tags...")
def tag_count_comparator(s1, s2):
    """Return True if s1 has more tags than or equal to s2"""
    return len(s1.tags) >= len(s2.tags)

snapshot_with_most_tags = manager.query.by_cmp(tag_count_comparator)
print(f"Snapshot with most tags: {snapshot_with_most_tags}")
print(f"Tags: {manager.get_tags(snapshot_with_most_tags)}")

# Find the oldest snapshot
print("\nFinding oldest snapshot...")
def timestamp_comparator(s1, s2):
    """Return True if s1 is older than or equal to s2"""
    return s1.metadata["created_at"] <= s2.metadata["created_at"]

oldest_snapshot_id = manager.query.by_cmp(timestamp_comparator)
print(f"Oldest snapshot: {oldest_snapshot_id}")
print(f"Creation time: {manager.get_metadata(oldest_snapshot_id)['created_at']}")
