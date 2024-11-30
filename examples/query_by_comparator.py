"""Example demonstrating querying by comparator in SnapshotManager.

This example shows:
1. Using custom comparators to find snapshots with specific criteria
2. Finding snapshots with the highest accuracy, most tags, and oldest creation time
3. Demonstrating the use of comparator functions
"""

from snapshot_manager import SnapshotManager

# Initialize the manager
manager = SnapshotManager()

# Save snapshots with metadata and tags
print("\nSaving test snapshots...")
manager.save_snapshot(
    {},
    snapshot_id="snap1",
    metadata={"accuracy": 0.85, "created_at": 1690000000.0},
    tags=["experiment", "draft"],
)
print("Saved snap1: accuracy=0.85, created_at=1690000000.0, tags=[experiment, draft]")

manager.save_snapshot(
    {},
    snapshot_id="snap2",
    metadata={"accuracy": 0.90, "created_at": 1695000000.0},
    tags=["draft"],
)
print("Saved snap2: accuracy=0.90, created_at=1695000000.0, tags=[draft]")

manager.save_snapshot(
    {},
    snapshot_id="snap3",
    metadata={"accuracy": 0.88, "created_at": 1790000000.0},
    tags=["final", "experiment", "published"],
)
print("Saved snap3: accuracy=0.88, created_at=1790000000.0, tags=[final, experiment, published]")

# Find snapshot with the highest accuracy
print("\nFinding snapshot with highest accuracy...")
def accuracy_comparator(s1, s2):
    """Return True if s1's accuracy is higher than or equal to s2's"""
    return s1.metadata["accuracy"] >= s2.metadata["accuracy"]

snapshot_with_highest_accuracy = manager.query.by_cmp(accuracy_comparator)
print(f"Snapshot with highest accuracy: {snapshot_with_highest_accuracy}")

# Find snapshot with the most tags
print("\nFinding snapshot with most tags...")
def tag_count_comparator(s1, s2):
    """Return True if s1 has more tags than or equal to s2"""
    return len(s1.tags) >= len(s2.tags)

snapshot_with_most_tags = manager.query.by_cmp(tag_count_comparator)
print(f"Snapshot with most tags: {snapshot_with_most_tags}")

# Find the oldest snapshot
print("\nFinding oldest snapshot...")
def timestamp_comparator(s1, s2):
    """Return True if s1 is older than or equal to s2"""
    return s1.metadata["created_at"] <= s2.metadata["created_at"]

oldest_snapshot_id = manager.query.by_cmp(timestamp_comparator)
print(f"Oldest snapshot: {oldest_snapshot_id}")
