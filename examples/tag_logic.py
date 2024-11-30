"""Example demonstrating tag logic in SnapshotManager.

This example shows:
1. Saving snapshots with different tag combinations
2. Querying snapshots using AND logic (all tags must match)
3. Querying snapshots using OR logic (any tag can match)
4. Displaying and verifying tag-based query results
"""

from snapshot_manager.snapshot_manager import SnapshotManager

# Initialize the SnapshotManager
print("\nInitializing SnapshotManager...")
manager = SnapshotManager()

# Save snapshots with specific tags
print("\nSaving snapshots with different tag combinations...")
snapshot1_id = manager.save_snapshot({"data": "snapshot1"}, tags=["important", "completed"])
print("Saved snapshot1 with tags: [important, completed]")

snapshot2_id = manager.save_snapshot({"data": "snapshot2"}, tags=["important"])
print("Saved snapshot2 with tags: [important]")

snapshot3_id = manager.save_snapshot({"data": "snapshot3"}, tags=["completed"])
print("Saved snapshot3 with tags: [completed]")

snapshot4_id = manager.save_snapshot({"data": "snapshot4"}, tags=["other"])
print("Saved snapshot4 with tags: [other]")

# Function to print snapshot IDs and their associated tags
def print_snapshots_with_tags(manager):
    print("\nAll Snapshots and Their Tags:")
    for snapshot_id in manager.list_snapshots():
        tags = manager.get_tags(snapshot_id)
        data = manager.get_snapshot(snapshot_id).data
        print(f"Snapshot ID: {snapshot_id}")
        print(f"  Tags: {tags}")
        print(f"  Data: {data}")

# Display all snapshots and their tags
print_snapshots_with_tags(manager)

# Query snapshots matching all specified tags (AND logic)
print("\nQuerying snapshots with AND logic...")
print("Finding snapshots with BOTH 'important' AND 'completed' tags:")
result_and = manager.query.by_tags(["important", "completed"], require_all=True)
for snapshot_id in result_and:
    snapshot = manager.get_snapshot(snapshot_id)
    tags = manager.get_tags(snapshot_id)
    print(f"\nSnapshot ID: {snapshot_id}")
    print(f"  Tags: {tags}")
    print(f"  Data: {snapshot.data}")

# Query snapshots matching any of the specified tags (OR logic)
print("\nQuerying snapshots with OR logic...")
print("Finding snapshots with EITHER 'important' OR 'completed' tags:")
result_or = manager.query.by_tags(["important", "completed"], require_all=False)
for snapshot_id in result_or:
    snapshot = manager.get_snapshot(snapshot_id)
    tags = manager.get_tags(snapshot_id)
    print(f"\nSnapshot ID: {snapshot_id}")
    print(f"  Tags: {tags}")
    print(f"  Data: {snapshot.data}")

# Summary
print("\nSummary:")
print(f"Total snapshots: {len(manager.list_snapshots())}")
print(f"Snapshots with both 'important' AND 'completed': {len(result_and)}")
print(f"Snapshots with either 'important' OR 'completed': {len(result_or)}")
