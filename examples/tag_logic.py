from snapshot_manager.snapshot_manager import SnapshotManager

# Initialize the SnapshotManager
manager = SnapshotManager()

# Save snapshots with specific tags
snapshot1_id = manager.save_snapshot({"data": "snapshot1"}, tags=["important", "completed"])
snapshot2_id = manager.save_snapshot({"data": "snapshot2"}, tags=["important"])
snapshot3_id = manager.save_snapshot({"data": "snapshot3"}, tags=["completed"])
snapshot4_id = manager.save_snapshot({"data": "snapshot4"}, tags=["other"])

# Function to print snapshot IDs and their associated tags
def print_snapshots_with_tags(manager):
    print("\nAll Snapshots:")
    for snapshot_id in manager.list_snapshots():
        tags = manager.get_tags(snapshot_id)
        print(f"Snapshot ID: {snapshot_id}, Tags: {tags}")

# Display all snapshots and their tags
print_snapshots_with_tags(manager)

# Query snapshots matching all specified tags (AND logic)
print("\nSnapshots with tags 'important' AND 'completed':")
result_and = manager.query.by_tags(["important", "completed"], require_all=True)
for snapshot_id in result_and:
    print(f"Snapshot ID: {snapshot_id}, Content: {manager.get_snapshot(snapshot_id).data}")

# Query snapshots matching any of the specified tags (OR logic)
print("\nSnapshots with tags 'important' OR 'completed':")
result_or = manager.query.by_tags(["important", "completed"], require_all=False)
for snapshot_id in result_or:
    print(f"Snapshot ID: {snapshot_id}, Content: {manager.get_snapshot(snapshot_id).data}")

