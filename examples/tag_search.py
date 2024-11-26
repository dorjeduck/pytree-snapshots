from snapshot_manager import SnapshotManager

manager = SnapshotManager()

# Save snapshots with tags
manager.save_snapshot({"a": 1}, snapshot_id="baseline", tags=["experiment", "baseline"])
manager.save_snapshot({"a": 2}, snapshot_id="variant", tags=["experiment", "variant"])

# Search by tags
experiment_ids = manager.query.by_tags("experiment")
print("Snapshots tagged as 'experiment':", experiment_ids)
