from pytree_snapshots import PytreeSnapshotManager

manager = PytreeSnapshotManager()

# Save snapshots with tags
manager.save_snapshot({"a": 1}, snapshot_id="baseline", tags=["experiment", "baseline"])
manager.save_snapshot({"a": 2}, snapshot_id="variant", tags=["experiment", "variant"])

# Search by tags
experiment_ids = manager.find_snapshots_by_tag("experiment")
print("Snapshots tagged as 'experiment':", experiment_ids)