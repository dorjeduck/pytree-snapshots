from pytree_snapshots import PytreeSnapshotManager

manager = PytreeSnapshotManager()

# Save snapshots
pytree1 = {"a": 1, "b": 2}
pytree2 = {"a": 1, "b": 3}
manager.save_snapshot(pytree1, snapshot_id="snap1")
manager.save_snapshot(pytree2, snapshot_id="snap2")

# Compare snapshots
differences = manager.compare_snapshots("snap1", "snap2")
print("Differences:", differences)