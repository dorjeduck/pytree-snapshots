from pytree_snapshots import PytreeSnapshotManager

# Create a manager and save a snapshot
manager = PytreeSnapshotManager()

manager.save_snapshot({"a": 1, "b": "ignore this"}, snapshot_id="snap1")
manager.save_snapshot({"a": 2, "b": "ignore this too"}, snapshot_id="snap2")

# Compare with a condition to ignore string differences
differences = manager.compare_snapshots(
    "snap1",
    "snap2",
    condition=lambda x: not isinstance(x, str),  # Ignore strings during comparison
)
print("Differences ignoring strings:", differences)
