from snapshot_manager import PyTreeSnapshotManager

manager = PyTreeSnapshotManager()

# Save snapshots
snapshot_id1 = manager.save_snapshot({"a": 1, "b": {"x": 2}})
snapshot_id2 = manager.save_snapshot({"c": 3, "d": {"y": 4}})

# Apply an in-place transformation to double each leaf value
manager.tree_map(lambda x: x * 2,[snapshot_id1, snapshot_id2])

# Retrieve the snapshots and verify transformation
snapshot1 = manager.get_snapshot(snapshot_id1, deepcopy=False)
snapshot2 = manager.get_snapshot(snapshot_id2, deepcopy=False)

print(snapshot1.data)

assert (
    snapshot1.data["a"] == 2 and snapshot1.data["b"]["x"] == 4
), "Snapshot1 not transformed correctly."
assert (
    snapshot2.data["c"] == 6 and snapshot2.data["d"]["y"] == 8
), "Snapshot2 not transformed correctly."
