from snapshot_manager import PyTreeSnapshotManager

manager = PyTreeSnapshotManager()

# Save snapshots
snapshot_id1 = manager.save_snapshot({"a": 1, "b": {"x": 2}})
snapshot_id2 = manager.save_snapshot({"c": 3, "d": {"y": 4}})

# Apply an in-place transformation to double each leaf value
manager.update_leaf_nodes([snapshot_id1, snapshot_id2], lambda x: x * 2)

# Retrieve the snapshots and verify transformation
snapshot1 = manager.get_snapshot(snapshot_id1, deepcopy=False)
snapshot2 = manager.get_snapshot(snapshot_id2, deepcopy=False)

print(snapshot1)

assert (
    snapshot1["a"] == 2 and snapshot1["b"]["x"] == 4
), "Snapshot1 not transformed correctly."
assert (
    snapshot2["c"] == 6 and snapshot2["d"]["y"] == 8
), "Snapshot2 not transformed correctly."
