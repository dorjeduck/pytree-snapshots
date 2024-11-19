from snapshot_manager import PyTreeSnapshotManager

# Initialize the PyTree manager
manager = PyTreeSnapshotManager()

# Save snapshots with PyTree data
manager.save_snapshot(
    {"a": 1, "b": [2, 3]},
    snapshot_id="snap1",
    metadata={"project": "example1"},
)
manager.save_snapshot(
    {"x": 5, "y": {"z": 10}},
    snapshot_id="snap2",
    metadata={"project": "example2"},
)
manager.save_snapshot(
    {"c": [0, -1], "d": 7},
    snapshot_id="snap3",
    metadata={"project": "example1"},
)

# Query snapshots with any leaf value greater than 5
query = manager.query.by_leaf_value(lambda x: x > 5)
results = manager.query.evaluate(query)

print("Snapshots with a leaf value > 5:", results)
# Output: Snapshots with a leaf value > 5: ['snap2', 'snap3']
