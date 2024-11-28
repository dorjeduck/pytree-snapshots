from snapshot_manager import SnapshotManager

# Initialize the manager
manager = SnapshotManager()

manager.save_snapshot(
    {},
    snapshot_id="snap1",
    metadata={"accuracy": 0.85, "created_at": 1690000000.0},
    tags=["experiment", "draft"],
)
manager.save_snapshot(
    {},
    snapshot_id="snap2",
    metadata={"accuracy": 0.90, "created_at": 1695000000.0},
    tags=["draft"],
)
manager.save_snapshot(
    {},
    snapshot_id="snap3",
    metadata={"accuracy": 0.88, "created_at": 1790000000.0},
    tags=["final", "experiment", "published"],
)

snapshot_with_highest_accuracy = manager.query.by_cmp(
    lambda s1, s2: s1.metadata["accuracy"] >= s2.metadata["accuracy"]
)

print(f"Snapshot with highest accuracy: {snapshot_with_highest_accuracy}")

# Use a cmp to find the snapshot with the most tags
snapshot_with_most_tags = manager.query.by_cmp(
    lambda s1, s2: len(s1.tags) >= len(s2.tags)
)

print(f"Snapshot with most tags: {snapshot_with_most_tags}")

# Use a cmp to find the oldest snapshot
oldest_snapshot_id = manager.query.by_cmp(
    lambda s1, s2: s1.metadata["created_at"] <= s2.metadata["created_at"]
)

print(f"Oldest snapshot: {oldest_snapshot_id}")
