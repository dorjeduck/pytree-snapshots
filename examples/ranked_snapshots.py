from snapshot_manager import SnapshotManager
import uuid


# Define a custom comparison function for ranking snapshots by accuracy
def cmp_by_accuracy(snapshot1, snapshot2):
    accuracy1 = snapshot1.metadata.get("accuracy", 0)
    accuracy2 = snapshot2.metadata.get("accuracy", 0)
    return accuracy1 - accuracy2


# Initialize the SnapshotManager with max_snapshots and custom comparison
manager = SnapshotManager(max_snapshots=3, cmp=cmp_by_accuracy)

# Example data: PyTrees with metadata containing accuracy
pytree1 = {"weights": [0.1, 0.2, 0.3]}
pytree2 = {"weights": [0.4, 0.5, 0.6]}
pytree3 = {"weights": [0.7, 0.8, 0.9]}
pytree4 = {"weights": [1.0, 1.1, 1.2]}

# Save snapshots with accuracy metadata
manager.save_snapshot(pytree1, snapshot_id="snap1", metadata={"accuracy": 0.8})
manager.save_snapshot(pytree2, snapshot_id="snap2", metadata={"accuracy": 0.9})
manager.save_snapshot(pytree3, snapshot_id="snap3", metadata={"accuracy": 0.7})

# List snapshots (should list 3 snapshots, ranked by age if no custom comparison)
print("Snapshots by age:", manager.list_snapshots())

# Save another snapshot with higher accuracy
manager.save_snapshot(pytree4, snapshot_id="snap4", metadata={"accuracy": 0.95})

# Since max_snapshots is 3, the snapshot with the lowest accuracy (snap3) will be removed
print(
    "Snapshots after saving snap4 (ranked by accuracy):",
    manager.get_ids_by_rank(),
)

# Retrieve the best snapshot (highest accuracy) and its metadata
best_snapshot_id = manager.get_ids_by_rank()[0]
best_snapshot = manager.get_snapshot(best_snapshot_id)
print(
    f"Best snapshot ID: {best_snapshot_id}, Best snapshot metadata: {best_snapshot.metadata}"
)
