"""Example demonstrating ranked snapshots in SnapshotManager.

This example shows:
1. Using a custom comparison function to rank snapshots by accuracy
2. Managing snapshots with a maximum limit and automatic removal
3. Retrieving snapshots based on their rank
"""

from snapshot_manager import SnapshotManager
import uuid


# Define a custom comparison function for ranking snapshots by accuracy
def cmp_by_accuracy(snapshot1, snapshot2):
    accuracy1 = snapshot1.metadata.get("accuracy", 0)
    accuracy2 = snapshot2.metadata.get("accuracy", 0)
    return accuracy1 - accuracy2


# Initialize the SnapshotManager with max_snapshots and custom comparison
print("\nInitializing SnapshotManager with max_snapshots=3 and custom comparator...")
manager = SnapshotManager(max_snapshots=3, cmp=cmp_by_accuracy)

# Example data: PyTrees with metadata containing accuracy
pytree1 = {"weights": [0.1, 0.2, 0.3]}
pytree2 = {"weights": [0.4, 0.5, 0.6]}
pytree3 = {"weights": [0.7, 0.8, 0.9]}
pytree4 = {"weights": [1.0, 1.1, 1.2]}

# Save snapshots with accuracy metadata
print("\nSaving snapshots with accuracy metadata...")
manager.save_snapshot(pytree1, snapshot_id="snap1", metadata={"accuracy": 0.8})
print("Saved snap1 with accuracy 0.8")

manager.save_snapshot(pytree2, snapshot_id="snap2", metadata={"accuracy": 0.9})
print("Saved snap2 with accuracy 0.9")

manager.save_snapshot(pytree3, snapshot_id="snap3", metadata={"accuracy": 0.7})
print("Saved snap3 with accuracy 0.7")

# List snapshots (should list 3 snapshots, ranked by age if no custom comparison)
print("\nSnapshots by age:", manager.list_snapshots())

# Save another snapshot with higher accuracy
print("\nSaving snap4 with higher accuracy 0.95...")
manager.save_snapshot(pytree4, snapshot_id="snap4", metadata={"accuracy": 0.95})

# Since max_snapshots is 3, the snapshot with the lowest accuracy (snap3) will be removed
print("\nSnapshots after saving snap4 (ranked by accuracy):")
ranked_snapshots = manager.get_ids_by_rank()
print(ranked_snapshots)

# Retrieve the best snapshot (highest accuracy) and its metadata
best_snapshot_id = ranked_snapshots[0]
best_snapshot = manager.get_snapshot(best_snapshot_id)
print(f"\nBest snapshot ID: {best_snapshot_id}, Best snapshot metadata: {best_snapshot.metadata}")
