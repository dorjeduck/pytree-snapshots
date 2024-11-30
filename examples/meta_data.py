"""Example demonstrating metadata operations in SnapshotManager.

This example shows:
1. Saving snapshots with initial metadata
2. Retrieving metadata
3. Updating metadata (both adding new fields and modifying existing ones)
4. Using nested metadata structures
5. Querying snapshots by metadata
"""

from snapshot_manager import SnapshotManager
from snapshot_manager.query import ByMetadataQuery, AndQuery

# Initialize manager
manager = SnapshotManager()

# Save snapshots with structured metadata
experiment_metadata = {
    "experiment": {
        "name": "model_training",
        "version": "1.0",
        "parameters": {
            "learning_rate": 0.001,
            "batch_size": 32
        }
    },
    "metrics": {
        "accuracy": 0.85,
        "loss": 0.15
    }
}

snapshot_id = manager.save_snapshot(
    {"weights": [1.0, 2.0, 3.0]},
    snapshot_id="model_v1",
    metadata=experiment_metadata
)

# Retrieve and display initial metadata
metadata = manager.get_metadata("model_v1")
print("\nInitial metadata:")
print("Experiment name:", metadata["experiment"]["name"])
print("Learning rate:", metadata["experiment"]["parameters"]["learning_rate"])
print("Accuracy:", metadata["metrics"]["accuracy"])

# Update metadata - add new information while preserving existing
update = {
    "metrics": {
        "accuracy": 0.90,  # Update existing value
        "f1_score": 0.88   # Add new metric
    },
    "timestamp": "2024-01-20",  # Add new top-level field
    "status": "completed"
}

manager.update_metadata("model_v1", update)

# Show updated metadata
updated = manager.get_metadata("model_v1")
print("\nUpdated metadata:")
print("New accuracy:", updated["metrics"]["accuracy"])
print("New F1 score:", updated["metrics"]["f1_score"])
print("Status:", updated["status"])

# Demonstrate metadata querying
# Save another snapshot for comparison
manager.save_snapshot(
    {"weights": [1.1, 2.1, 3.1]},
    snapshot_id="model_v2",
    metadata={
        "experiment": {"name": "model_training", "version": "2.0"},
        "metrics": {"accuracy": 0.92}
    }
)

# First find all training snapshots
training_snapshots = manager.query.evaluate(
    ByMetadataQuery("experiment.name", "model_training")
)

# Then filter by accuracy manually
high_accuracy_snapshots = [
    sid for sid in training_snapshots
    if manager.get_metadata(sid).get("metrics", {}).get("accuracy", 0) >= 0.90
]

print("\nSnapshots with accuracy >= 90%:", high_accuracy_snapshots)
