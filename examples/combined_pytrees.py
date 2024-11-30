"""Example demonstrating PyTree operations in SnapshotManager.

This example shows:
1. Working with structured PyTree data (like neural network parameters)
2. Combining multiple PyTrees using different strategies via the manager
3. Applying transformations to PyTrees using the manager
4. Practical use cases in machine learning contexts
"""

import jax.numpy as jnp
from snapshot_manager.pytree_snapshot_manager import PyTreeSnapshotManager

# Initialize the PyTreeSnapshotManager
manager = PyTreeSnapshotManager()

# Create some example model parameters (PyTree structures)
def create_model_params(scale=1.0):
    return {
        "encoder": {
            "conv1": {
                "weights": jnp.array([1.0, 2.0, 3.0]) * scale,
                "bias": jnp.array([0.1]) * scale
            },
            "conv2": {
                "weights": jnp.array([4.0, 5.0, 6.0]) * scale,
                "bias": jnp.array([0.2]) * scale
            }
        },
        "decoder": {
            "dense1": {
                "weights": jnp.array([7.0, 8.0]) * scale,
                "bias": jnp.array([0.3]) * scale
            }
        }
    }

# Save multiple model checkpoints with metadata
checkpoints = []
for i, scale in enumerate([1.0, 1.1, 1.2]):
    params = create_model_params(scale)
    snapshot_id = f"model_checkpoint_{i}"
    manager.save_snapshot(
        params,
        snapshot_id=snapshot_id,
        metadata={"epoch": i, "loss": 1.0 - 0.1 * i},
        tags=["checkpoint"]
    )
    checkpoints.append(snapshot_id)

# Define different combining strategies
def average_leaves(leaves):
    """Average corresponding leaves across PyTrees."""
    return sum(leaves) / len(leaves)

def weighted_average_leaves(leaves, weights):
    """Weighted average of leaves based on model performance."""
    return sum(w * l for w, l in zip(weights, leaves)) / sum(weights)

# Combine snapshots using simple averaging
print("\nSimple averaging of all checkpoints:")
averaged_params = manager.tree_combine(
    snapshot_ids=checkpoints,
    combine_fn=average_leaves
)
print("Averaged encoder conv1 weights:", averaged_params["encoder"]["conv1"]["weights"])

# Combine snapshots using weighted averaging based on loss
snapshots = [manager.get_snapshot(sid) for sid in checkpoints]
weights = [1.0 / snap.metadata["loss"] for snap in snapshots]  # Lower loss = higher weight

print("\nWeighted averaging based on loss:")
weighted_params = manager.tree_combine(
    snapshot_ids=checkpoints,
    combine_fn=lambda leaves: weighted_average_leaves(leaves, weights)
)
print("Weighted encoder conv1 weights:", weighted_params["encoder"]["conv1"]["weights"])

# Apply transformation to all leaves in a PyTree
def scale_parameters(leaf):
    """Scale all parameters by 0.5"""
    return leaf * 0.5 if isinstance(leaf, jnp.ndarray) else leaf

print("\nScaling all parameters:")
# Transform the first checkpoint's parameters using the manager
scaled_params = manager.tree_map(
    func=scale_parameters,
    snapshot_ids=checkpoints[0]  # Pass single ID as string
)
print("Scaled encoder conv1 weights:", scaled_params["encoder"]["conv1"]["weights"])

# Save the transformed parameters as a new snapshot
manager.save_snapshot(
    scaled_params,
    snapshot_id="scaled_checkpoint",
    metadata={"transformation": "scaled_by_0.5"},
    tags=["transformed"]
)

# Compare original vs transformed parameters
original = manager.get_snapshot(checkpoints[0]).data
transformed = manager.get_snapshot("scaled_checkpoint").data
print("\nParameter comparison:")
print("Original conv1 weights:", original["encoder"]["conv1"]["weights"])
print("Scaled conv1 weights:", transformed["encoder"]["conv1"]["weights"])
