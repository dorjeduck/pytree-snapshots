import jax.numpy as jnp
from snapshot_manager.pytree_snapshot_manager import PyTreeSnapshotManager

# Initialize the PyTreeSnapshotManager
manager = PyTreeSnapshotManager()

# Save a few snapshots with PyTree structures (e.g., neural network weights)
snapshot1 = {"layer1": jnp.array([1.0, 2.0]), "layer2": jnp.array([3.0])}
snapshot2 = {"layer1": jnp.array([4.0, 5.0]), "layer2": jnp.array([6.0])}
snapshot3 = {"layer1": jnp.array([7.0, 8.0]), "layer2": jnp.array([9.0])}

manager.save_snapshot(snapshot1, snapshot_id="snapshot1")
manager.save_snapshot(snapshot2, snapshot_id="snapshot2")
manager.save_snapshot(snapshot3, snapshot_id="snapshot3")

# Custom combine function: average corresponding leaves
def average_leaves(leaves):
    return sum(leaves) / len(leaves)

# Combine the snapshots using the average function
combined_pytree = manager.combine_snapshots(
    snapshot_ids=["snapshot1", "snapshot2", "snapshot3"], combine_fn=average_leaves
)

# Print the result
print("Combined PyTree:", combined_pytree)