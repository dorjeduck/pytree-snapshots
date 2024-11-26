from snapshot_manager import PyTreeSnapshotManager
from jax import numpy as jnp

manager = PyTreeSnapshotManager()

# Save snapshots with PyTree structures
pyt1 = {"a": jnp.array([1, 2, 3]), "b": jnp.array([4, 5, 6])}
pyt2 = {"c": jnp.array([7, 8, 9]), "d": jnp.array([10, 11, 12])}

snap1_id = manager.save_snapshot(pyt1, snapshot_id="snap1")
snap2_id = manager.save_snapshot(pyt2, snapshot_id="snap2")


# Define a transformation function
def increment_array(x):
    return x + 1 if isinstance(x, jnp.ndarray) else x


# Replace trees in all snapshots
manager.tree_replace(func=increment_array)

# Retrieve and validate transformed snapshots
transformed_snapshot1 = manager.get_snapshot(snap1_id)
transformed_snapshot2 = manager.get_snapshot(snap2_id)

print(transformed_snapshot1.data)
print(transformed_snapshot2.data)


# Test replacing a specific snapshot
def multiply_array(x):
    return x * 2 if isinstance(x, jnp.ndarray) else x


manager.tree_replace(func=multiply_array, snapshot_ids="snap1")

transformed_snapshot1 = manager.get_snapshot(snap1_id)
transformed_snapshot2 = manager.get_snapshot(snap2_id)

print(transformed_snapshot1.data)
print(transformed_snapshot2.data)
