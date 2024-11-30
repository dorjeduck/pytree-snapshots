"""Example demonstrating tree replacement in PyTreeSnapshotManager.

This example shows:
1. Saving snapshots with PyTree structures
2. Applying transformations to all snapshots
3. Applying transformations to specific snapshots
4. Verifying the transformed data
"""

from snapshot_manager import PyTreeSnapshotManager
from jax import numpy as jnp

# Initialize the manager
print("\nInitializing PyTreeSnapshotManager...")
manager = PyTreeSnapshotManager()

# Save snapshots with PyTree structures
print("\nSaving snapshots with PyTree structures...")
pyt1 = {"a": jnp.array([1, 2, 3]), "b": jnp.array([4, 5, 6])}
pyt2 = {"c": jnp.array([7, 8, 9]), "d": jnp.array([10, 11, 12])}

snap1_id = manager.save_snapshot(pyt1, snapshot_id="snap1")
print("Saved snap1:", pyt1)

snap2_id = manager.save_snapshot(pyt2, snapshot_id="snap2")
print("Saved snap2:", pyt2)

# Define a transformation function for incrementing arrays
print("\nDefining transformation to increment arrays...")
def increment_array(x):
    """Add 1 to each element if x is an array"""
    return x + 1 if isinstance(x, jnp.ndarray) else x

# Replace trees in all snapshots
print("\nApplying increment transformation to all snapshots...")
manager.tree_replace(func=increment_array)

# Retrieve and validate transformed snapshots
print("\nVerifying transformed snapshots...")
transformed_snapshot1 = manager.get_snapshot(snap1_id)
print("Transformed snap1:", transformed_snapshot1.data)

transformed_snapshot2 = manager.get_snapshot(snap2_id)
print("Transformed snap2:", transformed_snapshot2.data)

# Test replacing a specific snapshot
print("\nDefining transformation to multiply arrays...")
def multiply_array(x):
    """Multiply each element by 2 if x is an array"""
    return x * 2 if isinstance(x, jnp.ndarray) else x

print("\nApplying multiply transformation to snap1 only...")
manager.tree_replace(func=multiply_array, snapshot_ids="snap1")

# Verify final results
print("\nVerifying final results...")
transformed_snapshot1 = manager.get_snapshot(snap1_id)
print("snap1 (after multiply):", transformed_snapshot1.data)

transformed_snapshot2 = manager.get_snapshot(snap2_id)
print("snap2 (unchanged):", transformed_snapshot2.data)

# Summary of transformations
print("\nSummary of transformations:")
print("1. All arrays were incremented by 1")
print("2. Arrays in snap1 were then multiplied by 2")
print("3. Arrays in snap2 remained at the incremented values")
