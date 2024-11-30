"""Example demonstrating leaf node updates in PyTreeSnapshotManager.

This example shows:
1. Saving snapshots with nested structures
2. Applying transformations to leaf nodes
3. Verifying the transformations were applied correctly
"""

from snapshot_manager import PyTreeSnapshotManager
import jax.numpy as jnp

# Initialize the manager
print("\nInitializing PyTreeSnapshotManager...")
manager = PyTreeSnapshotManager()

# Save snapshots with nested structures
print("\nSaving snapshots with nested structures...")
data1 = {"a": jnp.array(1), "b": {"x": jnp.array(2)}}
snapshot_id1 = manager.save_snapshot(data1, snapshot_id="snap1")
print("Saved snap1:", data1)

data2 = {"c": jnp.array(3), "d": {"y": jnp.array(4)}}
snapshot_id2 = manager.save_snapshot(data2, snapshot_id="snap2")
print("Saved snap2:", data2)

# Define transformation function
print("\nDefining transformation to double leaf values...")
def double_value(x):
    """Double the value of a leaf node"""
    if isinstance(x, jnp.ndarray):
        return x * 2
    return x

# Apply transformation to double each leaf value
print("\nApplying transformation to both snapshots...")
manager.tree_replace(func=double_value, snapshot_ids=[snapshot_id1, snapshot_id2])

# Retrieve and verify transformed snapshots
print("\nVerifying transformations...")

# Check snap1
print("\nChecking snap1...")
snapshot1 = manager.get_snapshot(snapshot_id1, deepcopy=False)
print("Original data:", data1)
print("Transformed data:", snapshot1.data)

try:
    assert float(snapshot1.data["a"]) == 2 and float(snapshot1.data["b"]["x"]) == 4
    print("✓ Transformation successful: all values doubled correctly")
except AssertionError:
    print("✗ Error: Snapshot1 not transformed correctly")

# Check snap2
print("\nChecking snap2...")
snapshot2 = manager.get_snapshot(snapshot_id2, deepcopy=False)
print("Original data:", data2)
print("Transformed data:", snapshot2.data)

try:
    assert float(snapshot2.data["c"]) == 6 and float(snapshot2.data["d"]["y"]) == 8
    print("✓ Transformation successful: all values doubled correctly")
except AssertionError:
    print("✗ Error: Snapshot2 not transformed correctly")

# Summary
print("\nSummary of transformations:")
print("1. All leaf values were doubled")
print("2. Nested structure was preserved")
print("3. Both snapshots were transformed correctly")
