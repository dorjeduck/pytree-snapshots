"""Example demonstrating snapshot cloning and time-based queries.

This example shows:
1. Saving snapshots with timestamps
2. Cloning existing snapshots (copying data and metadata)
3. Querying snapshots by time range
4. Basic snapshot listing and retrieval
"""

import time
from snapshot_manager import SnapshotManager
import jax.numpy as jnp

# Initialize the manager with a max_snapshots limit
manager = SnapshotManager(max_snapshots=10)

# Create initial PyTrees
pytree1 = {"a": jnp.array([1, 2, 3]), "b": jnp.array([4, 5, 6])}
pytree2 = {"a": jnp.array([10, 11, 12]), "b": jnp.array([13, 14, 15])}

# Save snapshots with metadata
print("\nSaving original snapshots...")
snapshot_id1 = manager.save_snapshot(pytree1, metadata={"project": "exp1"})
print(f"Saved snapshot '{snapshot_id1}' with project 'exp1'")

time.sleep(1)  # Ensure a slight delay for distinct timestamps
snapshot_id2 = manager.save_snapshot(pytree2, metadata={"project": "exp2"})
print(f"Saved snapshot '{snapshot_id2}' with project 'exp2'")

# Clone the first snapshot by retrieving and saving it as a new snapshot
print("\nCloning snapshot...")
original_snapshot = manager.get_snapshot(snapshot_id1)
cloned_snapshot_id = manager.save_snapshot(
    original_snapshot.data, metadata=original_snapshot.metadata
)
print(f"Cloned snapshot '{snapshot_id1}' to '{cloned_snapshot_id}'")

# List all snapshots
print("\nListing all snapshots...")
all_snapshots = manager.list_snapshots()
print("All snapshots:", all_snapshots)

# Get snapshots created in a specific time range
print("\nQuerying by time range...")
start_time = manager.storage.get_snapshot(snapshot_id1).timestamp
end_time = manager.storage.get_snapshot(snapshot_id2).timestamp + 1  # Extend range slightly

time_range_snapshots = manager.query.by_time_range(start_time, end_time)
print(f"Snapshots created between {start_time} and {end_time}:", time_range_snapshots)

# Verify cloned snapshot's data matches original
original_data = manager.get_snapshot(snapshot_id1).data
cloned_data = manager.get_snapshot(cloned_snapshot_id).data
print("\nVerifying clone...")
print("Original array:", original_data["a"])
print("Cloned array:", cloned_data["a"])