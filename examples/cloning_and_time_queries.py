import time
from snapshot_manager import SnapshotManager
import jax.numpy as jnp

# Initialize the manager with a max_snapshots limit
manager = SnapshotManager(max_snapshots=10)

# Create initial PyTrees
pytree1 = {"a": jnp.array([1, 2, 3]), "b": jnp.array([4, 5, 6])}
pytree2 = {"a": jnp.array([10, 11, 12]), "b": jnp.array([13, 14, 15])}

# Save snapshots with metadata
snapshot_id1 = manager.save_snapshot(pytree1, metadata={"project": "exp1"})
time.sleep(1)  # Ensure a slight delay for distinct timestamps
snapshot_id2 = manager.save_snapshot(pytree2, metadata={"project": "exp2"})

# Clone the first snapshot
cloned_snapshot_id = manager.clone_snapshot(
    snapshot_id1, metadata={"cloned_from": snapshot_id1}
)
print(f"Cloned snapshot '{snapshot_id1}' to '{cloned_snapshot_id}'")

# List all snapshots
all_snapshots = manager.list_snapshots()
print("All snapshots:", all_snapshots)

# Get snapshots created in a specific time range
start_time = manager.storage.get_snapshot(snapshot_id1).timestamp
end_time = (
    manager.storage.get_snapshot(snapshot_id2).timestamp + 1
)  # Extend range slightly
time_range_snapshots = manager.query.by_time_range(start_time, end_time)
print("Snapshots created in time range:", time_range_snapshots)

# Retrieve cloned snapshot's PyTree
cloned_pytree = manager.get_snapshot(cloned_snapshot_id)
print(f"Cloned PyTree for '{cloned_snapshot_id}':", cloned_pytree)
