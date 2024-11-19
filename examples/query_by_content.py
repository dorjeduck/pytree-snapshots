# Example PyTreeSnapshotManager Usage
from snapshot_manager import SnapshotManager
import jax.numpy as jnp

# Initialize the manager
manager = SnapshotManager(max_snapshots=5)

# Save some snapshots
snapshot_id1 = manager.save_snapshot(
    {"a": jnp.array([1, 2, 3]), "b": 42}, snapshot_id="snap1"
)
snapshot_id2 = manager.save_snapshot(
    {"a": jnp.array([1, 2, 4]), "b": 42}, snapshot_id="snap2"
)
snapshot_id3 = manager.save_snapshot(
    {"a": jnp.array([1, 2, 3]), "b": 100}, snapshot_id="snap3"
)


# Define the query function
def query_func(pytree):
    """
    Example query function to find PyTrees where:
    - The key 'b' exists and its value is 42.
    """
    return pytree.get("b") == 42


# Use by_content to find matching snapshots
matching_snapshots = manager.query.by_content(query_func)

# Display the matching snapshot IDs
print(f"Snapshots matching the query: {matching_snapshots}")

# Example Output
# Snapshots matching the query: ['snapshot_id1', 'snapshot_id2']


def nested_query_func(pytree):
    """
    Example query function to find PyTrees where:
    - The key 'nested' exists and has a subkey 'key' with value 'target'.
    """
    nested = pytree.get("nested", {})
    return nested.get("key") == "target"


# Save a snapshot with a nested structure
snapshot_id4 = manager.save_snapshot(
    {"nested": {"key": "target"}, "b": 42}, snapshot_id="snap4"
)

# Use by_content with the nested query
nested_matches = manager.query.by_content(nested_query_func)

print(f"Snapshots matching the nested query: {nested_matches}")

# Example Output
# Snapshots matching the nested query: ['snapshot_id4']
