"""Example demonstrating querying by content in SnapshotManager.

This example shows:
1. Using content-based queries to find snapshots
2. Defining custom query functions for specific conditions
3. Querying both flat and nested structures
"""

# Example PyTreeSnapshotManager Usage
from snapshot_manager import SnapshotManager
import jax.numpy as jnp

# Initialize the manager
manager = SnapshotManager(max_snapshots=5)

# Save some snapshots
print("\nSaving test snapshots...")
snapshot_id1 = manager.save_snapshot(
    {"a": jnp.array([1, 2, 3]), "b": 42}, snapshot_id="snap1"
)
print("Saved snap1: a=[1, 2, 3], b=42")

snapshot_id2 = manager.save_snapshot(
    {"a": jnp.array([1, 2, 4]), "b": 42}, snapshot_id="snap2"
)
print("Saved snap2: a=[1, 2, 4], b=42")

snapshot_id3 = manager.save_snapshot(
    {"a": jnp.array([1, 2, 3]), "b": 100}, snapshot_id="snap3"
)
print("Saved snap3: a=[1, 2, 3], b=100")

# Define the query function
print("\nDefining query function for flat structure...")
def query_func(pytree):
    """
    Example query function to find PyTrees where:
    - The key 'b' exists and its value is 42.
    """
    return pytree.get("b") == 42

# Use by_content to find matching snapshots
print("\nQuerying snapshots with b=42...")
matching_snapshots = manager.query.by_content(query_func)

# Display the matching snapshot IDs
print(f"Snapshots matching the query: {matching_snapshots}")

# Define a nested query function
print("\nDefining query function for nested structure...")
def nested_query_func(pytree):
    """
    Example query function to find PyTrees where:
    - The key 'nested' exists and has a subkey 'key' with value 'target'.
    """
    nested = pytree.get("nested", {})
    return nested.get("key") == "target"

# Save a snapshot with a nested structure
print("\nSaving snapshot with nested structure...")
snapshot_id4 = manager.save_snapshot(
    {"nested": {"key": "target"}, "b": 42}, snapshot_id="snap4"
)
print("Saved snap4: nested={key: target}, b=42")

# Use by_content with the nested query
print("\nQuerying snapshots with nested key=target...")
nested_matches = manager.query.by_content(nested_query_func)

print(f"Snapshots matching the nested query: {nested_matches}")

# Verify results
print("\nVerifying matches...")
for snapshot_id in nested_matches:
    snapshot = manager.get_snapshot(snapshot_id)
    print(f"\nSnapshot: {snapshot_id}")
    print("Nested key value:", snapshot.data["nested"]["key"])
