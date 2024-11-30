"""Example demonstrating querying by leaf value in PyTreeSnapshotManager.

This example shows:
1. Saving snapshots with PyTree data
2. Using a query function to find snapshots with leaf values meeting a condition
3. Demonstrating queries with nested structures
"""

from snapshot_manager import PyTreeSnapshotManager

# Initialize the PyTree manager
manager = PyTreeSnapshotManager()

# Save snapshots with PyTree data
print("\nSaving test snapshots...")
manager.save_snapshot(
    {"a": 1, "b": [2, 3]},
    snapshot_id="snap1",
    metadata={"project": "example1"},
)
print("Saved snap1: a=1, b=[2, 3]")

manager.save_snapshot(
    {"x": 5, "y": {"z": 10}},
    snapshot_id="snap2",
    metadata={"project": "example2"},
)
print("Saved snap2: x=5, y={z: 10}")

manager.save_snapshot(
    {"c": [0, -1], "d": 7},
    snapshot_id="snap3",
    metadata={"project": "example1"},
)
print("Saved snap3: c=[0, -1], d=7")

# Define query for leaf values greater than 5
print("\nDefining query for leaf values > 5...")
def leaf_value_query(x):
    """Return True if leaf value is greater than 5"""
    return x > 5

# Query snapshots
print("\nQuerying snapshots with leaf value > 5...")
query = manager.query.by_leaf_value(leaf_value_query)
results = manager.query.evaluate(query)

# Display results
print("\nResults:")
print("Snapshots with a leaf value > 5:", results)

# Verify results
print("\nVerifying matches...")
for snapshot_id in results:
    snapshot = manager.get_snapshot(snapshot_id)
    print(f"\nSnapshot: {snapshot_id}")
    print("Data:", snapshot.data)
