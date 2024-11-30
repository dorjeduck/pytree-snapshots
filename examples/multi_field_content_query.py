"""Example demonstrating multi-field content queries in SnapshotManager.

This example shows:
1. Saving snapshots with multiple fields
2. Querying snapshots based on multiple field conditions
3. Using ByContentQuery with custom comparison functions
"""

from snapshot_manager import SnapshotManager
from snapshot_manager.query.base_queries import ByContentQuery

# Initialize the manager
manager = SnapshotManager()

# Save snapshots with structured content
print("\nSaving test snapshots...")
snapshots = [
    {"a": 1, "b": 2},  # snap1: a=1, b=2
    {"a": 3, "b": 4},  # snap2: a=3, b=4
    {"a": 1, "b": 4},  # snap3: a=1, b=4
]

for i, data in enumerate(snapshots, 1):
    snapshot_id = f"snap{i}"
    manager.save_snapshot(data, snapshot_id=snapshot_id)
    print(f"Saved {snapshot_id}:", data)

# Define query conditions
print("\nQuerying snapshots where a=1 AND b=4...")
def match_conditions(pytree):
    """Return True if pytree matches a=1 and b=4"""
    return pytree.get("a") == 1 and pytree.get("b") == 4

query = ByContentQuery(match_conditions)
results = manager.query.evaluate(query)

# Show results
print("\nResults:")
print("Matching snapshots:", results)

# Verify the matching snapshot's content
if results:
    for snapshot_id in results:
        snapshot = manager.get_snapshot(snapshot_id)
        print(f"\nVerifying {snapshot_id} content:")
        print("a =", snapshot.data["a"])
        print("b =", snapshot.data["b"])