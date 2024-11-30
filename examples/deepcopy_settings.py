"""Example demonstrating deepcopy settings in SnapshotManager.

This example shows:
1. Initializing manager with deepcopy disabled
2. Retrieving snapshots without deepcopy (shallow reference)
3. Demonstrating how modifications affect the original data
"""

from snapshot_manager import SnapshotManager

# Initialize the manager with deepcopy disabled
print("\nInitializing SnapshotManager with deepcopy_on_save=False...")
manager = SnapshotManager(deepcopy_on_save=False)  # Default behavior is deepcopy enabled

# Create and save initial data
initial_data = {"a": 1, "b": [2, 3]}
print("\nSaving initial data:", initial_data)
snapshot_id = manager.save_snapshot(initial_data)

# Retrieve snapshot without deepcopy (shallow reference)
print("\nRetrieving snapshot without deepcopy...")
retrieved_reference = manager.get_snapshot(snapshot_id, deepcopy=False)
print("Retrieved data:", retrieved_reference.data)

# Modify the retrieved data
print("\nModifying retrieved data by appending 4 to list b...")
retrieved_reference.data["b"].append(4)
print("Modified retrieved data:", retrieved_reference.data)

# Show that original snapshot was also modified
print("\nChecking original snapshot...")
stored_snapshot = manager.get_snapshot(snapshot_id)
print("Original snapshot data:", stored_snapshot.data)

# Verify the modification
assert stored_snapshot.data["b"] == [2, 3, 4], "Deepcopy override failed: Original snapshot was not updated."
print("\nVerification successful: Changes propagated to original snapshot")