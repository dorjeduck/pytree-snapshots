"""Example demonstrating snapshot overwrite tracking in SnapshotManager.

This example shows:
1. Saving snapshots with the same ID
2. Tracking overwrites and changes
3. Retrieving the latest version of a snapshot
"""

from snapshot_manager import SnapshotManager

# Initialize the manager
print("\nInitializing SnapshotManager...")
manager = SnapshotManager()

# Save an initial snapshot
print("\nSaving initial snapshot...")
initial_data = {"a": 1, "b": 2}
snapshot_id = manager.save_snapshot(initial_data, snapshot_id="snap1")
print(f"Initial snapshot 'snap1' saved with data: {initial_data}")

# Try to save without overwrite (should raise an error)
print("\nTrying to save new version without overwrite...")
try:
    new_data_1 = {"a": 1, "b": 3}
    manager.save_snapshot(new_data_1, snapshot_id="snap1", overwrite=False)
except ValueError as e:
    print(f"Error (expected): {e}")

# Save a new version with overwrite enabled
print("\nSaving new version with overwrite enabled...")
new_data_2 = {"a": 1, "b": 3}
manager.save_snapshot(new_data_2, snapshot_id="snap1", overwrite=True)
print(f"New version of 'snap1' saved with data: {new_data_2}")

# Retrieve and inspect the latest version
print("\nRetrieving latest version...")
latest_snapshot = manager.get_snapshot("snap1")
print("Latest snapshot data:", latest_snapshot.data)

# Compare with original data
print("\nComparing versions:")
print("Original data:", initial_data)
print("Current data:", latest_snapshot.data)
