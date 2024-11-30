"""Example demonstrating saving and retrieving manager state in SnapshotManager.

This example shows:
1. Saving the state of a SnapshotManager to a file
2. Loading the saved state into a new manager instance
3. Verifying that snapshots are preserved across sessions
"""

import os
from snapshot_manager import SnapshotManager


# Create a manager and save a snapshot
print("\nCreating SnapshotManager and saving a snapshot...")
manager = SnapshotManager()
pytree = {"a": 1, "b": 2}
manager.save_snapshot(pytree, snapshot_id="example_snapshot")
print("Saved snapshot with data:", pytree)

# Save the state
state_file = "snapshot_manager_state.pkl"
print(f"\nSaving manager state to '{state_file}'...")
manager.save_to_file(state_file)
print(f"Manager state saved to '{state_file}'.")

# Load the state into a new manager
print("\nLoading manager state into a new instance...")
new_manager = SnapshotManager.load_from_file(state_file)
print("Manager state loaded into a new manager instance.")

# Check if snapshots were loaded correctly
loaded_snapshots = list(new_manager.list_snapshots())
print(f"\nLoaded snapshots: {loaded_snapshots}")

if loaded_snapshots:
    retrieved_pytree = new_manager.get_snapshot(loaded_snapshots[0])
    print("Retrieved pytree data:", retrieved_pytree.data)
else:
    print(
        "No snapshots were loaded. Ensure the save and load processes are functioning correctly."
    )

# Clean up the state file
print(f"\nCleaning up: deleting '{state_file}'...")
os.remove(state_file)
print(f"Cleaned up: '{state_file}' deleted.")
