# Save multiple snapshots
import os
from pytree_snapshots import PytreeSnapshotManager

# Create a manager and save a snapshot
manager = PytreeSnapshotManager()

manager.save_snapshot({"a": 1}, snapshot_id="snap1")
manager.save_snapshot({"b": 2}, snapshot_id="snap2")
manager.save_snapshot({"c": 3}, snapshot_id="snap3")

# List snapshots sorted by age (newest to oldest)
snapshots_newest_to_oldest = manager.list_snapshots_by_age(ascending=False)
print(f"Snapshots from newest to oldest: {snapshots_newest_to_oldest}")
