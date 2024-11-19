import time
from snapshot_manager import SnapshotManager
from snapshot_manager.query.logical_queries import AndQuery
from snapshot_manager.query.base_queries import ByTimeRangeQuery, ByMetadataQuery

# Initialize the manager
manager = SnapshotManager()

# Save snapshots (timestamps are managed internally)
manager.save_snapshot({"a": 1}, snapshot_id="snap1", metadata={"status": "complete"})
time.sleep(1)  # Simulate delay between saves
manager.save_snapshot({"b": 2}, snapshot_id="snap2", metadata={"status": "in_progress"})
time.sleep(1)  # Simulate delay between saves
manager.save_snapshot({"c": 3}, snapshot_id="snap3", metadata={"status": "complete"})

# Define a time range: last 2 seconds
start_time = time.time() - 2
end_time = time.time()

# Combine time range and metadata query: "complete" snapshots saved in the last 2 seconds
query = AndQuery(
    ByMetadataQuery("status", "complete"),
    ByTimeRangeQuery(start_time, end_time)
)

results = manager.query.evaluate(query)
print("Complete snapshots from the last 2 seconds:", results)
# Expected Output: ['snap3']