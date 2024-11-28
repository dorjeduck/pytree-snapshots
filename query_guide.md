# Query Guide

`SnapshotManager` provides a flexible and powerful query system to retrieve snapshots based on various criteria such as metadata, tags, content, and more. This guide demonstrates how to use the query system effectively, from simple queries to complex, nested logical operations.

## Querying Snapshots by Time

Every snapshot saved using `SnapshotManager` is automatically assigned a timestamp. This timestamp is managed internally and represents the time when the snapshot was saved. You can use the `ByTimeRangeQuery` to filter snapshots based on these timestamps.

```python
import time
from snapshot_manager import SnapshotManager
from snapshot_manager.query import ByTimeRangeQuery

# Initialize the manager
manager = SnapshotManager()

# Save snapshots (timestamps are assigned automatically)
manager.save_snapshot({"a": 1}, snapshot_id="snap1")  # Snapshot saved at time T1
time.sleep(1)  # Simulate delay
manager.save_snapshot({"b": 2}, snapshot_id="snap2")  # Snapshot saved at time T2
time.sleep(1)  # Simulate delay
manager.save_snapshot({"c": 3}, snapshot_id="snap3")  # Snapshot saved at time T3

# Define a time range (e.g., last 2 seconds)
start_time = time.time() - 2  # 2 seconds ago
end_time = time.time()  # Current time

# Query snapshots saved in the time range
query = ByTimeRangeQuery(start_time, end_time)
results = manager.query.evaluate(query)

print("Snapshots saved in the last 2 seconds:", results)
# Output: ['snap3']
```


## Custom Criteria for Selecting Snapshots

This example shows how to identify a single snapshot that meets specific user-defined criteria using the `get_snapshot_by_cmp` method. You can use this feature to search for snapshots based on metadata, tags, or other properties, such as finding the snapshot with the highest accuracy, the most associated tags, or the earliest creation time.

```python
from snapshot_manager import SnapshotManager

# Initialize the manager
manager = SnapshotManager()

# Save snapshots with metadata and tags
manager.save_snapshot({}, snapshot_id="snap1", metadata={"accuracy": 0.85, "created_at": 1690000000.0}, tags=["experiment", "draft"])
manager.save_snapshot({}, snapshot_id="snap2", metadata={"accuracy": 0.90, "created_at": 1695000000.0}, tags=["draft"])
manager.save_snapshot({}, snapshot_id="snap3", metadata={"accuracy": 0.88, "created_at": 1790000000.0}, tags=["final", "experiment", "published"])

# Find snapshot with the highest accuracy
snapshot_with_highest_accuracy = manager.query.by_cmp(
    lambda s1, s2: s1.metadata["accuracy"] >= s2.metadata["accuracy"]
)
print(f"Snapshot with highest accuracy: {snapshot_with_highest_accuracy}")
# Output: Snapshot with highest accuracy: snap2

# Find snapshot with the most tags
snapshot_with_most_tags = manager.query.by_cmp(
    lambda s1, s2: len(s1.tags) >= len(s2.tags)
)
print(f"Snapshot with most tags: {snapshot_with_most_tags}")
# Output: Snapshot with most tags: snap3

# Find the oldest snapshot
oldest_snapshot_id = manager.query.by_cmp(
    lambda s1, s2: s1.metadata["created_at"] <= s2.metadata["created_at"]
)
print(f"Oldest snapshot: {oldest_snapshot_id}")
# Output: Oldest snapshot: snap1
```

## Custom SnapshotQuery

In this example, we’ll create a custom query class that logs certain query operation for debugging purposes.

```python
from snapshot_manager import SnapshotManager
from snapshot_manager.query import SnapshotQuery

class LoggingSnapshotQuery(SnapshotQuery):
    """
    A custom SnapshotQuery that logs all query operations.
    """

    def __init__(self, snapshots):
        self.snapshots = snapshots

    def by_metadata(self, key, value=None):
        print(f"Querying by metadata: {key} = {value}")
        return [
            snapshot_id
            for snapshot_id, snapshot in self.snapshots.items()
            if key in snapshot.metadata
            and (value is None or snapshot.metadata[key] == value)
        ]

    def by_tags(self, tag):
        print(f"Querying by tag: {tag}")
        return [
            snapshot_id
            for snapshot_id, snapshot in self.snapshots.items()
            if tag in snapshot.tags
        ]


# Inject the custom query class into SnapshotManager
manager = SnapshotManager(query_class=LoggingSnapshotQuery)

# Save some snapshots
manager.save_snapshot(
    {"a": 1, "b": 2},
    metadata={"project": "example1"},
    tags=["experiment"],
    snapshot_id="snap1",
)
manager.save_snapshot(
    {"x": 10, "y": 20},
    metadata={"project": "example2"},
    tags=["control"],
    snapshot_id="snap2",
)

# Perform queries

print("Metadata query results:", manager.query.by_metadata("project", "example1"))
# Output:
# Querying by metadata: project = example1
# Metadata query results: ['snap1']

print("Tag query results:", manager.query.by_tags("control"))
# Output:
# Querying by tag: control
# Tag query results: ['snap2']

```

## Nested Logical Queries

This example demonstrates how to combine logical operations (AND, OR, NOT) to create complex, nested queries. You can use these queries to filter snapshots based on metadata, tags, content, or other custom criteria.

```python
from snapshot_manager import SnapshotManager
from snapshot_manager.query import AndQuery, OrQuery, NotQuery ByMetadataQuery, ByTagQuery

# Initialize the manager
manager = SnapshotManager()

# Save snapshots with metadata and tags
manager.save_snapshot(
    {"a": 1}, snapshot_id="snap1", metadata={"project": "example1"}, tags=["experiment", "baseline"]
)
manager.save_snapshot(
    {"b": 2}, snapshot_id="snap2", metadata={"project": "example2"}, tags=["control"]
)
manager.save_snapshot(
    {"c": 3}, snapshot_id="snap3", metadata={"project": "example1"}, tags=["experiment", "published"]
)

# Logical Query: Find snapshots that are in project "example1" AND tagged with "experiment",
# OR snapshots that are NOT tagged with "control".
query = OrQuery(
    AndQuery(
        ByMetadataQuery("project", "example1"),
        ByTagQuery("experiment")
    ),
    NotQuery(ByTagQuery("control"))
)

# Evaluate the query
results = manager.query.evaluate(query)
print("Snapshots matching the logical query:", results)
# Output:
# Snapshots matching the logical query: ['snap1', 'snap3']
```

## Querying Snapshots by PyTree Leaf Values

This example shows how to query snapshots where any leaf in the PyTree satisfies a specific condition, such as being greater than a certain value.

```python
from snapshot_manager import PyTreeSnapshotManager

# Initialize the PyTree manager
manager = PyTreeSnapshotManager()

# Save snapshots with PyTree data
manager.save_snapshot(
    {"a": 1, "b": [2, 3]},
    snapshot_id="snap1",
    metadata={"project": "example1"},
)
manager.save_snapshot(
    {"x": 5, "y": {"z": 10}},
    snapshot_id="snap2",
    metadata={"project": "example2"},
)
manager.save_snapshot(
    {"c": [0, -1], "d": 7},
    snapshot_id="snap3",
    metadata={"project": "example1"},
)

# Query snapshots with any leaf value greater than 5
query = manager.query.by_leaf_value(lambda x: x > 5)
results = manager.query.evaluate(query)

print("Snapshots with a leaf value > 5:", results)
# Output: Snapshots with a leaf value > 5: ['snap2', 'snap3']
```
