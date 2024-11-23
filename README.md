# PyTree Snapshots

A lightweight and flexible manager for capturing and managing `PyTree` snapshots in JAX, currently in Beta.

## Features

- Save and retrieve snapshots of PyTrees.
- Metadata and tagging support.
- Snapshot query support.

## Installation

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/your-username/pytree-snapshots.git
cd pytree-snapshots
```

### Install the Package

Install the package using setup.py:

```bash
pip install .
```

## Quick Start

### Example 1: Save and retrieve snapshots

Basic example demonstrating how to save and retrieve snapshots of PyTrees.

```python
from pytree_snapshots import SnapshotManager

# Initialize the manager
manager = SnapshotManager()

# Save a snapshot
pytree = {"a": 1, "b": 2}
snapshot_id = manager.save_snapshot(pytree, metadata={"project": "example"})

# Retrieve the snapshot
retrieved = manager.get_snapshot(snapshot_id)
print("Retrieved snapshot:", retrieved)
# Output: Retrieved snapshot: {'a': 1, 'b': 2}
```

### Example 2: Managing Snapshots with Tags

This example shows how to organize snapshots using tags, search for snapshots by tags, and retrieve specific snapshots for inspection.

```python
from pytree_snapshots import SnapshotManager

# Initialize the manager
manager = SnapshotManager()

# Create PyTrees
pytree1 = {"a": 1, "b": 2}
pytree2 = {"a": 3, "b": 4}
pytree3 = {"x": 10, "y": 20}

# Save snapshots with tags
manager.save_snapshot(pytree1, snapshot_id="snap1", tags=["experiment", "baseline"])
manager.save_snapshot(pytree2, snapshot_id="snap2", tags=["experiment", "variant"])
manager.save_snapshot(pytree3, snapshot_id="snap3", tags=["control"])

# Find snapshots by tag
experiment_snapshots = manager.query.by_tag("experiment")
print("Experiment Snapshots:", experiment_snapshots)
# Output: Experiment Snapshots: ['snap1', 'snap2']

# Retrieve and inspect a snapshot
snapshot = manager.get_snapshot("snap1")
print("Snapshot snap1:", snapshot)
# Output: Snapshot snap1: {'a': 1, 'b': 2}
```

### Example 3: Custom Criteria for Selecting Snapshots

This example shows how to identify a single snapshot that meets specific user-defined criteria using the `get_snapshot_by_comparator` method. You can use this feature to search for snapshots based on metadata, tags, or other properties, such as finding the snapshot with the highest accuracy, the most associated tags, or the earliest creation time.

```python
from pytree_snapshots import SnapshotManager

# Initialize the manager
manager = SnapshotManager()

# Save snapshots with metadata and tags
manager.save_snapshot({}, snapshot_id="snap1", metadata={"accuracy": 0.85, "created_at": 1690000000.0}, tags=["experiment", "draft"])
manager.save_snapshot({}, snapshot_id="snap2", metadata={"accuracy": 0.90, "created_at": 1695000000.0}, tags=["draft"])
manager.save_snapshot({}, snapshot_id="snap3", metadata={"accuracy": 0.88, "created_at": 1790000000.0}, tags=["final", "experiment", "published"])

# Find snapshot with the highest accuracy
snapshot_with_highest_accuracy = manager.query.by_comparator(
    lambda s1, s2: s1.metadata["accuracy"] >= s2.metadata["accuracy"]
)
print(f"Snapshot with highest accuracy: {snapshot_with_highest_accuracy}")
# Output: Snapshot with highest accuracy: snap2

# Find snapshot with the most tags
snapshot_with_most_tags = manager.query.by_comparator(
    lambda s1, s2: len(s1.tags) >= len(s2.tags)
)
print(f"Snapshot with most tags: {snapshot_with_most_tags}")
# Output: Snapshot with most tags: snap3

# Find the oldest snapshot
oldest_snapshot_id = manager.query.by_comparator(
    lambda s1, s2: s1.metadata["created_at"] <= s2.metadata["created_at"]
)
print(f"Oldest snapshot: {oldest_snapshot_id}")
# Output: Oldest snapshot: snap1
```

### Example 4: Custom SnapshotQuery

In this example, we’ll create a custom query class that logs certain query operation for debugging purposes.

```python
from pytree_snapshots import SnapshotManager, SnapshotQuery

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

    def by_tag(self, tag):
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

print("Tag query results:", manager.query.by_tag("control"))
# Output:
# Querying by tag: control
# Tag query results: ['snap2']

```

### Example 5: Nested Logical Queries

This example demonstrates how to combine logical operations (AND, OR, NOT) to create complex, nested queries. You can use these queries to filter snapshots based on metadata, tags, content, or other custom criteria.

```python
from pytree_snapshots import SnapshotManager
from pytree_snapshots.query.logical_queries import AndQuery, OrQuery, NotQuery
from pytree_snapshots.query.query_base import ByMetadataQuery, ByTagQuery

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

Explore the [`examples` folder](./examples) for additional demos showcasing various features and use cases.

## Roadmap

- Listening to feedback ...

## Contribution

We warmly welcome contributions and look forward to your pull requests!

## Changelog

- 2024.11.23
  - Refactored
  - Logic Queries
- 2024.11.20
  - PyTree comparison removed - for now.
- 2024.11.18
  - Initial repository setup and first commit.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
