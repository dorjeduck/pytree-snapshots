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

### Advanced Querying

For advanced query examples, including custom criteria and logical queries, check out the [Query Guide](./query_guide.md) page.

### Further Examples

Explore the [`examples` folder](./examples) for a random collection of demos showcasing various features and use cases.

## Deepcopy Logic in SnapshotManager

The `SnapshotManager` uses the `deepcopy` parameter to control whether snapshots are returned as deep copies or as references to the original PyTree.

### What is Deepcopy?

A **deep copy** creates a new copy of the entire data structure. Modifications to the copy will not affect the original, ensuring data safety. This is particularly useful for immutable snapshots or when you want to avoid unintended side effects.

In contrast, a **shallow copy** or direct reference means changes to the retrieved PyTree will also modify the stored snapshot.

### Default Behavior

By default, snapshots are returned as deep copies, ensuring that modifications to the retrieved PyTree do not affect the stored snapshot.

You can override this behavior globally when initializing the SnapshotManager. 

```python
manager = SnapshotManager(deepcopy=False)
```

### Overriding Deepcopy Per Retrieval

The SnapshotManager allows you to override the initial deepcopy setting for individual snapshot retrievals. This is useful when you want a specific snapshot retrieval to behave differently from the default setting.

```python
from pytree_snapshots import SnapshotManager

# Initialize the manager
manager = SnapshotManager(deepcopy=True)  # Default behavior is deepcopy enabled

# Save a snapshot
snapshot_id = manager.save_snapshot({"a": 1, "b": [2, 3]})

# Retrieve a snapshot without deepcopy (shallow reference)
retrieved_reference = manager.get_snapshot(snapshot_id, deepcopy=False)

# Modify the retrieved snapshot
retrieved_reference["b"].append(4)

# Since deepcopy was disabled for this retrieval, the original snapshot is also modified
stored_snapshot = manager.get_snapshot(snapshot_id)
assert stored_snapshot["b"] == [2, 3, 4], "Deepcopy override failed: Original snapshot was not updated."
```

## Snapshot Compression

Snapshots can be compressed during saving to reduce memory usage. However, this compression slows down saving and retrieval, making it less suitable for scenarios requiring frequent access.

### Default Compression Setting

Compression is disabled by default. You can enable compression globally during initialization:

```python
# Enable snapshot compression globally
manager = SnapshotManager(compress=True)
```

### Overriding Compression for Individual Snapshots

You can override the global compression setting for specific snapshots by using the compress parameter when calling save_snapshot.

```python
# Save a snapshot with default compression (inherits global setting)
snapshot_id_default = manager.save_snapshot({"a": 1})

# Save a snapshot with explicit compression
snapshot_id_compressed = manager.save_snapshot({"b": 2}, compress=True)

# Save a snapshot without compression
snapshot_id_uncompressed = manager.save_snapshot({"c": 3}, compress=False)
```

Compressed snapshots are automatically decompressed when retrieved.

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
