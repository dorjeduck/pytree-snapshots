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
