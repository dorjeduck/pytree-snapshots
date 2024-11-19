# PyTree Snapshots

A lightweight and flexible manager for capturing, managing, and comparing PyTree snapshots in JAX. 

⚠️ Currently in Beta: While the core functionality seems solid, PyTreeSnapshots is still evolving. Expect ongoing improvements, additional features, and extended documentation. Your feedback and contributions are very welcome!

## Features

- Save and retrieve snapshots of PyTrees.
- Compare differences between snapshots with unified structures.
- Metadata and tagging support for organization.
- Support for custom PyTree nodes.

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

### Example 1: Basic Snapshot Comparison

This example demonstrates how to save two snapshots of PyTrees and compare their differences. It’s a simple way to get started with the core functionality of `PyTreeSnapshots`.

```python
from pytree_snapshots import PytreeSnapshotManager

manager = PytreeSnapshotManager()

pytree1 = {"a": 1, "b": 2}
pytree2 = {"a": 1, "b": 3}

manager.save_snapshot(pytree1, snapshot_id="snap1")
manager.save_snapshot(pytree2, snapshot_id="snap2")

differences = manager.compare_snapshots("snap1", "snap2")
print(differences)  # {'a': None, 'b': (2, 3)}
```

### Example 2: Managing Snapshots with Tags

This example shows how to organize snapshots using tags, search for snapshots by tags, and retrieve specific snapshots for inspection.

```python
from pytree_snapshots import PytreeSnapshotManager

# Initialize the manager
manager = PytreeSnapshotManager()

# Create PyTrees
pytree1 = {"a": 1, "b": 2}
pytree2 = {"a": 3, "b": 4}
pytree3 = {"x": 10, "y": 20}

# Save snapshots with tags
manager.save_snapshot(pytree1, snapshot_id="snap1", tags=["experiment", "baseline"])
manager.save_snapshot(pytree2, snapshot_id="snap2", tags=["experiment", "variant"])
manager.save_snapshot(pytree3, snapshot_id="snap3", tags=["control"])

# Find snapshots by tag
experiment_snapshots = manager.find_snapshots_by_tag("experiment")
print("Experiment Snapshots:", experiment_snapshots)
# Output: Experiment Snapshots: ['snap1', 'snap2']

# Retrieve and inspect a snapshot
snapshot = manager.get_snapshot("snap1")
print("Snapshot snap1:", snapshot)
# Output: Snapshot snap1: {'a': 1, 'b': 2}
```

Example 3: Find Snapshots by User-Defined Criteria

This example demonstrates how to search for snapshots based on custom criteria using a user-defined comparator function. You can use this functionality to identify snapshots that meet specific conditions, such as the highest accuracy, the most tags, or the earliest creation time, as demonstrated in the example below.

```python
from pytree_snapshots import PytreeSnapshotManager

# Initialize the manager
manager = PytreeSnapshotManager()

manager.save_snapshot({}, snapshot_id="snap1", metadata={"accuracy": 0.85,"created_at": 1690000000.0},tags=["experiment", "draft"])
manager.save_snapshot({}, snapshot_id="snap2", metadata={"accuracy": 0.90,"created_at": 1695000000.0},tags=["draft"])
manager.save_snapshot({}, snapshot_id="snap3", metadata={"accuracy": 0.88,"created_at": 1790000000.0},tags=["final", "experiment", "published"])

best_snapshot_id = manager.find_snapshot_by_criteria(
    lambda s1, s2: s1.metadata["accuracy"] >= s2.metadata["accuracy"]
)

print(f"Snapshot with highest accuracy: {best_snapshot_id}")

# Use a comparator to find the snapshot with the most tags
snapshot_with_most_tags = manager.find_snapshot_by_criteria(
    lambda s1, s2: len(s1.tags) >= len(s2.tags)
)

print(f"Snapshot with most tags: {snapshot_with_most_tags}")

# Use a comparator to find the oldest snapshot
oldest_snapshot_id = manager.find_snapshot_by_criteria(
    lambda s1, s2: s1.metadata["created_at"] <= s2.metadata["created_at"]
)

print(f"Oldest snapshot: {oldest_snapshot_id}")

# Output
# Snapshot with highest accuracy: snap2
# Snapshot with most tags: snap3
# Oldest snapshot: snap1
```

Explore the [`examples` folder](./examples) for additional demos showcasing various features and use cases.

## Roadmap

- Add support for handling nested custom PyTree nodes.
- Improve metadata and tag querying capabilities.
- ...

## Changelog

- 2024.11.18
  - Initial repository setup and first commit.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
