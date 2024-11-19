# PyTree Snapshots

A lightweight and flexible manager for capturing, managing, and comparing PyTree snapshots in JAX. Designed with simplicity and versatility in mind, this library is perfect for exploring and managing PyTree structures.

⚠️ Currently in Beta: While the core functionality is solid, PyTreeSnapshots is still evolving. Expect ongoing improvements, additional features, and extended documentation. Your feedback and contributions are welcome!

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

### Run the Examples

```bash
cd examples
python quickstart_demo.py
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

Check out the examples folder for additional demos. 

## Roadmap

- Add support for handling nested custom PyTree nodes.
- Improve metadata and tag querying capabilities.
- ...

## Changelog

- 2024.11.18
  - Initial repository setup and first commit.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
