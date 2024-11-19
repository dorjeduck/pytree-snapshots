# Snapshot Manager

A lightweight and versatile tool for managing snapshots of Python objects.

Initially created with `JAX PyTrees` in mind, it works seamlessly with any Python object. A specialized `PyTreeSnapshotManager` is included for PyTree-specific operations, but its features are still limited and evolving.

**Note**: This project began as a personal experiment to explore snapshot management. As it grew into a potentially useful tool, I decided to share it on GitHub to gather feedback and suggestions. Currently, it’s in beta and may undergo significant changes as it matures.

## Features

- **Capture and Manage Snapshots**: Save and retrieve snapshots of data structures.
- **Metadata and Tagging:** Associate metadata and tags with snapshots.
- **Advanced Query Support**: Perform complex searches using metadata, tags, time ranges, custom criteria, and logical operations.
- **Flexible Deepcopy Options**: Control how snapshots are stored and retrieved for optimal performance or data safety.
- **PyTreeSnapshotManager**: Includes specialized features for JAX PyTrees, such as validation, transformations and PyTree-specific queries. (more to come)

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

### Save and retrieve snapshots

Basic example demonstrating how to save and retrieve snapshots of PyTrees.

```python
from snapshot_manager import SnapshotManager

# Initialize the manager
manager = SnapshotManager()

# Save a snapshot
data = {"a": 1, "b": 2}
snapshot_id = manager.save_snapshot(data, metadata={"project": "example"})

# Retrieve the snapshot
retrieved = manager.get_snapshot(snapshot_id)
print("Retrieved snapshot:", retrieved)
# Output: Retrieved snapshot: {'a': 1, 'b': 2}
```

### Save and Transform PyTrees

For `JAX` users, here’s an example demonstrating PyTree-specific functionality.

```python
from snapshot_manager import PyTreeSnapshotManager

# Initialize the PyTree manager
manager = PyTreeSnapshotManager()

# Save a PyTree
pytree = {"a": 2, "b": [2, 3]}
snapshot_id = manager.save_snapshot(pytree, metadata={"experiment": "baseline"})

# Apply a transformation to the PyTree
def square_leaf(x):
    return x ** 2 if isinstance(x, int) else x

transformed_pytree = manager.apply_leaf_transformation(snapshot_id, square_leaf)
print("Transformed PyTree:", transformed_pytree)
# Output: {'a': 4, 'b': [4, 9]}
```

### Managing Snapshots with Tags

This example shows how to organize snapshots using tags, search for snapshots by tags, and retrieve specific snapshots.

```python
from snapshot_manager import SnapshotManager

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

- For advanced query examples, including custom criteria and logical queries, check out the [Query Guide](./query_guide.md) page.
- Explore the [examples](./examples) folder for a random collection of demos showcasing various features and use cases.

## Deepcopy Logic

`SnapshotManager` provides flexibility in how snapshots are saved and retrieved, allowing you to choose between using *deepcopy* or *reference*. By default, snapshots are saved and retrieved using deepcopies to ensure they are isolated from the original data. However, you can override this behavior globally or on a per-operation basis to optimize for performance gains or specific use cases.

Set the behavior during initialization to apply consistently across all operations:

```python
manager = SnapshotManager(deepcopy_on_save=False, deepcopy_on_retrieve=False)
```

Override the behavior for specific save or retrieve calls:

```python
snapshot = manager.get_snapshot(snapshot_id, deepcopy=False)
```

## Roadmap

- Expanding PyTree-specific functionality to enhance JAX integration.
- Listening to feedback ...

## Contribution

We warmly welcome contributions and look forward to your pull requests!

## Changelog

- 2024.11.26
  - Repo renamed to SnapshotManager
- 2024.11.25
  - Reinforcement Learning use case added 
- 2024.11.24
  - Deepcopy on Save 
- 2024.11.23
  - Refactored
  - Logic Queries
  - PyTree specialization
- 2024.11.20
  - PyTree comparison removed - for now.
- 2024.11.18
  - Initial repository setup and first commit.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
