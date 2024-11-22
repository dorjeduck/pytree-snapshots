# PyTree Snapshots

A lightweight and flexible manager for capturing and managing data snapshots in Python. While built with **JAX PyTrees** in mind, offering specialized features like leaf transformations and structure-based queries, it also works seamlessly with any Python object, making it a versatile solution for snapshotting and restoring complex data structures.

**Note:** PyTree Snapshots is currently in **beta**. While it is stable for most use cases, some features may undergo changes, and you may encounter bugs as we gather feedback and refine the library. Please report any issues to help us improve!

## Features

- **Capture and Manage Snapshots**: Save and retrieve snapshots of data structures with ease.
- **General-Purpose Compatibility**: Works seamlessly with any Python object.
- **Metadata and Tagging**: Organize snapshots with metadata and tags for better discoverability.
- **Advanced Query Support**: Perform complex searches using metadata, tags, time ranges, or custom criteria.
- **Deepcopy**: Fine-tune storage and retrieval behavior.
- **Designed for PyTrees**: Includes specialized features for JAX PyTrees, such as validation, transformations and PyTree-specific queries. (more to come)

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

### Save and Transform PyTrees

For JAX users, here’s an example demonstrating PyTree-specific functionality.

```python
from pytree_snapshots import PyTreeSnapshotManager

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

Explore the [examples](./examples) folder for a random collection of demos showcasing various features and use cases.

## Deepcopy Logic in SnapshotManager

The `SnapshotManager` offers flexible control over `deepcopy` behavior with separate options for saving and retrieving snapshots. This ensures both data safety and performance optimization based on your use case.

### What is Deepcopy?

A **deep copy** creates a new copy of the entire data structure. Modifications to the copy will not affect the original, ensuring data safety. This is particularly useful for immutable snapshots or when you want to avoid unintended side effects. However, deep copies come at the cost of performance, as creating them can be computationally expensive for large PyTrees.

In contrast, a **shallow copy** or direct reference means changes to the retrieved PyTree will also modify the stored snapshot.

### Default Behavior

By default:

- Saving Snapshots: Deepcopy is enabled to ensure that the saved snapshot remains isolated from modifications to the original PyTree.

- Retrieving Snapshots: Deepcopy is also enabled to ensure the retrieved PyTree is independent of the stored snapshot.

This behavior provides maximum data safety but may introduce performance overhead.

You can customize the default deepcopy behavior during initialization of the SnapshotManager:

```python
from pytree_snapshots import SnapshotManager

# Disable deepcopy globally for saving and retrieving
manager = SnapshotManager(deepcopy_on_save=False, deepcopy_on_retrieve=False)
```

### Overriding Deepcopy Setting Per Operation

#### Save Operation

You can override the global deepcopy_on_save setting for individual save operations by using the deepcopy parameter:

```python
snapshot_id = manager.save_snapshot(
    {"a": 1, "b": [2, 3]},
    deepcopy=False  # Overrides the global setting
)
```

#### Retrieve Operation

Similarly, the global deepcopy_on_retrieve setting can be overridden during individual retrievals:

```python
from pytree_snapshots import SnapshotManager

# Initialize the manager
manager = SnapshotManager(deepcopy_on_retrieve=True)  # Default behavior is deepcopy enabled

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

The deepcopy logic used in get_snapshot also applies to other retrieval methods:

```python
# Retrieve the latest snapshot without deep copying
latest_snapshot = manager.get_latest_snapshot(deepcopy=False)

# Retrieve the oldest snapshot with deep copying
oldest_snapshot = manager.get_oldest_snapshot(deepcopy=True)

# Retrieve a snapshot by index
snapshot_by_index = manager[0, deepcopy=False]  # Overrides the default
```

## Roadmap

- Expanding PyTree-specific functionality to enhance JAX integration.
- Listening to feedback ...

## Contribution

We warmly welcome contributions and look forward to your pull requests!

## Changelog

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
