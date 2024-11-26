# Snapshot Manager

A lightweight and versatile tool for managing Python objects, allowing users to create isolated snapshots or manage live references, both enriched with metadata and tags.

Initially created with `JAX PyTrees` in mind, it works seamlessly with a wide range of Python object. A specialized `PyTreeSnapshotManager` is included for PyTree-specific operations, but its features are still limited and evolving.

**Note**: This project began as a personal experiment to explore snapshot management. As it grew into a potentially useful tool, I decided to share it on GitHub to gather feedback and suggestions. Currently, it’s in beta and may undergo significant changes as it matures.

## Features

- **Capture and Manage Snapshots**: Save and retrieve snapshots of data structures.
- **Metadata and Tagging:** Associate metadata and tags with snapshots.
- **Advanced Query Support**: Perform complex searches using metadata, tags, time ranges, custom criteria, and logical operations.
- **Flexible Deepcopy Options**: Control how snapshots are stored and retrieved for optimal performance or data safety.
- **Persistence**: Save all snapshots to disk and reload them later for continued training, evaluation, or analysis across sessions.
- **PyTreeSnapshotManager**: Includes specialized features for JAX PyTrees, such as validation, transformations and PyTree-specific queries. (more to come)

## Installation

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/dorjeduck/snapshot-manager.git
cd snapshot_manager
```

### Install the Package

Install the package using setup.py:

```bash
pip install .
```

## Quick Start

### Save and retrieve snapshots

Basic example demonstrating how to save and retrieve snapshots of a dictionary.

```python
from snapshot_manager import SnapshotManager

# Initialize the manager
manager = SnapshotManager()

# Save a snapshot with metadata
dic1 = {"a": 1, "b": 2}
snapshot_id = manager.save_snapshot(dic1, metadata={"project": "example"})

# Retrieve the snapshot
retrieved_snapshot = manager.get_snapshot(snapshot_id)

print("Data of retrieved snapshot:", retrieved_snapshot.data)
# Output: Data of retrieved snapshot: {'a': 1, 'b': 2}

print("Metadata:", retrieved_snapshot.metadata)
# Output: Metadata: {'project': 'example'}
```

### Save and Transform PyTrees

For `JAX` users, here’s an example demonstrating PyTree-specific functionality.

```python
from snapshot_manager import PyTreeSnapshotManager

manager = PyTreeSnapshotManager()

# Save snapshots
sid = manager.save_snapshot(
    {
        "txt": "hello pytorch",
        "x": 42,
    }
)

nsid = manager.tree_map(
    lambda x: x.replace("pytorch", "jax") if isinstance(x, str) else x,
    snapshot_id=sid,
)

print(manager.get_snapshot(nsid).data)
# Output: {'txt': 'hello jax', 'x': 42}
```

### Persistence

You can save all snapshots to a file and reload them later for further use:

```python
from snapshot_manager import SnapshotManager

# Initialize the manager
manager = SnapshotManager()

# Save some snapshots
snapshot_id1 = manager.save_snapshot({"a": 1, "b": 2}, metadata={"project": "example1"})
snapshot_id2 = manager.save_snapshot({"x": 10, "y": 20}, metadata={"project": "example2"})

# Save all snapshots to a file
manager.save_to_file("snapshots.pkl")
print("Snapshots saved to file.")

# Reload the snapshots from the file
new_manager = SnapshotManager.load_from_file("snapshots.pkl")
print("Snapshots reloaded from file.")

# Verify the reloaded snapshots
retrieved_snapshot = new_manager.get_snapshot(snapshot_id1)
print("Reloaded Data:", retrieved_snapshot.data)
# Output: Reloaded Data: {'a': 1, 'b': 2}
```

### Query by Tags

This example shows how to organize and query snapshots using tags.

```python
from snapshot_manager import SnapshotManager

# Initialize the manager
manager = SnapshotManager()

# Save snapshots with tags
manager.save_snapshot({"a": 1, "b": 2}, snapshot_id="snap1", tags=["experiment", "baseline"])
manager.save_snapshot({"a": 3, "b": 4}, snapshot_id="snap2", tags=["experiment", "variant"])
manager.save_snapshot({"x": 10, "y": 20}, snapshot_id="snap3", tags=["control"])

# Find snapshots by tag
experiment_snapshots = manager.query.by_tags("experiment")

print("Experiment Snapshots:", experiment_snapshots)
# Output: Experiment Snapshots: ['snap1', 'snap2']
```

For advanced query examples, including custom criteria and logical queries, check out the [Query Guide](./query_guide.md) page.

### Additional Examples

Explore the [examples](./examples) folder for a random collection of demos showcasing various features and use cases.

## Deepcopy Logic

By default, `SnapshotManager` saves and retrieves snapshots as deepcopies, ensuring the stored data is completely isolated from the original. This guarantees that modifications to the original data or the snapshot do not affect each other.

However, SnapshotManager also allows for alternative behavior to suit different workflows. You can configure it to save and retrieve data as live references, optimizing for performance in cases where data isolation is unnecessary. This flexibility can be applied globally or customized for specific operations.

You can set the behavior globally during initialization to ensure consistency across all operations.

```python
manager = SnapshotManager(deepcopy_on_save=False, deepcopy_on_retrieve=False)
```

Override the behavior for specific save or retrieve calls. For instance:

```python
snapshot = manager.get_snapshot(snapshot_id, deepcopy=False)
```

### Accessing Snapshots with `manager[snapshot_id]`

Indexed access to snapshots behaves identically to get_snapshot in terms of deepcopy behavior. It returns either a deepcopy or a reference based on the global `deepcopy_on_retrieve` setting:

```python
# Equivalent to get_snapshot("snap1")
snapshot1 = manager["snap1"]  
```

You can override the global `deepcopy_on_retrieve` setting for a specific retrieval. For instance:

```python
# Equivalent to get_snapshot("snap1",deepcopy=False) 
snapshot1 = manager["snap1",deepcopy=False]
```

## Use Cases

### Reinforcement Learning

Learn how `SnapshotManager` can be integrated into a reinforcement learning pipeline to manage, rank, and experiment with saved network states. This use case closely reflects the original motivation for developing `SnapshotManager`. See the [Reinforcement Learning Use Case](./use_case_rl.md) for details.

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
