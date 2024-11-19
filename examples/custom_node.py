from pytree_snapshots import PytreeSnapshotManager
from jax.tree_util import register_pytree_node

# Define a custom PyTree node
class CustomNode:
    def __init__(self, value, metadata=None):
        self.value = value
        self.metadata = metadata or {}

    def __repr__(self):
        return f"CustomNode(value={self.value}, metadata={self.metadata})"

# Define flattening and unflattening functions
def flatten_custom_node(node):
    # Leaves to save, plus auxiliary data
    return [node.value], node.metadata

def unflatten_custom_node(aux_data, children):
    # Reconstruct node from leaves and auxiliary data
    return CustomNode(children[0], aux_data)

# Register the custom node
register_pytree_node(CustomNode, flatten_custom_node, unflatten_custom_node)

# Initialize the manager
manager = PytreeSnapshotManager()

# Create PyTrees with custom nodes
pytree1 = {"a": CustomNode(42, {"type": "custom"}), "b": 2}
pytree2 = {"a": CustomNode(43, {"type": "custom"}), "b": 3}

# Save snapshots
manager.save_snapshot(pytree1, snapshot_id="custom_snap1")
manager.save_snapshot(pytree2, snapshot_id="custom_snap2")

# Compare snapshots
differences = manager.compare_snapshots("custom_snap1", "custom_snap2")
print("Differences:", differences)
# Output: Differences: {'a': (CustomNode(value=42, metadata={'type': 'custom'}), CustomNode(value=43, metadata={'type': 'custom'})), 'b': (2, 3)}