import pytest
from pytree_snapshots.pytree_snapshot_manager import PytreeSnapshotManager
from jax.tree_util import register_pytree_node


@pytest.fixture
def manager():
    return PytreeSnapshotManager()


def test_simple_case(manager):
    # Simple PyTree comparison
    pytree1 = {"a": 1, "b": 2}
    pytree2 = {"a": 1, "b": 3, "x": 4}

    manager.save_snapshot(pytree1, snapshot_id="id1")
    manager.save_snapshot(pytree2, snapshot_id="id2")

    differences = manager.compare_snapshots("id1", "id2")

    # Expected differences as a PyTree
    expected_differences = {"a": PytreeSnapshotManager.NO_DIFFERENCE, "b": (2, 3), "x": (PytreeSnapshotManager.LEAF_MISSING, 4)}

    # Assert that the PyTree structures match
    assert differences == expected_differences


def test_identical_pytrees(manager):
    # Identical PyTrees
    pytree1 = {"a": 1, "b": {"x": 2}}
    pytree2 = {"a": 1, "b": {"x": 2}}

    manager.save_snapshot(pytree1, snapshot_id="id1")
    manager.save_snapshot(pytree2, snapshot_id="id2")

    differences = manager.compare_snapshots("id1", "id2")

    # Expected differences as a PyTree
    expected_differences = {"a": PytreeSnapshotManager.NO_DIFFERENCE, "b": {"x": PytreeSnapshotManager.NO_DIFFERENCE}}

    assert differences == expected_differences


def test_nested_structures(manager):
    # Nested PyTree comparison
    pytree1 = {"a": {"x": 1, "y": 2}, "b": 3}
    pytree2 = {"a": {"x": 1, "y": 3}, "b": 3, "c": 4}

    manager.save_snapshot(pytree1, snapshot_id="id1")
    manager.save_snapshot(pytree2, snapshot_id="id2")

    differences = manager.compare_snapshots("id1", "id2")

    # Expected differences as a PyTree
    expected_differences = {
        "a": {"x": PytreeSnapshotManager.NO_DIFFERENCE, "y": (2, 3)},
        "b": PytreeSnapshotManager.NO_DIFFERENCE,
        "c": (PytreeSnapshotManager.LEAF_MISSING, 4),
    }

    assert differences == expected_differences


def test_empty_pytrees(manager):
    # Comparing empty PyTrees
    pytree1 = {}
    pytree2 = {}

    manager.save_snapshot(pytree1, snapshot_id="id1")
    manager.save_snapshot(pytree2, snapshot_id="id2")

    differences = manager.compare_snapshots("id1", "id2")

    # Expected differences as a PyTree
    expected_differences = {}

    assert differences == expected_differences


def test_mismatched_keys(manager):
    # Mismatched keys
    pytree1 = {"a": 1}
    pytree2 = {"b": 2}

    manager.save_snapshot(pytree1, snapshot_id="id1")
    manager.save_snapshot(pytree2, snapshot_id="id2")

    differences = manager.compare_snapshots("id1", "id2")

    # Expected differences as a PyTree
    expected_differences = {"a": (1, PytreeSnapshotManager.LEAF_MISSING), "b": (PytreeSnapshotManager.LEAF_MISSING, 2)}

    assert differences == expected_differences


def test_custom_comparator(manager):
    # Custom comparator function
    def custom_comparator(x, y):
        return x == y

    pytree1 = {"a": 10, "b": 20}
    pytree2 = {"a": 15, "b": 20, "x": 12}

    manager.save_snapshot(pytree1, snapshot_id="id1")
    manager.save_snapshot(pytree2, snapshot_id="id2")

    # Perform comparison with the custom comparator
    differences = manager.compare_snapshots(
        "id1", "id2", custom_comparator=custom_comparator
    )

    # Expected differences as a PyTree
    expected_differences = {"a": (10, 15), "b": PytreeSnapshotManager.NO_DIFFERENCE, "x": (PytreeSnapshotManager.LEAF_MISSING, 12)}

    # Assert that the PyTree structures match
    assert differences == expected_differences


def test_condition_function(manager):
    # Comparison with condition function
    def condition(x):
        return isinstance(x, int) and x > 10

    pytree1 = {"a": 5, "b": 15}
    pytree2 = {"a": 5, "b": 20, "x": "aa"}

    manager.save_snapshot(pytree1, snapshot_id="id1")
    manager.save_snapshot(pytree2, snapshot_id="id2")

    differences = manager.compare_snapshots("id1", "id2", condition=condition)

    # Expected differences as a PyTree
    expected_differences = {"a": PytreeSnapshotManager.NOT_COMPARED, "b": (15, 20), "x": PytreeSnapshotManager.NOT_COMPARED}

    assert differences == expected_differences


def test_custom_node_with_children(manager):
    # Define a custom node class with nested children
    class CustomNode:
        def __init__(self, children, metadata=PytreeSnapshotManager.NO_DIFFERENCE):
            self.children = children
            self.metadata = metadata or {}

        def __repr__(self):
            return f"CustomNode(children={self.children}, metadata={self.metadata})"

        def __eq__(self, other):
            if not isinstance(other, CustomNode):
                return False
            return self.children == other.children and self.metadata == other.metadata
    
    # Define flattening and unflattening functions
    def flatten_custom_node_with_children(node):
        return node.children, node.metadata

    def unflatten_custom_node_with_children(aux_data, children):
        return CustomNode(children, aux_data)

    # Register the custom node with JAX
    register_pytree_node(
        CustomNode,
        flatten_custom_node_with_children,
        unflatten_custom_node_with_children,
    )

    # Create PyTrees with custom nodes
    pytree1 = {
        "a": CustomNode([{"x": 1}, {"y": 2}], metadata={"type": "nested"}),
        "b": 3,
    }
    pytree2 = {
        "a": CustomNode([{"x": 1}, {"y": 3}], metadata={"type": "nested"}),
        "b": 3,
        "c": 4,
    }

    manager.save_snapshot(pytree1, snapshot_id="id1")
    manager.save_snapshot(pytree2, snapshot_id="id2")

    # Compare SnapTrees
    differences = manager.compare_snapshots("id1", "id2")

    # Expected differences as a PyTree
    expected_differences = {
        "a" : CustomNode(children=({"x": PytreeSnapshotManager.NO_DIFFERENCE}, {"y": (2, 3)}), metadata={"type": "nested"}),
        "b": PytreeSnapshotManager.NO_DIFFERENCE,
        "c": (PytreeSnapshotManager.LEAF_MISSING,4)
    }

    assert differences == expected_differences
