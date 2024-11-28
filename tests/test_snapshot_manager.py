import pytest
import jax.numpy as jnp

from snapshot_manager.snapshot_manager import SnapshotManager
from snapshot_manager.query import (
    AndQuery,
    OrQuery,
    NotQuery,
    ByMetadataQuery,
    ByTagQuery,
    ByContentQuery,
)


@pytest.fixture
def setup_manager():
    """Fixture to set up a SnapshotManager instance."""
    return SnapshotManager(max_snapshots=5)


def test_save_and_retrieve_snapshot(setup_manager):
    """Test saving and retrieving a snapshot."""
    manager = setup_manager  # Fixture provides a SnapshotManager instance

    # Define a PyTree to store
    pytree = {"a": jnp.array([1, 2, 3]), "b": {"x": jnp.array([4, 5])}}

    # Save the PyTree as a snapshot
    snapshot_id = manager.save_snapshot(pytree)

    # Retrieve the snapshot
    retrieved_snapshot = manager.get_snapshot(snapshot_id)

    # Assertions to verify the integritytests/test_snapshot_manager.py::test_snapshot_metadatatests/test_snapshot_manager.py::test_snapshot_metadata of the saved and retrieved PyTree
    assert jnp.array_equal(
        retrieved_snapshot.data["a"], pytree["a"]
    ), "Array 'a' does not match."
    assert jnp.array_equal(
        retrieved_snapshot.data["b"]["x"], pytree["b"]["x"]
    ), "Array 'b.x' does not match."


def test_snapshot_metadata(setup_manager):
    """Test adding and retrieving metadata for a snapshot."""
    manager = setup_manager
    pytree = {"a": jnp.array([1, 2, 3])}
    metadata = {"experiment": "test", "iteration": 42}
    snapshot_id = manager.save_snapshot(pytree, metadata=metadata)

    retrieved_metadata = manager.get_metadata(snapshot_id)
    assert retrieved_metadata == metadata

    new_metadata = {"accuracy": 0.95}
    manager.update_metadata(snapshot_id, new_metadata)
    updated_metadata = manager.get_metadata(snapshot_id)
    assert updated_metadata["accuracy"] == 0.95
    assert updated_metadata["experiment"] == "test"


def test_snapshot_tags():
    """Test adding and removing tags for a snapshot."""
    manager = SnapshotManager()
    pytree = {"a": jnp.array([1, 2, 3])}

    # Save the PyTree with initial tags
    snapshot_id = manager.save_snapshot(pytree, tags=["important"])

    # Add new tags
    manager.add_tags(snapshot_id, ["new", "experiment"])
    tags = manager.get_tags(snapshot_id)
    assert "new" in tags, "Tag 'new' should be present."
    assert "important" in tags, "Tag 'important' should be present."

    # Remove a tag
    manager.remove_tags(snapshot_id, ["important"])
    tags = manager.get_tags(snapshot_id)
    assert "important" not in tags, "Tag 'important' should not be present."
    assert "experiment" in tags, "Tag 'experiment' should still be present."


def test_insertion_order_limit(setup_manager):
    """Test enforcing the max_snapshots limit."""
    manager = SnapshotManager(
        max_snapshots=4
    )  # Fixture provides a SnapshotManager instance

    # Save more snapshots than the max_snapshots limit
    for i in range(6):
        manager.save_snapshot({"val": i})

    # Verify the number of snapshots does not exceed the max_snapshots limit
    assert (
        len(manager.storage.snapshots) == manager.storage.max_snapshots
    ), "The number of snapshots exceeds the max_snapshots limit."

    # Verify the order of snapshots
    insertion_order = manager.storage.insertion_order
    assert (
        len(insertion_order) == manager.storage.max_snapshots
    ), "The snapshot order does not match the max_snapshots limit."

    # The oldest snapshot (first inserted) should have been removed
    oldest_remaining_snapshot = insertion_order[0]
    oldest_remaining_pytree = manager.get_snapshot(oldest_remaining_snapshot)
    assert (
        oldest_remaining_pytree.data["val"] == 2
    ), "The oldest snapshot was not removed correctly."


def test_save_and_load_from_file(tmp_path, setup_manager):
    """Test saving and loading the manager state."""
    manager = setup_manager  # Fixture provides a SnapshotManager instance

    # Create and save snapshots
    pytree1 = {"a": jnp.array([1, 2, 3])}
    pytree2 = {"b": jnp.array([4, 5, 6])}
    snapshot_id1 = manager.save_snapshot(pytree1, metadata={"key": "value1"})
    snapshot_id2 = manager.save_snapshot(pytree2, metadata={"key": "value2"})

    # Save the manager state to a file
    state_file = tmp_path / "state.pkl"
    manager.save_to_file(state_file)

    # Load the state into a new manager
    loaded_manager = SnapshotManager.load_from_file(state_file)

    # Validate that the snapshots were correctly restored
    assert len(loaded_manager.storage.snapshots) == len(
        manager.storage.snapshots
    ), "The number of snapshots in the loaded manager does not match the original."

    # Validate metadata of the restored snapshots
    assert (
        loaded_manager.get_metadata(snapshot_id1)["key"] == "value1"
    ), "Metadata for the first snapshot does not match."
    assert (
        loaded_manager.get_metadata(snapshot_id2)["key"] == "value2"
    ), "Metadata for the second snapshot does not match."

    # Validate the contents of the restored snapshots
    assert jnp.array_equal(
        loaded_manager.get_snapshot(snapshot_id1).data["a"], pytree1["a"]
    ), "The first snapshot's PyTree content does not match."
    assert jnp.array_equal(
        loaded_manager.get_snapshot(snapshot_id2).data["b"], pytree2["b"]
    ), "The second snapshot's PyTree content does not match."


def test_logical_queries(setup_manager):
    """Test logical queries using AndQuery, OrQuery, and NotQuery."""
    manager = setup_manager

    # Save snapshots
    manager.save_snapshot(
        {"a": 1},
        snapshot_id="snap1",
        metadata={"project": "example1"},
        tags=["experiment"],
    )
    manager.save_snapshot(
        {"b": 2},
        snapshot_id="snap2",
        metadata={"project": "example2"},
        tags=["control"],
    )
    manager.save_snapshot(
        {"c": 3},
        snapshot_id="snap3",
        metadata={"project": "example1"},
        tags=["experiment", "published"],
    )

    # Logical Query: Find snapshots in project "example1" AND tagged with "experiment"
    query = AndQuery(ByMetadataQuery("project", "example1"), ByTagQuery("experiment"))
    results = manager.query.evaluate(query)
    assert "snap1" in results and "snap3" in results, "Logical AND query failed."

    # Logical Query: Find snapshots in project "example1" OR tagged with "control"
    query = OrQuery(ByMetadataQuery("project", "example1"), ByTagQuery("control"))
    results = manager.query.evaluate(query)
    assert (
        "snap1" in results and "snap2" in results and "snap3" in results
    ), "Logical OR query failed."

    # Logical Query: Find snapshots NOT tagged with "control"
    query = NotQuery(ByTagQuery("control"))
    results = manager.query.evaluate(query)
    assert (
        "snap1" in results and "snap3" in results and "snap2" not in results
    ), "Logical NOT query failed."


from snapshot_manager.query import ByContentQuery


def test_by_content_query(setup_manager):
    """Test querying snapshots based on their content."""
    manager = setup_manager

    # Save snapshots with complex content
    manager.save_snapshot({"key": 1, "nested": {"a": 2}}, snapshot_id="snap1")
    manager.save_snapshot({"key": 3}, snapshot_id="snap2")
    manager.save_snapshot({"nested": {"b": 4}}, snapshot_id="snap3")

    # Query for snapshots containing a specific key
    query = ByContentQuery(lambda content: "key" in content)
    results = manager.query.evaluate(query)
    assert (
        "snap1" in results and "snap2" in results and "snap3" not in results
    ), "ByContentQuery failed for key existence."

    # Query for snapshots with nested key "a"
    query = ByContentQuery(
        lambda content: "nested" in content and "a" in content["nested"]
    )
    results = manager.query.evaluate(query)
    assert (
        "snap1" in results and "snap2" not in results and "snap3" not in results
    ), "ByContentQuery failed for nested key."


def test_remove_snapshot(setup_manager):
    """Test removing a snapshot."""
    manager = setup_manager

    # Save snapshots
    snapshot_id1 = manager.save_snapshot({"a": 1})
    snapshot_id2 = manager.save_snapshot({"b": 2})

    # Remove the first snapshot
    manager.remove_snapshot(snapshot_id1)

    # Verify the snapshot is removed
    with pytest.raises(KeyError, match="Snapshot with ID .* not found"):
        manager.get_snapshot(snapshot_id1)

    # Verify the remaining snapshot is unaffected
    retrieved = manager.get_snapshot(snapshot_id2)
    assert retrieved.data["b"] == 2, "Remaining snapshot was affected by removal."


def test_duplicate_snapshots(setup_manager):
    """Test saving duplicate snapshots."""
    manager = setup_manager

    # Save identical snapshots
    snapshot_id1 = manager.save_snapshot({"a": 1})
    snapshot_id2 = manager.save_snapshot({"a": 1})

    # Verify that the snapshots have distinct IDs
    assert snapshot_id1 != snapshot_id2, "Duplicate snapshots have the same ID."

    # Verify that both snapshots are accessible
    retrieved1 = manager.get_snapshot(snapshot_id1)
    retrieved2 = manager.get_snapshot(snapshot_id2)
    assert (
        retrieved1.data == retrieved2.data
    ), "Duplicate snapshots should have identical content."


def test_insertion_order_after_state_restore(tmp_path, setup_manager):
    """Test that snapshot order is preserved after restoring state."""
    manager = setup_manager

    # Save snapshots
    manager.save_snapshot({"a": 1}, snapshot_id="snap1")
    manager.save_snapshot({"b": 2}, snapshot_id="snap2")

    # Save the manager state
    state_file = tmp_path / "state.pkl"
    manager.save_to_file(state_file)

    # Restore the state
    restored_manager = SnapshotManager.load_from_file(state_file)

    # Verify that the snapshot order is preserved
    assert (
        restored_manager.storage.insertion_order == manager.storage.insertion_order
    ), "Snapshot order was not preserved after restoring state."


from snapshot_manager.pytree_snapshot_manager import PyTreeSnapshotManager
import jax.numpy as jnp


def test_query_by_leaf_value_simple_condition():
    """Test querying snapshots with a simple condition on leaf values."""
    manager = PyTreeSnapshotManager()

    # Save PyTree snapshots
    manager.save_snapshot(
        {"a": jnp.array([1, 2, 3]), "b": {"x": jnp.array([4, 5])}}, snapshot_id="snap1"
    )
    manager.save_snapshot(
        {"x": jnp.array([6, 7, 8]), "y": {"z": jnp.array([9])}}, snapshot_id="snap2"
    )
    manager.save_snapshot({"p": jnp.array([-1, -2])}, snapshot_id="snap3")

    # Query for snapshots where any leaf contains a value > 5
    query = manager.query.by_leaf_value(lambda x: jnp.any(x > 5))
    results = manager.query.evaluate(query)

    # Assert that only the relevant snapshots are returned
    assert "snap2" in results, "Snapshot with leaf value > 5 is missing."
    assert (
        "snap1" not in results
    ), "Snapshot with no leaf value > 5 is incorrectly included."
    assert (
        "snap3" not in results
    ), "Snapshot with no leaf value > 5 is incorrectly included."


def test_query_by_leaf_value_complex_condition():
    """Test querying snapshots with a complex condition on leaf values."""
    manager = PyTreeSnapshotManager()

    # Save PyTree snapshots
    manager.save_snapshot(
        {"a": jnp.array([1, 2, 3]), "b": {"x": jnp.array([4, 5])}}, snapshot_id="snap1"
    )
    manager.save_snapshot(
        {"x": jnp.array([-6, 7, 8]), "y": {"z": jnp.array([9])}}, snapshot_id="snap2"
    )
    manager.save_snapshot({"p": jnp.array([-1, -2])}, snapshot_id="snap3")

    # Query for snapshots where any leaf contains a negative value
    query = manager.query.by_leaf_value(lambda x: jnp.any(x < 0))
    results = manager.query.evaluate(query)

    # Assert that the relevant snapshots are returned
    assert "snap3" in results, "Snapshot with negative leaf values is missing."
    assert "snap2" in results, "Snapshot with negative leaf values is missing."
    assert (
        "snap1" not in results
    ), "Snapshot with no negative leaf values is incorrectly included."


def test_prune_snapshots_by_accuracy():
    """Test pruning snapshots by accuracy when max_snapshots is reached."""

    def cmp_by_accuracy(snapshot1, snapshot2):
        return snapshot1.metadata.get("accuracy", 0) - snapshot2.metadata.get(
            "accuracy", 0
        )

    # Initialize manager with a maximum of 3 snapshots
    manager = SnapshotManager(max_snapshots=3, cmp=cmp_by_accuracy)

    # Save snapshots with varying accuracy
    manager.save_snapshot({"a": 1}, snapshot_id="snap1", metadata={"accuracy": 0.5})
    manager.save_snapshot({"b": 2}, snapshot_id="snap2", metadata={"accuracy": 0.7})
    manager.save_snapshot({"c": 3}, snapshot_id="snap3", metadata={"accuracy": 0.6})

    # Save a new snapshot with higher accuracy
    manager.save_snapshot({"d": 4}, snapshot_id="snap4", metadata={"accuracy": 0.8})

    # Verify that only the top 3 snapshots are retained
    snapshots = manager.get_ids_by_rank()
    assert (
        len(snapshots) == 3
    ), "Number of retained snapshots does not match max_snapshots."
    assert "snap1" not in snapshots, "Lowest accuracy snapshot was not removed."
    assert snapshots == [
        "snap4",
        "snap2",
        "snap3",
    ], "Snapshots are not ordered by accuracy."


def test_reject_low_ranked_snapshot():
    """Test rejecting a low-ranked snapshot when max_snapshots is reached."""

    def cmp_by_accuracy(snapshot1, snapshot2):
        return snapshot1.metadata.get("accuracy", 0) - snapshot2.metadata.get(
            "accuracy", 0
        )

    # Initialize manager with a maximum of 3 snapshots
    manager = SnapshotManager(max_snapshots=3, cmp=cmp_by_accuracy)

    # Save snapshots with varying accuracy
    manager.save_snapshot({"a": 1}, snapshot_id="snap1", metadata={"accuracy": 0.5})
    manager.save_snapshot({"b": 2}, snapshot_id="snap2", metadata={"accuracy": 0.7})
    manager.save_snapshot({"c": 3}, snapshot_id="snap3", metadata={"accuracy": 0.6})

    # Attempt to save a new snapshot with lower accuracy than the current lowest
    manager.save_snapshot({"e": 5}, snapshot_id="snap5", metadata={"accuracy": 0.4})

    # Verify that the low-ranked snapshot was not added
    snapshots = manager.get_ids_by_rank()
    assert len(snapshots) == 3, "Number of snapshots exceeds max_snapshots."
    assert "snap5" not in snapshots, "Low-ranked snapshot was incorrectly added."
    assert snapshots == [
        "snap2",
        "snap3",
        "snap1",
    ], "Snapshots are not ordered correctly after rejection."


def test_override_deepcopy_on_retrieve():
    """Test overriding deepcopy_on_retrieve during a retrieval operation."""
    manager = SnapshotManager(deepcopy_on_save=False)

    # Save a snapshot
    pytree = {"a": [1, 2, 3]}
    snapshot_id = manager.save_snapshot(pytree)

    # Retrieve the snapshot without deepcopy
    retrieved = manager.get_snapshot(snapshot_id, deepcopy=False)

    # Modify the retrieved PyTree
    retrieved.data["a"].append(4)

    # Retrieve the snapshot again
    original = manager.get_snapshot(snapshot_id)

    # Assert the original and retrieved are not isolated
    assert original.data["a"] == [
        1,
        2,
        3,
        4,
    ], "Deepcopy override on retrieve did not work correctly."
    assert retrieved.data["a"] == [
        1,
        2,
        3,
        4,
    ], "Modified retrieved PyTree was not as expected."


def test_default_deepcopy_logic():
    """Test the default deepcopy settings for saving and retrieving snapshots."""
    manager = SnapshotManager(deepcopy_on_save=True, deepcopy_on_retrieve=True)

    # Save a snapshot with default deepcopy setting
    pytree = {"a": [1, 2, 3]}
    snapshot_id = manager.save_snapshot(pytree)

    # Modify the original PyTree
    pytree["a"].append(4)

    # Retrieve the snapshot
    retrieved = manager.get_snapshot(snapshot_id)

    # Assert the original and retrieved are isolated
    assert retrieved.data["a"] == [
        1,
        2,
        3,
    ], "Deepcopy on save failed to isolate the snapshot."
    assert pytree["a"] == [1, 2, 3, 4], "Original PyTree was unexpectedly modified."


def test_override_deepcopy_on_save():
    """Test overriding deepcopy_on_save during a save operation."""
    manager = SnapshotManager(deepcopy_on_save=True)

    # Save a snapshot with deepcopy explicitly disabled
    pytree = {"a": [1, 2, 3]}
    snapshot_id = manager.save_snapshot(pytree, deepcopy=False)

    # Modify the original PyTree
    pytree["a"].append(4)

    # Retrieve the snapshot
    retrieved = manager.get_snapshot(snapshot_id)

    # Assert the original and retrieved are not isolated
    assert retrieved.data["a"] == [
        1,
        2,
        3,
        4,
    ], "Deepcopy override on save did not work correctly."


def test_tree_map():
    """Test updating all snapshots' leaf nodes in place."""
    manager = PyTreeSnapshotManager()

    sid = manager.save_snapshot(
        {
            "txt": "hello pytorch",
            "x": 42,
        }
    )

    new_pytree = manager.tree_map(
        lambda x: x.replace("pytorch", "jax") if isinstance(x, str) else x,
        snapshot_ids=sid,
    )

    assert new_pytree["txt"] == "hello jax"


def test_tree_combine_average():
    """Test combining snapshots using an average function."""
    manager = PyTreeSnapshotManager()

    # Save snapshots with PyTree structures
    snapshot1 = {"layer1": jnp.array([1.0, 2.0]), "layer2": jnp.array([3.0])}
    snapshot2 = {"layer1": jnp.array([4.0, 5.0]), "layer2": jnp.array([6.0])}
    snapshot3 = {"layer1": jnp.array([7.0, 8.0]), "layer2": jnp.array([9.0])}

    manager.save_snapshot(snapshot1, snapshot_id="snap1")
    manager.save_snapshot(snapshot2, snapshot_id="snap2")
    manager.save_snapshot(snapshot3, snapshot_id="snap3")

    # Combine snapshots with an average function
    def average_leaves(leaves):
        return sum(leaves) / len(leaves)

    combined_pytree = manager.tree_combine(
        snapshot_ids=["snap1", "snap2", "snap3"], combine_fn=average_leaves
    )


    # Verify the combined PyTree
    # Expected result
    expected_pytree = {"layer1": jnp.array([4.0, 5.0]), "layer2": jnp.array([6.0])}

    # Verify the combined PyTree
    for key in combined_pytree.keys():
        assert jnp.array_equal(
            combined_pytree[key], expected_pytree[key]
        ), f"Mismatch for key {key}: {combined_pytree[key]} != {expected_pytree[key]}"


def test_by_tags_and_logic():
    snapshot_manager = SnapshotManager()
    # Setup: Add snapshots with specific tags
    snapshot_manager.save_snapshot(
        {"data": "snapshot1"}, tags=["important", "completed"]
    )
    snapshot_manager.save_snapshot({"data": "snapshot2"}, tags=["important"])
    snapshot_manager.save_snapshot({"data": "snapshot3"}, tags=["completed"])

    # Test: Find snapshots matching all specified tags
    result = snapshot_manager.query.by_tags(
        ["important", "completed"], require_all=True
    )

    # Assert: Only the first snapshot matches all tags
    assert len(result) == 1
    assert result[0] == "snapshot1"


def test_by_tags_and_logic():
    snapshot_manager = SnapshotManager()

    # Setup: Add snapshots with specific tags and capture IDs
    snapshots = {
        "snapshot1": snapshot_manager.save_snapshot(
            {"data": "snapshot1"}, tags=["important", "completed"]
        ),
        "snapshot2": snapshot_manager.save_snapshot(
            {"data": "snapshot2"}, tags=["important"]
        ),
        "snapshot3": snapshot_manager.save_snapshot(
            {"data": "snapshot3"}, tags=["completed"]
        ),
    }

    # Test: Find snapshots matching all specified tags
    result = snapshot_manager.query.by_tags(
        ["important", "completed"], require_all=True
    )

    # Assert: Only the first snapshot matches all tags
    assert len(result) == 1
    assert result[0] == snapshots["snapshot1"]


def test_tree_replace(setup_manager):
   
    manager = PyTreeSnapshotManager()

    # Save snapshots with PyTree structures
    snapshot1 = {"a": jnp.array([1, 2, 3]), "b": jnp.array([4, 5, 6])}
    snapshot2 = {"c": jnp.array([7, 8, 9]), "d": jnp.array([10, 11, 12])}
    snap1_id = manager.save_snapshot(snapshot1, snapshot_id="snap1")
    snap2_id = manager.save_snapshot(snapshot2, snapshot_id="snap2")

    # Define a transformation function
    def increment_array(x):
        return x + 1 if isinstance(x, jnp.ndarray) else x

    # Replace trees in all snapshots
    manager.tree_replace(func=increment_array)

    # Retrieve and validate transformed snapshots
    transformed_snapshot1 = manager.get_snapshot(snap1_id)
    transformed_snapshot2 = manager.get_snapshot(snap2_id)

    assert jnp.array_equal(
        transformed_snapshot1.data["a"], jnp.array([2, 3, 4])
    ), "Snapshot1 'a' was not correctly transformed."
    assert jnp.array_equal(
        transformed_snapshot1.data["b"], jnp.array([5, 6, 7])
    ), "Snapshot1 'b' was not correctly transformed."

    assert jnp.array_equal(
        transformed_snapshot2.data["c"], jnp.array([8, 9, 10])
    ), "Snapshot2 'c' was not correctly transformed."
    assert jnp.array_equal(
        transformed_snapshot2.data["d"], jnp.array([11, 12, 13])
    ), "Snapshot2 'd' was not correctly transformed."

    # Test replacing a specific snapshot
    def multiply_array(x):
        return x * 2 if isinstance(x, jnp.ndarray) else x

    manager.tree_replace(func=multiply_array, snapshot_ids="snap1")

    transformed_snapshot1 = manager.get_snapshot(snap1_id)
    transformed_snapshot2 = manager.get_snapshot(snap2_id)

    assert jnp.array_equal(
        transformed_snapshot1.data["a"], jnp.array([4, 6, 8])
    ), "Snapshot1 'a' was not correctly multiplied."
    assert jnp.array_equal(
        transformed_snapshot1.data["b"], jnp.array([10, 12, 14])
    ), "Snapshot1 'b' was not correctly multiplied."

    # Snapshot2 should remain unchanged
    assert jnp.array_equal(
        transformed_snapshot2.data["c"], jnp.array([8, 9, 10])
    ), "Snapshot2 'c' was incorrectly transformed."
    assert jnp.array_equal(
        transformed_snapshot2.data["d"], jnp.array([11, 12, 13])
    ), "Snapshot2 'd' was incorrectly transformed."


def test_cmp_edge_cases():
    """Test edge cases for custom cmp function."""

    def cmp(snap1, snap2):
        # Prioritize snapshots with lower metadata values
        return snap1.metadata.get("priority", 0) - snap2.metadata.get("priority", 0)

    manager = SnapshotManager(max_snapshots=3, cmp=cmp)

    # Save snapshots with varying priority
    manager.save_snapshot({"a": 1}, snapshot_id="snap1", metadata={"priority": 2})
    manager.save_snapshot({"b": 2}, snapshot_id="snap2", metadata={"priority": 3})
    manager.save_snapshot({"c": 3}, snapshot_id="snap3", metadata={"priority": 1})

    # Save a snapshot with higher priority
    manager.save_snapshot({"d": 4}, snapshot_id="snap4", metadata={"priority": 4})

    # Validate that the snapshots are ordered by priority
    snapshot_ids = manager.get_ids_by_rank()
    assert snapshot_ids == [
        "snap4",
        "snap2",
        "snap1",
    ], "Snapshots not ranked correctly."

    # Save a snapshot with lower priority than the lowest in the list
    manager.save_snapshot({"e": 5}, snapshot_id="snap5", metadata={"priority": 0})

    # Ensure that the low-priority snapshot was not added
    snapshot_ids = manager.get_ids_by_rank()
    assert len(snapshot_ids) == 3, "Snapshot limit exceeded."
    assert "snap5" not in snapshot_ids, "Low-priority snapshot was incorrectly added."


def test_update_max_limit():
    """Test dynamically updating max_snapshots."""
    manager = SnapshotManager(max_snapshots=2)

    # Save snapshots to reach the limit
    manager.save_snapshot({"a": 1}, snapshot_id="snap1")
    manager.save_snapshot({"b": 2}, snapshot_id="snap2")

    # Increase the limit
    manager.update_max_snapshots(3)
    manager.save_snapshot({"c": 3}, snapshot_id="snap3")
    assert len(manager.storage.snapshots) == 3, "Failed to handle increased limit."

    # Decrease the limit
    manager.update_max_snapshots(2)
    snapshot_ids = manager.storage.insertion_order
    assert len(snapshot_ids) == 2, "Failed to handle decreased limit."
    assert "snap1" not in snapshot_ids, "Oldest snapshot not removed on limit decrease."


def test_switch_cmp():
    """Test switching between cmp functions."""

    def priority_cmp(snap1, snap2):
        return snap1.metadata.get("priority", 0) - snap2.metadata.get("priority", 0)

    def reverse_priority_cmp(snap1, snap2):
        return snap2.metadata.get("priority", 0) - snap1.metadata.get("priority", 0)

    manager = SnapshotManager(max_snapshots=3, cmp=priority_cmp)

    # Save snapshots with varying priority
    manager.save_snapshot({"a": 1}, snapshot_id="snap1", metadata={"priority": 2})
    manager.save_snapshot({"b": 2}, snapshot_id="snap2", metadata={"priority": 3})
    manager.save_snapshot({"c": 3}, snapshot_id="snap3", metadata={"priority": 1})

    # Switch to reverse priority cmp
    manager.update_cmp(reverse_priority_cmp)

    # Save a snapshot with medium priority
    manager.save_snapshot({"d": 4}, snapshot_id="snap4", metadata={"priority": 2.4})

    # Validate that snapshots are now ordered by reverse priority
    snapshot_ids = manager.get_ids_by_rank()
    assert snapshot_ids == [
        "snap3",
        "snap1",
        "snap4",
    ], "Snapshots not ranked correctly after cmp change."
