import pytest
import jax.numpy as jnp

from pytree_snapshots.snapshot_manager import SnapshotManager
from pytree_snapshots.query import (
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
    retrieved = manager.get_snapshot(snapshot_id)

    # Assertions to verify the integrity of the saved and retrieved PyTree
    assert jnp.array_equal(retrieved["a"], pytree["a"]), "Array 'a' does not match."
    assert jnp.array_equal(
        retrieved["b"]["x"], pytree["b"]["x"]
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


def test_snapshot_tags(setup_manager):
    """Test adding and removing tags for a snapshot."""
    manager = setup_manager
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


def test_snapshot_order_limit(setup_manager):
    """Test enforcing the max_snapshots limit."""
    manager = setup_manager  # Fixture provides a SnapshotManager instance

    # Save more snapshots than the max_snapshots limit
    for i in range(6):
        manager.save_snapshot({"val": i})

    # Verify the number of snapshots does not exceed the max_snapshots limit
    assert (
        len(manager.storage.snapshots) == manager.storage.max_snapshots
    ), "The number of snapshots exceeds the max_snapshots limit."

    # Verify the order of snapshots
    snapshot_order = manager.storage.snapshot_order
    assert (
        len(snapshot_order) == manager.storage.max_snapshots
    ), "The snapshot order does not match the max_snapshots limit."

    # The oldest snapshot (first inserted) should have been removed
    oldest_remaining_snapshot = snapshot_order[0]
    oldest_remaining_pytree = manager.get_snapshot(oldest_remaining_snapshot)
    assert (
        oldest_remaining_pytree["val"] == 1
    ), "The oldest snapshot was not removed correctly."


def test_save_and_load_state(tmp_path, setup_manager):
    """Test saving and loading the manager state."""
    manager = setup_manager  # Fixture provides a SnapshotManager instance

    # Create and save snapshots
    pytree1 = {"a": jnp.array([1, 2, 3])}
    pytree2 = {"b": jnp.array([4, 5, 6])}
    snapshot_id1 = manager.save_snapshot(pytree1, metadata={"key": "value1"})
    snapshot_id2 = manager.save_snapshot(pytree2, metadata={"key": "value2"})

    # Save the manager state to a file
    state_file = tmp_path / "state.pkl"
    manager.save_state(state_file)

    # Load the state into a new manager
    loaded_manager = SnapshotManager.load_state(state_file)

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
        loaded_manager.get_snapshot(snapshot_id1)["a"], pytree1["a"]
    ), "The first snapshot's PyTree content does not match."
    assert jnp.array_equal(
        loaded_manager.get_snapshot(snapshot_id2)["b"], pytree2["b"]
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


from pytree_snapshots.query import ByContentQuery


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
    assert retrieved["b"] == 2, "Remaining snapshot was affected by removal."


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
        retrieved1 == retrieved2
    ), "Duplicate snapshots should have identical content."


def test_snapshot_order_after_state_restore(tmp_path, setup_manager):
    """Test that snapshot order is preserved after restoring state."""
    manager = setup_manager

    # Save snapshots
    manager.save_snapshot({"a": 1}, snapshot_id="snap1")
    manager.save_snapshot({"b": 2}, snapshot_id="snap2")

    # Save the manager state
    state_file = tmp_path / "state.pkl"
    manager.save_state(state_file)

    # Restore the state
    restored_manager = SnapshotManager.load_state(state_file)

    # Verify that the snapshot order is preserved
    assert (
        restored_manager.storage.snapshot_order == manager.storage.snapshot_order
    ), "Snapshot order was not preserved after restoring state."

from pytree_snapshots.pytree_snapshot_manager import PyTreeSnapshotManager
import jax.numpy as jnp

def test_query_by_leaf_value_simple_condition():
    """Test querying snapshots with a simple condition on leaf values."""
    manager = PyTreeSnapshotManager()

    # Save PyTree snapshots
    manager.save_snapshot({"a": jnp.array([1, 2, 3]), "b": {"x": jnp.array([4, 5])}}, snapshot_id="snap1")
    manager.save_snapshot({"x": jnp.array([6, 7, 8]), "y": {"z": jnp.array([9])}}, snapshot_id="snap2")
    manager.save_snapshot({"p": jnp.array([-1, -2])}, snapshot_id="snap3")

    # Query for snapshots where any leaf contains a value > 5
    query = manager.query.by_leaf_value(lambda x: jnp.any(x > 5))
    results = manager.query.evaluate(query)

    # Assert that only the relevant snapshots are returned
    assert "snap2" in results, "Snapshot with leaf value > 5 is missing."
    assert "snap1" not in results, "Snapshot with no leaf value > 5 is incorrectly included."
    assert "snap3" not in results, "Snapshot with no leaf value > 5 is incorrectly included."

def test_query_by_leaf_value_complex_condition():
    """Test querying snapshots with a complex condition on leaf values."""
    manager = PyTreeSnapshotManager()

    # Save PyTree snapshots
    manager.save_snapshot({"a": jnp.array([1, 2, 3]), "b": {"x": jnp.array([4, 5])}}, snapshot_id="snap1")
    manager.save_snapshot({"x": jnp.array([-6, 7, 8]), "y": {"z": jnp.array([9])}}, snapshot_id="snap2")
    manager.save_snapshot({"p": jnp.array([-1, -2])}, snapshot_id="snap3")

    # Query for snapshots where any leaf contains a negative value
    query = manager.query.by_leaf_value(lambda x: jnp.any(x < 0))
    results = manager.query.evaluate(query)

    # Assert that the relevant snapshots are returned
    assert "snap3" in results, "Snapshot with negative leaf values is missing."
    assert "snap2" in results, "Snapshot with negative leaf values is missing."
    assert "snap1" not in results, "Snapshot with no negative leaf values is incorrectly included."