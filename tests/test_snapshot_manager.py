import pytest
import jax.numpy as jnp
from pytree_snapshots.snapshot_manager import SnapshotManager


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
