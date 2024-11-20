import pytest
import jax.numpy as jnp
from pytree_snapshots.pytree_snapshot_manager import PytreeSnapshotManager

@pytest.fixture
def setup_manager():
    """Fixture to set up a PytreeSnapshotManager instance."""
    return PytreeSnapshotManager(max_snapshots=5)

def test_save_and_retrieve_snapshot(setup_manager):
    """Test saving and retrieving a snapshot."""
    manager = setup_manager
    pytree = {"a": jnp.array([1, 2, 3]), "b": {"x": jnp.array([4, 5])}}
    snapshot_id = manager.save_snapshot(pytree)

    retrieved = manager.get_snapshot(snapshot_id)
    assert jnp.array_equal(retrieved["a"], pytree["a"])
    assert jnp.array_equal(retrieved["b"]["x"], pytree["b"]["x"])

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
    snapshot_id = manager.save_snapshot(pytree, tags=["important"])

    manager.add_tags(snapshot_id, ["new", "experiment"])
    assert manager.snapshots[snapshot_id].has_tag("new")
    assert manager.snapshots[snapshot_id].has_tag("important")

    manager.remove_tags(snapshot_id, ["important"])
    assert not manager.snapshots[snapshot_id].has_tag("important")
    assert manager.snapshots[snapshot_id].has_tag("experiment")

def test_snapshot_order_limit(setup_manager):
    """Test enforcing the max_snapshots limit."""
    manager = setup_manager
    for i in range(6):
        manager.save_snapshot({"val": i})

    assert len(manager.snapshots) == 5
    assert manager.snapshot_order[0] != 0  # Oldest snapshot removed

def test_save_and_load_state(tmp_path, setup_manager):
    """Test saving and loading the manager state."""
    manager = setup_manager
    pytree1 = {"a": jnp.array([1, 2, 3])}
    pytree2 = {"b": jnp.array([4, 5, 6])}
    manager.save_snapshot(pytree1, metadata={"key": "value1"})
    manager.save_snapshot(pytree2, metadata={"key": "value2"})

    state_file = tmp_path / "state.pkl"
    manager.save_state(state_file)

    loaded_manager = PytreeSnapshotManager.load_state(state_file)
    assert len(loaded_manager.snapshots) == len(manager.snapshots)
    assert loaded_manager.get_metadata(manager.snapshot_order[0])["key"] == "value1"
    assert loaded_manager.get_metadata(manager.snapshot_order[1])["key"] == "value2"