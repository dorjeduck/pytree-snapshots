from pytree_snapshots import PytreeSnapshotManager
import jax.numpy as jnp

# Initialize the manager
manager = PytreeSnapshotManager()

# Custom comparator function for floats
def float_tolerance_comparator(x, y, tolerance=1e-5):
    if isinstance(x, (float, jnp.ndarray)) and isinstance(y, (float, jnp.ndarray)):
        return abs(x - y) <= tolerance
    return x == y  # Fallback for other types

# Sample PyTrees with floating-point values
pytree1 = {"a": 1.0, "b": 3.1415926,"c":2.1}
pytree2 = {"a": 1.0, "b": 3.14159,"c":2.2}

# Save SnapTrees
manager.save_snapshot(pytree1, snapshot_id="snap1")
manager.save_snapshot(pytree2, snapshot_id="snap2")

# Compare SnapTrees with custom comparator
differences = manager.compare_snapshots(
    "snap1",
    "snap2",
    custom_comparator=lambda x, y: float_tolerance_comparator(x, y, tolerance=1e-4),
)

print("Differences:", differences)