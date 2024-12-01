"""Example demonstrating SnapshotManager with JAX for MNIST training.

This example shows:
1. Training a simple MLP on MNIST using JAX
2. Using SnapshotManager to track model checkpoints
3. Ranking snapshots by accuracy
4. Visualizing training progress
"""

import torch
import torchvision
from torchvision import transforms
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
from pathlib import Path

from snapshot_manager import PyTreeSnapshotManager

# Set parameters
BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
PRINT_EVERY = 10
SAVE_EVERY = 5  # Save snapshot every N epochs

# Create directory for MNIST data
Path("MNIST").mkdir(exist_ok=True)

print("\nLoading and preprocessing MNIST dataset...")

# Normalization transformation for MNIST
normalize_data = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# Load MNIST dataset using PyTorch's DataLoader
train_dataset = torchvision.datasets.MNIST(
    root="MNIST", train=True, download=True, transform=normalize_data
)
test_dataset = torchvision.datasets.MNIST(
    root="MNIST", train=False, download=True, transform=normalize_data
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Convert PyTorch tensors to NumPy arrays to use with JAX
def pytorch_to_numpy(data_loader):
    """Convert PyTorch DataLoader to NumPy arrays."""
    images, labels = [], []
    for x_batch, y_batch in data_loader:
        images.append(x_batch.view(-1, 28 * 28).numpy())  # Flatten the images
        labels.append(y_batch.numpy())
    images = np.vstack(images)
    labels = np.concatenate(labels)
    return images, labels

print("\nConverting data to JAX format...")
train_images, train_labels = pytorch_to_numpy(train_loader)
train_labels = jax.nn.one_hot(train_labels, num_classes=10)
test_images, test_labels = pytorch_to_numpy(test_loader)
test_labels = jax.nn.one_hot(test_labels, num_classes=10)

# Initialize MLP parameters
def init_mlp_params(layer_widths):
    """Initialize MLP parameters with Xavier initialization."""
    params = []
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        params.append(
            {
                "weights": np.random.normal(size=(n_in, n_out)) * np.sqrt(2 / n_in),
                "biases": np.zeros(shape=(n_out,)),  # Initialize biases to zero
            }
        )
    return params

print("\nInitializing MLP parameters...")
params = init_mlp_params([784, 256, 128, 10])

# Define the forward pass (using softmax for classification)
def forward(params, x):
    """Forward pass through the MLP."""
    *hidden, last = params
    for layer in hidden:
        x = jax.nn.relu(x @ layer["weights"] + layer["biases"])
    return jax.nn.softmax(x @ last["weights"] + last["biases"], axis=-1)

# Define the cross-entropy loss function
@jax.jit
def loss_fn(params, x, y):
    """Compute cross-entropy loss."""
    predictions = forward(params, x)
    return -jnp.mean(jnp.sum(y * jnp.log(predictions + 1e-10), axis=1))

# Update function using JAX's jit and grad
@jax.jit
def update(params, x, y, lr=LEARNING_RATE):
    """Update parameters using gradient descent."""
    grads = jax.grad(loss_fn)(params, x, y)
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)

# Define accuracy calculation
def compute_accuracy(params, x, y):
    """Compute classification accuracy."""
    predictions = forward(params, x)
    predicted_classes = jnp.argmax(predictions, axis=1)
    true_classes = jnp.argmax(y, axis=1)
    return jnp.mean(predicted_classes == true_classes)

# Initialize tracking variables
loss_history = []
accuracy_history = []
best_accuracy = 0.0

# Initialize PyTreeSnapshotManager with max_snapshots and custom comparison function
def cmp_by_accuracy(snapshot1, snapshot2):
    """Compare snapshots by their accuracy."""
    return snapshot1.metadata["accuracy"] - snapshot2.metadata["accuracy"]

print("\nInitializing PyTreeSnapshotManager...")
ptsm = PyTreeSnapshotManager(
    deepcopy_on_retrieve=False, max_snapshots=5, cmp=cmp_by_accuracy
)

print("\nStarting training...")
print("=" * 80)

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()

    # Training step
    for i in range(0, len(train_images), BATCH_SIZE):
        x_batch = train_images[i : i + BATCH_SIZE]
        y_batch = train_labels[i : i + BATCH_SIZE]
        params = update(params, x_batch, y_batch)

    # Calculate metrics
    epoch_duration = time.time() - epoch_start_time
    current_loss = loss_fn(params, train_images, train_labels)
    test_accuracy = compute_accuracy(params, test_images, test_labels)
    
    # Update histories
    loss_history.append(float(current_loss))
    accuracy_history.append(float(test_accuracy))

    # Print progress
    if (epoch + 1) % PRINT_EVERY == 0:
        print(
            f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} | "
            f"Loss: {current_loss:.4f} | "
            f"Test Accuracy: {test_accuracy * 100:.2f}% | "
            f"Time: {epoch_duration:.2f}s"
        )

    # Save snapshot every SAVE_EVERY epochs or if it's the best model
    if (epoch + 1) % SAVE_EVERY == 0 or test_accuracy > best_accuracy:
        snapshot_id = f"epoch_{epoch + 1}"
        ptsm.save_snapshot(
            params,
            snapshot_id=snapshot_id,
            metadata={
                "epoch": epoch + 1,
                "loss": float(current_loss),
                "accuracy": float(test_accuracy),
                "duration": epoch_duration,
            },
        )
        
        '''
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print(f"New best accuracy: {best_accuracy * 100:.2f}%")
        '''
print("\nTraining completed!")
print("=" * 80)

# Retrieve and print the ranked list of snapshots
ranked_snapshots = ptsm.get_ids_by_rank()

print("\nTop 5 Snapshots by accuracy:")
print("-" * 80)
print(f"{'Snapshot ID':15} | {'Epoch':8} | {'Accuracy':12} | {'Loss':10} | {'Time/Epoch':10}")
print("-" * 80)
for snapshot_id in ranked_snapshots:
    metadata = ptsm.get_metadata(snapshot_id)
    print(
        f"{snapshot_id:15} | "
        f"{metadata['epoch']:8d} | "
        f"{metadata['accuracy'] * 100:10.2f}% | "
        f"{metadata['loss']:10.4f} | "
        f"{metadata['duration']:8.2f}s"
    )

# Plotting the training loss and test accuracy
plt.figure(figsize=(12, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(loss_history, 'b-', label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.legend()

# Plot test accuracy
plt.subplot(1, 2, 2)
plt.plot(accuracy_history, 'g-', label='Test Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Test Accuracy Over Time")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
