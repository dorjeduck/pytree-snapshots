import torch
import torchvision
from torchvision import transforms
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

from snapshot_manager import PyTreeSnapshotManager

# Set parameters
BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
PRINT_EVERY = 10

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


# Convert PyTorch tensors to NumPy arrays to use with JAX
def pytorch_to_numpy(data_loader):
    images, labels = [], []
    for x_batch, y_batch in data_loader:
        images.append(x_batch.view(-1, 28 * 28).numpy())  # Flatten the images
        labels.append(y_batch.numpy())
    images = np.vstack(images)
    labels = np.concatenate(labels)
    return images, labels


train_images, train_labels = pytorch_to_numpy(train_loader)
train_labels = jax.nn.one_hot(train_labels, num_classes=10)
test_images, test_labels = pytorch_to_numpy(test_loader)
test_labels = jax.nn.one_hot(test_labels, num_classes=10)


# Initialize MLP parameters
def init_mlp_params(layer_widths):
    params = []
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        params.append(
            {
                "weights": np.random.normal(size=(n_in, n_out)) * np.sqrt(2 / n_in),
                "biases": np.ones(shape=(n_out,)),
            }
        )
    return params


params = init_mlp_params([784, 256, 128, 10])


# Define the forward pass (using softmax for classification)
def forward(params, x):
    *hidden, last = params
    for layer in hidden:
        x = jax.nn.relu(x @ layer["weights"] + layer["biases"])
    return jax.nn.softmax(x @ last["weights"] + last["biases"], axis=-1)


# Define the cross-entropy loss function
@jax.jit
def loss_fn(params, x, y):
    predictions = forward(params, x)
    return -jnp.mean(jnp.sum(y * jnp.log(predictions + 1e-10), axis=1))


# Update function using JAX's jit and grad
@jax.jit
def update(params, x, y, lr=LEARNING_RATE):
    grads = jax.grad(loss_fn)(params, x, y)
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)


# Define accuracy calculation
def compute_accuracy(params, x, y):
    predictions = forward(params, x)
    predicted_classes = jnp.argmax(predictions, axis=1)
    true_classes = jnp.argmax(y, axis=1)
    return jnp.mean(predicted_classes == true_classes)


# Training loop
loss_history = []
accuracy_history = []


# Initialize PyTreeSnapshotManager with max_snapshots and custom comparison function
def cmp_by_accuracy(snapshot1, snapshot2):
    return snapshot1.metadata["accuracy"] - snapshot2.metadata["accuracy"]


ptsm = PyTreeSnapshotManager(
    deepcopy_on_retrieve=False, max_snapshots=5, cmp=cmp_by_accuracy
)


for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()  # Record start time for the epoch

    # Training step
    for i in range(0, len(train_images), BATCH_SIZE):
        x_batch = train_images[i : i + BATCH_SIZE]
        y_batch = train_labels[i : i + BATCH_SIZE]
        params = update(params, x_batch, y_batch)

    # Calculate epoch duration in seconds
    epoch_duration = time.time() - epoch_start_time

    # Evaluate loss on the full training set for monitoring
    current_loss = loss_fn(params, train_images, train_labels)
    loss_history.append(current_loss)

    # Evaluate accuracy on the test set
    test_accuracy = compute_accuracy(params, test_images, test_labels)
    accuracy_history.append(test_accuracy)

    # Print progress every PRINT_EVERY epochs
    if (epoch + 1) % PRINT_EVERY == 0:
        print(
            f"Epoch {epoch + 1}, Loss: {current_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%, Time per epoch: {epoch_duration:.2f} sec/epoch"
        )

    # Save snapshot
    ptsm.save_snapshot(
        params,
        snapshot_id=f"epoch_{epoch}",
        metadata={
            "epoch": epoch,
            "loss": current_loss,
            "accuracy": float(test_accuracy),
        },
    )

# Retrieve and print the ranked list of snapshots
ranked_snapshots = ptsm.get_ids_by_rank()

print("\nFive Snapshots with the highest accuracy:")
for snapshot_id in ranked_snapshots:
    metadata = ptsm.get_metadata(snapshot_id)
    print(
        f"Snapshot ID: {snapshot_id}, Epoch: {metadata['epoch']}, Accuracy: {metadata['accuracy'] * 100:.2f}%, Loss: {metadata['loss']:.4f}"
    )

# Plotting the training loss and test accuracy

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.subplot(1, 2, 2)
plt.plot(accuracy_history)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Test Accuracy")

plt.show()
