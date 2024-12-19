import gymnasium as gym
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from collections import deque
from typing import Optional, Tuple, List
import random
import matplotlib.pyplot as plt
from itertools import count

from snapshot_manager import SnapshotManager
from policy_circus import PolicyEvaluator, DQNPolicy

# Constants
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
MAX_SNAPSHOTS = 10

class DQN(eqx.Module):
    layers: list

    def __init__(self, n_observations: int, n_actions: int, key):
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(n_observations, 128, key=keys[0]),
            jax.nn.relu,
            eqx.nn.Linear(128, 128, key=keys[1]),
            jax.nn.relu,
            eqx.nn.Linear(128, n_actions, key=keys[2])
        ]

    def __call__(self, x):
        x = jnp.array(x, dtype=jnp.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        for layer in self.layers:
            x = layer(x)
        return x

def make_network(n_observations: int, n_actions: int, key) -> DQN:
    """Factory function to create DQN with proper initialization."""
    return DQN(n_observations, n_actions, key)

# Initialize optimizer globally
optimizer = optax.adamw(learning_rate=LR)

class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))
    
    def sample(self, batch_size: int, key) -> Tuple[jnp.ndarray, ...]:
        batch = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards = zip(*batch)
        
        # Convert to JAX arrays with proper dtypes
        states = jnp.stack([s.astype(jnp.float32) for s in states])
        actions = jnp.array(actions, dtype=jnp.int32)
        next_states = jnp.stack([s.astype(jnp.float32) if s is not None else jnp.zeros_like(states[0]) 
                                for s in next_states])
        rewards = jnp.array(rewards, dtype=jnp.float32)
        
        return states, actions, next_states, rewards
    
    def __len__(self):
        return len(self.memory)

@jax.jit
def select_action(model: DQN, state: jnp.ndarray, eps_threshold: float, key) -> int:
    def exploit():
        q_values = model(state)
        return jnp.argmax(q_values).astype(jnp.int32)
    
    def explore():
        return jax.random.randint(key, (), 0, 2, dtype=jnp.int32)
    
    return jax.lax.cond(
        jax.random.uniform(key) > eps_threshold,
        exploit,
        explore
    )

@jax.jit
def compute_loss(model: DQN, target_model: DQN, batch):
    states, actions, next_states, rewards = batch
    
    # Current Q-values
    q_values = jax.vmap(model)(states)
    state_action_values = q_values[jnp.arange(len(actions)), actions]
    
    # Target Q-values
    next_q_values = jax.vmap(target_model)(next_states)
    next_state_values = jnp.max(next_q_values, axis=1)
    expected_state_action_values = (rewards + GAMMA * next_state_values).astype(jnp.float32)
    
    # Huber loss
    diff = state_action_values - expected_state_action_values
    return jnp.mean(jnp.where(jnp.abs(diff) < 1, 
                             0.5 * diff ** 2,
                             jnp.abs(diff) - 0.5))

@jax.jit
def update_step(model: DQN, target_model: DQN, opt_state, batch):
    params = eqx.filter(model, eqx.is_array)
    loss, grads = jax.value_and_grad(compute_loss)(model, target_model, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state, params=params)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss.astype(jnp.float32)

def soft_update(target_model: DQN, model: DQN, tau: float) -> DQN:
    """Soft update target network parameters"""
    target_params = eqx.filter(target_model, eqx.is_array)
    model_params = eqx.filter(model, eqx.is_array)
    
    new_params = jax.tree_util.tree_map(
        lambda target, source: (1 - tau) * target + tau * source,
        target_params, model_params
    )
    
    return eqx.combine(new_params, target_model)

def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = jnp.array(episode_durations, dtype=jnp.float32)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t)
    
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = jnp.array([jnp.mean(durations_t[max(0, i-100):i]) 
                          for i in range(1, len(durations_t)+1)], dtype=jnp.float32)
        means = jnp.concatenate([jnp.zeros(99, dtype=jnp.float32), means])
        plt.plot(means)

    plt.pause(0.001)

def main():
    # Initialize environment
    env = gym.make("CartPole-v1")
    
    # Initialize networks
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    model = make_network(n_observations, n_actions, subkey)
    key, subkey = jax.random.split(key)
    target_model = make_network(n_observations, n_actions, subkey)
    
    # Initialize optimizer state
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Initialize replay memory
    memory = ReplayMemory(10000)
    
    # Initialize SnapshotManager
    def cmp_by_reward(snapshot1, snapshot2):
        return snapshot1.metadata["reward"] - snapshot2.metadata["reward"]
    
    snapshot_manager = SnapshotManager(max_snapshots=MAX_SNAPSHOTS, cmp=cmp_by_reward)
    
    # Training loop
    num_episodes = 300 if jax.default_backend() != "cpu" else 50
    episode_durations = []
    steps_done = 0
    printed_dot_last = False
    
    for i_episode in range(num_episodes):
        # Initialize the environment
        state, _ = env.reset()
        state = jnp.array(state, dtype=jnp.float32)
        
        episode_reward = 0
        for t in range(1000):  # Max steps per episode
            # Select action
            key, subkey = jax.random.split(key)
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                          jnp.exp(-1. * steps_done / EPS_DECAY)
            
            action = select_action(model, state, eps_threshold, subkey)
            steps_done += 1
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            
            if terminated:
                next_state = None
            
            # Store transition
            memory.push(state, action, 
                       None if next_state is None else jnp.array(next_state, dtype=jnp.float32), 
                       reward)
            
            # Move to next state
            if not done:
                state = jnp.array(next_state, dtype=jnp.float32)
            
            # Optimize model
            if len(memory) >= BATCH_SIZE:
                key, subkey = jax.random.split(key)
                batch = memory.sample(BATCH_SIZE, subkey)
                model, opt_state, loss = update_step(model, target_model, opt_state, batch)
                
                # Soft update target network
                target_model = soft_update(target_model, model, TAU)
            
            episode_reward += reward
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                
                # Save snapshot
                snapshot_id = snapshot_manager.save_snapshot(
                    eqx.filter(model, eqx.is_array),
                    snapshot_id=f"episode_{i_episode}",
                    metadata={"episode": i_episode, "reward": t + 1}
                )
                
                if snapshot_id:
                    if printed_dot_last:
                        print("")
                        printed_dot_last = False
                    print(f"Snapshot from Episode {i_episode} entered top {MAX_SNAPSHOTS} with Reward {t + 1}")
                else:
                    print(".", end="", flush=True)
                    printed_dot_last = True
                break
    
    print("\nComplete")
    plot_durations(episode_durations, show_result=True)
    plt.ioff()
    plt.show()
    
    # Evaluate final policies
    print("\nTop snapshots by reward:")
    ranked_snapshots = snapshot_manager.get_ids_by_rank()
    for snapshot_id in ranked_snapshots:
        metadata = snapshot_manager.get_metadata(snapshot_id)
        print(f"Episode: {metadata['episode']}, Reward: {metadata['reward']}")

if __name__ == "__main__":
    main()
