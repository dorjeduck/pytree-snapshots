import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from snapshot_manager import SnapshotManager

from policy_circus import (
    AveragedStatePolicy,
    WeightedStatePolicy,
    PolicyEvaluator,
    DQNPolicy,
    WeightedActionPolicy,
    WeightedQValuePolicy,
    compute_policy_weights,
)


env = gym.make("CartPole-v1")

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

MAX_SNAPSHOTS = 10


# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 300
else:
    num_episodes = 100


# Initialize SnapshotManager with a custom comparison function
def cmp_by_reward(snapshot1, snapshot2):
    return snapshot1.metadata["reward"] - snapshot2.metadata["reward"]


snapshot_manager = SnapshotManager(
    max_snapshots=MAX_SNAPSHOTS, cmp_function=cmp_by_reward
)

printed_dot_last = False

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()

            # Save snapshot of the policy network and metadata
            snapshot_id = snapshot_manager.save_snapshot(
                policy_net.state_dict(),
                snapshot_id=f"episode_{i_episode}",
                metadata={"episode": i_episode, "reward": t + 1},
            )
            if snapshot_id:
                if printed_dot_last:
                    print("")
                    printed_dot_last = False

                print(
                    f"Snapshot from Episode {i_episode} entered the top {MAX_SNAPSHOTS} with Reward {t + 1}",
                )

            else:
                print(".", end="", flush=True)
                printed_dot_last = True

            break


# Retrieve and print the top snapshots
ranked_snapshots = snapshot_manager.get_ranked_snapshots()
print(f"\n\nTop {MAX_SNAPSHOTS} Snapshots by Reward:")
for snapshot_id in ranked_snapshots:
    metadata = snapshot_manager.get_metadata(snapshot_id)
    print(f"Episode: {metadata['episode']}, Reward: {metadata['reward']}")

NUM_EVALUATION_EPISODES = 10
policy_evaluator = PolicyEvaluator(env, device)


# Evaluate individual policies using IndividualPolicyEvaluator
print("\bTesting Individual Policies...")

individual_results = []
for snapshot_id in ranked_snapshots:

    dqn_policy = DQNPolicy(
        snapshot_id,
        policy_net=policy_net,
        state_dict=snapshot_manager[snapshot_id],
        device=device,
    )

    # Evaluate the policy
    psr = policy_evaluator.eval(policy=dqn_policy, num_episodes=NUM_EVALUATION_EPISODES)

    # Retrieve metadata for better reporting
    metadata = snapshot_manager.get_metadata(snapshot_id)
    episode = metadata["episode"]
    training_reward = metadata["reward"]

    # Store the results
    individual_results.append(psr)

    # Print the results for this policy
    print(
        f"(Episode: {episode}) Training Reward: {training_reward}, "
        f"Avg Test Reward: {psr.avg_reward:.2f}, Variance: {psr.variance:.2f}"
    )

policy_weights = compute_policy_weights(individual_results, 0.5)

weighted_state_dicts = [
    (snapshot_manager[id], weight)
    for id, weight in zip(ranked_snapshots, policy_weights)
]

# Define evaluators and configurations
policy_classes = [
    (
        "Averaged Policy",
        AveragedStatePolicy,
        {"individual_results": individual_results},
    ),
    (
        "Weighted Policy",
        WeightedStatePolicy,
        {"weighted_state_dicts": weighted_state_dicts},
    ),
    (
        "Softmax Policy",
        WeightedActionPolicy,
        {"weighted_state_dicts": weighted_state_dicts, "n_actions": n_actions},
    ),
    (
        "QValue Policy",
        WeightedQValuePolicy,
        {"weighted_state_dicts": weighted_state_dicts, "n_actions": n_actions},
    ),
]

# Evaluate each policy
for label, policy_class, extra_args in policy_classes:
    print(f"\nTesting {label}...")
    policy = policy_class(
        id=label,
        policy_net=policy_net,
        device=device,
        **extra_args,  # Pass extra arguments specific to each evaluator
    )
    psr = policy_evaluator.eval(policy=policy, num_episodes=NUM_EVALUATION_EPISODES)

    print(f"Avg Test Reward: {psr.avg_reward:.2f}, Variance: {psr.variance:.2f}")

print("\nComplete")
plot_durations(show_result=True)
plt.ioff()
plt.show()
