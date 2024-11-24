import torch

from abc import ABC, abstractmethod

from abc import ABC, abstractmethod
import torch


class BaseEnsembleEvaluator(ABC):
    """
    Abstract base class for ensemble evaluation strategies.
    """

    def __init__(self, device, policy_net, env, ranked_snapshots, manager):
        self.device = device
        self.policy_net = policy_net
        self.env = env
        self.ranked_snapshots = ranked_snapshots
        self.manager = manager
        self.n_actions = env.action_space.n

    def evaluate_policy(self, state_dict, num_episodes=5):
        """
        Evaluate a single policy over a number of episodes.

        Args:
            state_dict: State dictionary of the policy to evaluate.
            num_episodes: Number of episodes to run for evaluation.

        Returns:
            float: Average reward across episodes.
        """
        self.policy_net.load_state_dict(state_dict)
        total_rewards = []

        for _ in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            while True:
                state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                action = self.policy_net(state_tensor).max(1).indices.item()
                state, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
                if done or truncated:
                    break
            total_rewards.append(episode_reward)

        return sum(total_rewards) / len(total_rewards)

    @abstractmethod
    def evaluate(self, num_episodes=10, num_snapshots=None):
        """
        Evaluate the ensemble using the defined strategy.

        Args:
            num_episodes (int): Number of episodes to evaluate.
            num_snapshots (int, optional): Number of top snapshots to include in the ensemble.

        Returns:
            float: Average test reward of the ensemble.
        """
        pass


class SoftmaxEnsembleEvaluator(BaseEnsembleEvaluator):
    def evaluate(self, num_episodes=10, num_snapshots=None):
        # Use specified number of snapshots or all ranked snapshots
        snapshots_to_use = (
            self.ranked_snapshots[:num_snapshots]
            if num_snapshots
            else self.ranked_snapshots
        )

        # Collect state dictionaries and test rewards
        snapshot_data = [
            (
                self.manager[snapshot_id],
                self.evaluate_policy(self.manager[snapshot_id], num_episodes=5),
            )
            for snapshot_id in snapshots_to_use
        ]
        total_reward = sum(reward for _, reward in snapshot_data)
        snapshot_data = [
            (state_dict, reward / total_reward) for state_dict, reward in snapshot_data
        ]

        total_rewards = []
        for _ in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            while True:
                state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                action_probs = torch.zeros(self.n_actions, device=self.device)
                for state_dict, weight in snapshot_data:
                    self.policy_net.load_state_dict(state_dict)
                    probs = torch.softmax(self.policy_net(state_tensor), dim=1).squeeze(
                        0
                    )
                    action_probs += probs * weight
                action = torch.argmax(action_probs).item()
                state, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
                if done or truncated:
                    break
            total_rewards.append(episode_reward)

        return sum(total_rewards) / len(total_rewards)


class QValueEnsembleEvaluator(BaseEnsembleEvaluator):
    def evaluate(self, num_episodes=10, num_snapshots=None):
        # Use specified number of snapshots or all ranked snapshots
        snapshots_to_use = (
            self.ranked_snapshots[:num_snapshots]
            if num_snapshots
            else self.ranked_snapshots
        )

        # Collect state dictionaries and test rewards
        snapshot_data = [
            (
                self.manager[snapshot_id],
                self.evaluate_policy(self.manager[snapshot_id], num_episodes=5),
            )
            for snapshot_id in snapshots_to_use
        ]
        total_reward = sum(reward for _, reward in snapshot_data)
        snapshot_data = [
            (state_dict, reward / total_reward) for state_dict, reward in snapshot_data
        ]

        total_rewards = []
        for _ in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            while True:
                state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                q_values = torch.zeros(self.n_actions, device=self.device)
                for state_dict, weight in snapshot_data:
                    self.policy_net.load_state_dict(state_dict)
                    q_values += self.policy_net(state_tensor).squeeze(0) * weight
                action = torch.argmax(q_values).item()
                state, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
                if done or truncated:
                    break
            total_rewards.append(episode_reward)

        return sum(total_rewards) / len(total_rewards)
