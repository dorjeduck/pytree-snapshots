import torch

from collections import namedtuple
from typing import List

# Define the shared namedtuple
PolicyEvaluationResult = namedtuple(
    "PolicyEvaluationResult", ["policy", "avg_reward", "variance"]
)


class PolicyEvaluator:
    def __init__(self, env, device):
        """
        Initializes the PolicyEvaluator with the environment and device.

        Args:
            env: The environment to evaluate the policy in.
            device: The device (CPU/GPU) for computations.
        """
        self.env = env
        self.device = device

    def eval(self, policy, num_episodes=5):
        """
        Evaluates a single policy over a specified number of episodes.

        Args:
            policy: A policy object implementing the `get_action` method.
            num_episodes (int): The number of episodes to evaluate the policy.

        Returns:
            tuple: (average_reward, variance) of rewards over the episodes.
        """
        total_rewards = []

        for _ in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0

            while True:
                state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                action = policy.get_action(state_tensor)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break

            total_rewards.append(episode_reward)

        average_reward = sum(total_rewards) / len(total_rewards)
        variance = sum(
            (reward - average_reward) ** 2 for reward in total_rewards
        ) / len(total_rewards)

        return PolicyEvaluationResult(policy, average_reward, variance)
