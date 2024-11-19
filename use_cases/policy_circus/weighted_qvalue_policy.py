from .base_policy import BasePolicy
import torch


class WeightedQValuePolicy(BasePolicy):
    def __init__(self, id, policy_net, weighted_state_dicts, n_actions, device):
        super().__init__(id, policy_net)

        self.device = device
        self.n_actions = n_actions
        self.weighted_state_dicts = weighted_state_dicts

        # Pre-load models into memory
        self.models = []
        for state_dict, weight in self.weighted_state_dicts:
            model = self._load_model(state_dict)
            self.models.append((model, weight))

    def _load_model(self, state_dict):
        """
        Load a policy network with the given state dictionary.

        Args:
            state_dict (dict): State dictionary of the policy network.

        Returns:
            torch.nn.Module: Policy network with loaded weights.
        """
        model = self.policy_net.to(self.device)  # Ensure the model is on the device
        model.load_state_dict(state_dict)
        model.eval()  # Set the model to evaluation mode
        return model

    def get_action(self, state_tensor):
        """
        Compute the action to take based on the current state tensor.

        Args:
            state_tensor (torch.Tensor): Input state tensor.

        Returns:
            int: Index of the selected action.
        """

        # Compute Q-values
        q_values = torch.zeros(self.n_actions, device=self.device)
        for state_dict, weight in self.weighted_state_dicts:
            self.policy_net.load_state_dict(state_dict)
            q_values += self.policy_net(state_tensor).squeeze(0) * weight

        # Select the action with the highest Q-value
        return torch.argmax(q_values).item()
