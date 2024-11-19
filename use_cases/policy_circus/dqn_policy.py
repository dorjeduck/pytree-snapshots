import torch

from .base_policy import BasePolicy


class DQNPolicy(BasePolicy):

    def __init__(self, id, policy_net, state_dict, device):
        """
        Initialize the DQNPolicy.

        Args:
            id (str): Policy identifier.
            policy_net (torch.nn.Module): The policy network.
            state_dict (dict): State dictionary for the policy network.
            device (torch.device): Device to load the model onto (e.g., 'cpu' or 'cuda').
        """
        super().__init__(id, policy_net)

        # Move the policy network to the specified device
        self.policy_net = self.policy_net.to(device)

        # Load the state dictionary
        self.policy_net.load_state_dict(state_dict)

        # Set the policy network to evaluation mode
        self.policy_net.eval()

        self.device = device

    def get_action(self, state_tensor):
        """
        Selects the action based on the maximum Q-value from the policy network.

        Args:
            state_tensor: Torch tensor representation of the current state.

        Returns:
            int: The index of the action with the maximum Q-value.
        """
        # Ensure the input state tensor is on the correct device
        state_tensor = state_tensor.to(self.device)

        # Compute Q-values without tracking gradients
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        # Return the index of the action with the maximum Q-value
        return q_values.max(1).indices.item()