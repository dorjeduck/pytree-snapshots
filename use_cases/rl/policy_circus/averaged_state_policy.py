from .base_policy import BasePolicy
import torch


class AveragedStatePolicy(BasePolicy):
    def __init__(self, id, policy_net, individual_results, device):
        """
        Initialize the AveragedStatePolicy.

        Args:
            id (str): Identifier for the policy.
            policy_net (torch.nn.Module): The neural network representing the policy.
            individual_results (list): List of results, each containing a policy object.
            device (torch.device): Device to load the model onto (e.g., 'cpu' or 'cuda').
        """
        super().__init__(id, policy_net)  # Initialize BasePolicy

        # Move the policy network to the specified device
        self.policy_net = self.policy_net.to(device)

        # Extract state_dicts from the individual results
        state_dicts = [psr.policy.policy_net.state_dict() for psr in individual_results]

        # Compute the averaged state dictionary
        average_state_dict = self._get_average_state_dict(state_dicts)

        # Load the averaged state dictionary into the policy network
        self.policy_net.load_state_dict(average_state_dict)

        # Set the policy network to evaluation mode
        self.policy_net.eval()

        self.device = device

    def _get_average_state_dict(self, state_dicts):
        """
        Compute the average state dictionary from multiple state dictionaries.

        Args:
            state_dicts (list): List of state dictionaries to average.

        Returns:
            dict: The averaged state dictionary.
        """
        averaged_state = {}

        # Iterate over keys and average the values
        for key in state_dicts[0].keys():
            values = [state[key] for state in state_dicts]
            averaged_state[key] = sum(values) / len(values)

        return averaged_state
