from .base_policy import BasePolicy
import torch


class WeightedStatePolicy(BasePolicy):
    def __init__(self, id, policy_net, weighted_state_dicts, device):
        super().__init__(id, policy_net)

        weighted_state_dict = self._get_average_weighted_state_dict(
            weighted_state_dicts, device
        )

        self.policy_net.load_state_dict(weighted_state_dict)

    def _get_average_weighted_state_dict(self, weighted_states, device):
        """
        Compute the weighted average of state dictionaries.

        Args:
            weighted_states (list): A list of tuples, where each tuple contains:
                - snapshot_id (str): The ID of the snapshot.
                - weight (float): The weight associated with the snapshot.

        Returns:
            dict: A new state dictionary with weighted averaged weights.
        """
        # Extract state dictionaries and weights from the weighted snapshots
        state_dicts, weights = zip(*weighted_states)

        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        # Initialize the averaged state dictionary
        averaged_state_dict = {}

        # Iterate over keys in the first state dictionary
        for key in state_dicts[0].keys():
            # Stack tensors with weights applied
            weighted_tensors = torch.stack(
                [
                    state_dict[key] * weight
                    for state_dict, weight in zip(state_dicts, weights)
                ]
            )
            # Compute the weighted average
            averaged_state_dict[key] = weighted_tensors.sum(dim=0) / weights.sum()

        return averaged_state_dict
