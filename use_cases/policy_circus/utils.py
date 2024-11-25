from typing import List, Dict, Tuple
from .policy_evaluator import PolicyEvaluationResult


def compute_policy_weights(
    individual_results: List[PolicyEvaluationResult],
    alpha: float,
) -> List[Tuple[dict, float]]:
    """
    Calculate weighted states based on individual evaluation results.

    Args:
        state_dict_map (Dict[str, dict]): A mapping of policy IDs to state dictionaries.
        individual_results (List[PolicyEvaluationResult]): A list of policy evaluation results.
        alpha (float): A weighting factor.

    Returns:
        List[Tuple[dict, float]]: A list of tuples (state_dict, normalized_weight).
    """
    # Extract reward and variance for each policy
    policy_ids = [psr.policy.id for psr in individual_results]
    reward_map = {psr.policy.id: psr.avg_reward for psr in individual_results}
    variance_map = {psr.policy.id: psr.variance for psr in individual_results}

    # Prepare policy data with reward and variance
    policy_data = [
        (
            reward_map[policy_id],
            variance_map[policy_id],
        )
        for policy_id in policy_ids
    ]

    # Determine max values for normalization
    max_reward, max_variance = map(max, zip(*policy_data))

    # Calculate weights using alpha to balance reward and variance
    weighted_data = [
        (
            (reward / max_reward) ** alpha  # Reward contribution
            * (1 - (variance / (max_variance + 1e-8)))
            ** (1 - alpha)  # Variance penalty
        )
        for reward, variance in policy_data
    ]

    # Normalize weights so they sum to 1
    total_weight = sum(weighted_data)

    if total_weight == 0:
        raise ValueError("Total weight for policies is zero. Cannot normalize.")

    return [weight / total_weight for weight in weighted_data]
