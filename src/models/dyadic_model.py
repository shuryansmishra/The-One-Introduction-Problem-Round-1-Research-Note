"""Dyadic reciprocal compatibility and multi-stage outcome prediction model.

Preserves the distinction between directional probabilities P(A -> B) and P(B -> A)
and models dyadic covariance in mutual acceptance:
P(Mutual Yes) > P(A -> B) * P(B -> A) due to latent chemistry.
"""
import math
from kit import SOFT

# Prior weights over soft feature matches, reflecting domain compatibility
DEFAULT_WEIGHTS = {
    'relationship_goal': 0.65,
    'relationship_pace': 0.40,
    'lifestyle': 0.25,
    'conversations': 0.20,
    'emotional_availability': 0.15,
    'space_for_relationship': 0.10,
    'relocate': 0.05,
}

def sigmoid(z: float) -> float:
    if z > 15:
        return 1.0
    if z < -15:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))

def dyadic_soft_fit(fa: dict, fb: dict, weights: dict = None) -> float:
    """Calculate the reciprocal soft alignment score between two feature vectors."""
    w = weights or DEFAULT_WEIGHTS
    score = 0.0
    for k, weight in w.items():
        va = fa.get(k)
        vb = fb.get(k)
        if va is not None and vb is not None:
            if va == vb:
                score += weight * 1.0
            else:
                score += weight * (-0.5)
        # If either is unknown, expected contribution is 0 (unbiased under missingness)
    return score

def estimate_directional_acceptance(actor: dict, other: dict, weights: dict = None, dyadic_sigma: float = 0.45, drift_offset: float = 0.0) -> float:
    """Estimate marginal directional acceptance probability P(actor -> other),
    integrating over latent dyadic chemistry xi ~ N(0, sigma^2).
    """
    fa, fb = actor.get('fields', {}), other.get('fields', {})
    fit = dyadic_soft_fit(fa, fb, weights)
    points = [(-2.0, 0.05), (-1.0, 0.24), (0.0, 0.42), (1.0, 0.24), (2.0, 0.05)]
    p_marginal = 0.0
    for std_node, weight in points:
        z = -0.25 + fit + (std_node * dyadic_sigma) + drift_offset
        p_marginal += weight * sigmoid(z)
    return p_marginal

def estimate_mutual_acceptance(a: dict, b: dict, weights: dict = None, dyadic_sigma: float = 0.45, drift_offset: float = 0.0) -> float:
    """Estimate joint mutual acceptance probability P(Mutual Yes).
    
    Uses numerical Gauss-Hermite integration over latent shared chemistry xi ~ N(0, sigma^2):
    P(A accepts B AND B accepts A) = E_xi [ P(A accepts | xi) * P(B accepts | xi) ]
    """
    fa, fb = a.get('fields', {}), b.get('fields', {})
    fit_a = dyadic_soft_fit(fa, fb, weights)
    fit_b = dyadic_soft_fit(fb, fa, weights)
    
    # 5-point Gauss-Hermite quadrature nodes and weights for N(0, dyadic_sigma^2)
    # nodes x_k and standard normal weights
    points = [
        (-2.0, 0.05),
        (-1.0, 0.24),
        (0.0, 0.42),
        (1.0, 0.24),
        (2.0, 0.05)
    ]
    
    p_mutual = 0.0
    for std_node, weight in points:
        xi = std_node * dyadic_sigma
        za = -0.25 + fit_a + xi + drift_offset
        zb = -0.25 + fit_b + xi + drift_offset
        p_mutual += weight * (sigmoid(za) * sigmoid(zb))
        
    return p_mutual

def estimate_expected_msmi(a: dict, b: dict, weights: dict = None, drift_offset: float = 0.0) -> float:
    """Compute end-to-end Expected MSMI Utility E[U_ij] across the full funnel:
    Introduction -> Mutual Acceptance -> First Date (within 30d) -> Mutual 2nd Meeting (within 3d).
    """
    # 1. Per-member response probabilities from simulator field (range 0.55–0.98).
    # Falls back to the empirical average 0.76 if not present.
    p_resp_a = a.get('response_rate', 0.76)
    p_resp_b = b.get('response_rate', 0.76)
    p_both_respond = p_resp_a * p_resp_b
    
    # 2. Mutual acceptance
    p_mutual_accept = estimate_mutual_acceptance(a, b, weights=weights, drift_offset=drift_offset)
    
    # 3. Date happening within 30 days given mutual acceptance (~78% in simulator)
    p_date_given_accept = 0.75
    
    # 4. Mutual second-meeting intention given date (~25-35% dependent on relationship goal match)
    fa, fb = a.get('fields', {}), b.get('fields', {})
    goal_match = (fa.get('relationship_goal') is not None and 
                  fa.get('relationship_goal') == fb.get('relationship_goal'))
    base_second_logit = 0.15 + (0.4 if goal_match else 0.0)
    p_second_given_date = sigmoid(base_second_logit) * sigmoid(base_second_logit) * 0.90 # on-time factor
    
    expected_msmi = p_both_respond * p_mutual_accept * p_date_given_accept * p_second_given_date
    return expected_msmi
