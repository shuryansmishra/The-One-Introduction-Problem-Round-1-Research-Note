"""Bayesian Thompson Sampling for online parameter estimation under selective labels."""
import random
import copy
from src.models.dyadic_model import DEFAULT_WEIGHTS

class ThompsonFeatureSampler:
    """Maintains Gaussian posteriors for soft feature weights to enable Bayesian exploration.
    
    Each soft attribute k has belief w_k ~ N(mu_k, sigma_k^2).
    In each round, sample w_k_tilde ~ N(mu_k, sigma_k^2).
    Updates posteriors from observed mutual feedback events with decay factor gamma.
    """
    def __init__(self, gamma: float = 0.98):
        self.gamma = gamma
        self.priors = {
            k: {'mu': v, 'var': 0.15**2} for k, v in DEFAULT_WEIGHTS.items()
        }
        self.beliefs = copy.deepcopy(self.priors)
        
    def sample_weights(self, rng: random.Random) -> dict:
        """Sample a plausible feature weight vector from the current posterior (Thompson Sampling)."""
        sampled = {}
        for k, dist in self.beliefs.items():
            std = dist['var']**0.5
            sampled[k] = rng.gauss(dist['mu'], std)
        return sampled

    def update(self, feature_matches: dict, outcome: bool):
        """Update Gaussian beliefs using a selective label observation (e.g. mutual acceptance).
        
        feature_matches: dict of feature_key -> bool (True if matched, False if mismatched)
        outcome: bool (True for positive mutual acceptance, False otherwise)
        """
        # Apply exponential decay to uncertainty to account for non-stationarity
        for k in self.beliefs:
            self.beliefs[k]['var'] = (1.0 / self.gamma) * self.beliefs[k]['var']
            
        target_shift = 0.10 if outcome else -0.10
        for k, matched in feature_matches.items():
            if k not in self.beliefs:
                continue
            delta = target_shift if matched else -target_shift
            # Update mean and reduce variance slightly
            self.beliefs[k]['mu'] += 0.05 * delta
            # Bound variance to avoid collapsing or exploding
            self.beliefs[k]['var'] = max(0.01**2, min(0.30**2, self.beliefs[k]['var'] * 0.95))

    def to_dict(self) -> dict:
        return {'beliefs': self.beliefs, 'gamma': self.gamma}

    @classmethod
    def from_dict(cls, data: dict):
        inst = cls(gamma=data.get('gamma', 0.98))
        if 'beliefs' in data:
            inst.beliefs = data['beliefs']
        return inst
