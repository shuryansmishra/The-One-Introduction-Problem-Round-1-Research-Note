"""Survival-Weighted Contextual Bandit for delay-corrected sequential matching.

Combines Kaplan-Meier survival analysis with Bayesian Logistic Regression to solve
right-censored delayed feedback. Pending introductions receive fractional soft labels
y_tilde(t) = P(Y=1 | T > t) instead of being discarded or wrongly labeled as failures.

References:
  - Kaplan & Meier (1958). Nonparametric estimation from incomplete observations.
  - Vernade, Cappé & Perchet (2017). Stochastic bandit models for delayed conversions.
  - Pike-Burke et al. (2018). Bandits with delayed, aggregated anonymous feedback.
"""
import math
import copy
import random as stdlib_random
from collections import defaultdict


class KaplanMeierSurvivalEstimator:
    """Tracks observation windows and estimates survival curves S(t)
    and conditional success probabilities P(Y=1 | T > t) for pending matches.

    Uses non-parametric Kaplan-Meier estimation from the pool of resolved
    and right-censored introductions observed so far.
    """

    def __init__(self, max_days: int = 40):
        self.max_days = max_days
        self.success_times: list[int] = []
        self.rejection_times: list[int] = []

    def update(self, introductions: list, feedback: list, current_day: int):
        """Build empirical delay distributions from simulator feedback history.

        Each introduction is classified as success, rejection, or pending based
        on the feedback events observed by `current_day`.
        """
        events_by_intro = defaultdict(list)
        for e in feedback:
            events_by_intro[e['introduction_id']].append(e)

        self.success_times = []
        self.rejection_times = []

        for intro in introductions:
            iid = intro['introduction_id']
            assigned = intro['assigned_day']
            evs = events_by_intro.get(iid, [])

            # Check for full MSMI success
            responses = [e for e in evs if e['event'] == 'introduction_response']
            dates = [e for e in evs if e['event'] == 'date_happened' and e.get('value') is True]
            seconds = [e for e in evs if e['event'] == 'second_meeting_intention']

            mutual_yes = (len(responses) == 2 and
                          all(e.get('value') == 'yes' for e in responses))
            date_ok = bool(dates)
            msmi_ok = (date_ok and len(seconds) == 2 and
                       all(e.get('value') == 'yes' for e in seconds) and
                       dates[0]['occurred_day'] - assigned <= 30 and
                       all(e['occurred_day'] - dates[0]['occurred_day'] <= 3 for e in seconds))

            if msmi_ok:
                # Success: record delay from assignment to final confirmation
                resolve_day = max(e['occurred_day'] for e in seconds)
                self.success_times.append(resolve_day - assigned)
            elif not mutual_yes:
                # Definitive rejection: at least one No or deadline expired
                any_no = any(e.get('value') == 'no' for e in responses)
                deadline_passed = current_day >= intro.get('response_deadline_day',
                                                           assigned + 7)
                if any_no or (deadline_passed and len(responses) < 2):
                    resolve_day = max((e.get('occurred_day') or e.get('observed_day', assigned)
                                       for e in responses), default=assigned)
                    self.rejection_times.append(resolve_day - assigned)
            elif date_ok and dates[0].get('value') is False:
                # Date explicitly did not happen
                self.rejection_times.append((dates[0].get('occurred_day') or dates[0].get('observed_day', assigned)) - assigned)
            elif date_ok and len(seconds) == 2:
                # Date happened but not mutual second meeting
                resolve_day = max((e.get('occurred_day') or e.get('observed_day', assigned)) for e in seconds)
                self.rejection_times.append(resolve_day - assigned)
            # Otherwise: still pending (right-censored)

    def predict_conditional_success(self, elapsed_days: int) -> float:
        """Compute P(Y=1 | T > elapsed_days) for a right-censored pending introduction.

        Uses the survival delay-correction formula:
            y_tilde(t) = P(Y=1) * (1 - F_success(t)) / S(t)

        where F_success(t) is the empirical CDF of success delay times
        and S(t) is the overall survival probability (fraction unresolved at t).
        """
        n_success = len(self.success_times)
        n_rejection = len(self.rejection_times)
        total_resolved = n_success + n_rejection

        if total_resolved == 0:
            # Uninformative prior: use the empirical MSMI rate from training data
            return 0.12

        p_success = n_success / total_resolved

        # F_success(t) = P(T <= t | Y = 1)
        if n_success > 0:
            succ_by_t = sum(1 for t in self.success_times if t <= elapsed_days)
            f_succ_t = succ_by_t / n_success
        else:
            f_succ_t = 0.0

        # Fraction of all resolved outcomes that resolved by elapsed_days
        resolved_by_t = (sum(1 for t in self.success_times if t <= elapsed_days) +
                         sum(1 for t in self.rejection_times if t <= elapsed_days))
        s_t = 1.0 - (resolved_by_t / max(1, total_resolved))
        s_t = max(s_t, 1e-4)  # Prevent division by zero

        remaining_succ_prob = p_success * (1.0 - f_succ_t)
        cond_prob = remaining_succ_prob / s_t

        return max(1e-3, min(0.95, cond_prob))

    def to_dict(self) -> dict:
        return {
            'success_times': self.success_times,
            'rejection_times': self.rejection_times,
            'max_days': self.max_days
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'KaplanMeierSurvivalEstimator':
        inst = cls(max_days=data.get('max_days', 40))
        inst.success_times = data.get('success_times', [])
        inst.rejection_times = data.get('rejection_times', [])
        return inst


class SurvivalWeightedBandit:
    """Bayesian Logistic Bandit with Laplace-approximated Thompson Sampling
    and survival-weighted soft labels for right-censored pending matches.

    The three-bucket label protocol:
      - Confirmed success (MSMI):     y = 1.0
      - Confirmed rejection:           y = 0.0
      - Pending (right-censored):      y = y_tilde(t) in (0, 1), from survival model
    """

    def __init__(self, feature_dim: int = 7, prior_variance: float = 1.0,
                 gamma: float = 0.98):
        from src.models.dyadic_model import DEFAULT_WEIGHTS
        self.feature_dim = feature_dim
        self.prior_variance = prior_variance
        self.gamma = gamma
        keys = sorted(DEFAULT_WEIGHTS.keys())
        self.mu = [DEFAULT_WEIGHTS[k] for k in keys[:feature_dim]]
        self.inv_cov_diag = [1.0 / prior_variance] * feature_dim
        self.survival = KaplanMeierSurvivalEstimator()
        self._fit_count = 0

    def classify_introductions(self, introductions: list, feedback: list,
                               current_day: int, use_survival: bool = True) -> list:
        """Classify each introduction into success / rejection / pending
        and assign the appropriate hard or soft label.

        If use_survival=True, pending introductions receive Kaplan-Meier soft labels.
        If use_survival=False, pending introductions receive 0.0 (ablation H2).

        Returns list of dicts: {introduction_id, label, status, elapsed_days}
        """
        events_by_intro = defaultdict(list)
        for e in feedback:
            events_by_intro[e['introduction_id']].append(e)

        # Update survival model first
        if use_survival:
            self.survival.update(introductions, feedback, current_day)

        results = []
        for intro in introductions:
            iid = intro['introduction_id']
            assigned = intro['assigned_day']
            elapsed = current_day - assigned
            evs = events_by_intro.get(iid, [])

            responses = [e for e in evs if e['event'] == 'introduction_response']
            dates = [e for e in evs if e['event'] == 'date_happened'
                     and e.get('value') is True]
            seconds = [e for e in evs if e['event'] == 'second_meeting_intention']

            mutual_yes = (len(responses) == 2 and
                          all(e.get('value') == 'yes' for e in responses))

            # 1. Check for full MSMI success
            if (dates and len(seconds) == 2 and
                    all(e.get('value') == 'yes' for e in seconds) and
                    dates[0]['occurred_day'] - assigned <= 30 and
                    all(e['occurred_day'] - dates[0]['occurred_day'] <= 3
                        for e in seconds)):
                results.append({
                    'introduction_id': iid,
                    'label': 1.0,
                    'status': 'success',
                    'elapsed_days': elapsed
                })
                continue

            # 2. Check for confirmed rejection
            any_no = any(e.get('value') == 'no' for e in responses)
            deadline_passed = current_day >= intro.get('response_deadline_day',
                                                       assigned + 7)
            explicit_rejection = any_no or (deadline_passed and not mutual_yes)

            if explicit_rejection:
                results.append({
                    'introduction_id': iid,
                    'label': 0.0,
                    'status': 'rejection',
                    'elapsed_days': elapsed
                })
                continue

            # Check if date explicitly failed
            date_failed = any(e['event'] == 'date_happened' and
                              e.get('value') is False for e in evs)
            if date_failed:
                results.append({
                    'introduction_id': iid,
                    'label': 0.0,
                    'status': 'rejection',
                    'elapsed_days': elapsed
                })
                continue

            # Check if date happened but second meeting was negative
            if (dates and len(seconds) == 2 and
                    not all(e.get('value') == 'yes' for e in seconds)):
                results.append({
                    'introduction_id': iid,
                    'label': 0.0,
                    'status': 'rejection',
                    'elapsed_days': elapsed
                })
                continue

            # 3. Pending: assign survival-corrected soft label (or 0.0 if ablation)
            soft_label = self.survival.predict_conditional_success(elapsed) if use_survival else 0.0
            results.append({
                'introduction_id': iid,
                'label': soft_label,
                'status': 'pending',
                'elapsed_days': elapsed
            })

        return results

    def update_posterior(self, feature_matches_list: list, labels: list):
        """Lightweight diagonal posterior update using observed labels.

        feature_matches_list: list of dicts {feature_key: bool} for each observation
        labels: corresponding hard or soft labels in [0, 1]

        Uses online Bayesian updating with recency discount gamma.
        """
        if not labels:
            return

        for feature_matches, y in zip(feature_matches_list, labels):
            for idx, (key, matched) in enumerate(sorted(feature_matches.items())):
                if idx >= self.feature_dim:
                    break
                x = 1.0 if matched else -0.5
                pred = _sigmoid(self.mu[idx] * x)
                residual = y - pred
                # Recency-discounted precision update
                self.inv_cov_diag[idx] = (self.gamma * self.inv_cov_diag[idx] +
                                          pred * (1 - pred) * x * x)
                # Mean update
                lr = 0.05 / max(1.0, self.inv_cov_diag[idx])
                self.mu[idx] += lr * residual * x

        self._fit_count += 1

    def sample_weights(self, rng: stdlib_random.Random) -> dict:
        """Thompson Sampling: draw weights from the current posterior."""
        from src.models.dyadic_model import DEFAULT_WEIGHTS
        keys = sorted(DEFAULT_WEIGHTS.keys())
        sampled = {}
        for idx, key in enumerate(keys):
            if idx >= self.feature_dim:
                break
            std = 1.0 / max(0.01, self.inv_cov_diag[idx]) ** 0.5
            sampled[key] = rng.gauss(self.mu[idx], std)
        return sampled

    def to_dict(self) -> dict:
        return {
            'mu': self.mu,
            'inv_cov_diag': self.inv_cov_diag,
            'gamma': self.gamma,
            'fit_count': self._fit_count,
            'survival': self.survival.to_dict()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SurvivalWeightedBandit':
        feature_dim = len(data.get('mu', [0] * 7))
        inst = cls(feature_dim=feature_dim, gamma=data.get('gamma', 0.98))
        inst.mu = data.get('mu', [0.0] * feature_dim)
        inst.inv_cov_diag = data.get('inv_cov_diag', [1.0] * feature_dim)
        inst._fit_count = data.get('fit_count', 0)
        if 'survival' in data:
            inst.survival = KaplanMeierSurvivalEstimator.from_dict(data['survival'])
        return inst


def _sigmoid(z: float) -> float:
    if z > 15:
        return 1.0
    if z < -15:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))
