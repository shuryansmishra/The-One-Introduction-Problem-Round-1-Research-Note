"""Unit tests for newly created research modules in src/."""
import unittest
from src.models.dyadic_model import (
    dyadic_soft_fit,
    estimate_directional_acceptance,
    estimate_mutual_acceptance,
    estimate_expected_msmi
)
from src.models.thompson_sampler import ThompsonFeatureSampler
from src.decision.waiting_policy import get_reservation_threshold
from src.decision.voi_ask import select_clarification_asks
from src.decision.allocator import allocate_pairs

class TestResearchModules(unittest.TestCase):
    def setUp(self):
        self.member_a = {
            'member_id': 'syn_001',
            'pool_id': 'dev_01',
            'age': 28,
            'gender': 'woman',
            'zone': 'zone_a',
            'available': True,
            'arrived_day': 0,
            'fields': {
                'age_min': 25, 'age_max': 35, 'who_to_meet': ['man'],
                'relationship_structure': 'monogamous', 'smoking': 'no',
                'partner_smoking': 'no_smoking', 'has_children': False,
                'partner_children': 'no_children', 'wants_children': 'yes',
                'acceptable_zones': ['zone_a'],
                'schedule': ['weekend_day'],
                'relationship_goal': 'long_term', 'relationship_pace': 'steady',
                'lifestyle': 'social', 'conversations': 'ideas'
            },
            'field_status': {k: 'observed' for k in ['relationship_goal', 'relationship_pace', 'lifestyle', 'conversations']}
        }
        self.member_b = {
            'member_id': 'syn_002',
            'pool_id': 'dev_01',
            'age': 30,
            'gender': 'man',
            'zone': 'zone_a',
            'available': True,
            'arrived_day': 0,
            'fields': {
                'age_min': 25, 'age_max': 35, 'who_to_meet': ['woman'],
                'relationship_structure': 'monogamous', 'smoking': 'no',
                'partner_smoking': 'no_smoking', 'has_children': False,
                'partner_children': 'no_children', 'wants_children': 'yes',
                'acceptable_zones': ['zone_a'],
                'schedule': ['weekend_day'],
                'relationship_goal': 'long_term', 'relationship_pace': 'steady',
                'lifestyle': 'social', 'conversations': 'ideas'
            },
            'field_status': {k: 'observed' for k in ['relationship_goal', 'relationship_pace', 'lifestyle', 'conversations']}
        }

    def test_dyadic_models(self):
        fit = dyadic_soft_fit(self.member_a['fields'], self.member_b['fields'])
        self.assertGreater(fit, 1.0)
        p_dir = estimate_directional_acceptance(self.member_a, self.member_b)
        self.assertGreater(p_dir, 0.5)
        p_mutual = estimate_mutual_acceptance(self.member_a, self.member_b)
        self.assertGreater(p_mutual, p_dir * p_dir) # Dyadic covariance property
        u = estimate_expected_msmi(self.member_a, self.member_b)
        self.assertGreater(u, 0.0)

    def test_thompson_sampler(self):
        import random
        sampler = ThompsonFeatureSampler()
        weights = sampler.sample_weights(random.Random(42))
        self.assertIn('relationship_goal', weights)
        # Update
        sampler.update({'relationship_goal': True}, outcome=True)
        self.assertGreater(sampler.beliefs['relationship_goal']['mu'], 0.65)

    def test_waiting_threshold(self):
        self.assertEqual(get_reservation_threshold(10), 0.0)
        self.assertEqual(get_reservation_threshold(59), 0.0)

    def test_allocation_and_ask(self):
        state = {
            'day': 5,
            'ask_budget_remaining': 12,
            'members': [self.member_a, self.member_b],
            'introductions': [],
            'feedback': [],
            'ask_log': []
        }
        pairs = allocate_pairs(state)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(set(pairs[0]), {'syn_001', 'syn_002'})


class TestSurvivalWeightedBandit(unittest.TestCase):
    """Tests for the Survival-Weighted Contextual Bandit framework."""

    def test_kaplan_meier_uninformative_prior(self):
        """With no historical data, conditional success should return the uninformative prior."""
        from src.models.survival_bandit import KaplanMeierSurvivalEstimator
        km = KaplanMeierSurvivalEstimator()
        p = km.predict_conditional_success(5)
        self.assertAlmostEqual(p, 0.12, places=2)

    def test_kaplan_meier_with_history(self):
        """Conditional success probability follows correct survival formula properties."""
        from src.models.survival_bandit import KaplanMeierSurvivalEstimator
        km = KaplanMeierSurvivalEstimator()
        # Successes resolve between days 10-20; rejections resolve between days 3-14
        km.success_times = [10, 12, 15, 18, 20]
        km.rejection_times = [3, 5, 7, 7, 8, 9, 10, 12, 14]

        # Before any resolutions: conditional = marginal = 5/14
        p_zero = km.predict_conditional_success(0)
        self.assertAlmostEqual(p_zero, 5 / 14, delta=0.05)

        # At intermediate times: rejections clear faster than successes,
        # so surviving longer INCREASES the conditional (selection effect).
        # This is mathematically correct: P(success | survived t) rises.
        p_mid = km.predict_conditional_success(8)
        self.assertGreater(p_mid, p_zero)

        # After all outcomes have resolved (S(t)=0), clips to minimum
        p_after = km.predict_conditional_success(25)
        self.assertLess(p_after, 0.01)

        # All probabilities remain in valid range [0, 1]
        for t in range(0, 30):
            p = km.predict_conditional_success(t)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_three_bucket_classification(self):
        """Classify introductions into success, rejection, and pending with correct labels."""
        from src.models.survival_bandit import SurvivalWeightedBandit

        bandit = SurvivalWeightedBandit(feature_dim=7)

        introductions = [
            {'introduction_id': 'intro_success', 'assigned_day': 0,
             'response_deadline_day': 7, 'user_a': 'a', 'user_b': 'b'},
            {'introduction_id': 'intro_reject', 'assigned_day': 0,
             'response_deadline_day': 7, 'user_a': 'c', 'user_b': 'd'},
            {'introduction_id': 'intro_pending', 'assigned_day': 5,
             'response_deadline_day': 12, 'user_a': 'e', 'user_b': 'f'},
        ]

        feedback = [
            # intro_success: full MSMI pipeline
            {'introduction_id': 'intro_success', 'member_id': 'a',
             'event': 'introduction_response', 'value': 'yes',
             'occurred_day': 2, 'observed_day': 2, 'missing_reason': None},
            {'introduction_id': 'intro_success', 'member_id': 'b',
             'event': 'introduction_response', 'value': 'yes',
             'occurred_day': 3, 'observed_day': 3, 'missing_reason': None},
            {'introduction_id': 'intro_success', 'member_id': None,
             'event': 'date_happened', 'value': True,
             'occurred_day': 8, 'observed_day': 8},
            {'introduction_id': 'intro_success', 'member_id': 'a',
             'event': 'second_meeting_intention', 'value': 'yes',
             'occurred_day': 9, 'observed_day': 9, 'missing_reason': None},
            {'introduction_id': 'intro_success', 'member_id': 'b',
             'event': 'second_meeting_intention', 'value': 'yes',
             'occurred_day': 10, 'observed_day': 10, 'missing_reason': None},
            # intro_reject: one person said no
            {'introduction_id': 'intro_reject', 'member_id': 'c',
             'event': 'introduction_response', 'value': 'yes',
             'occurred_day': 2, 'observed_day': 2, 'missing_reason': None},
            {'introduction_id': 'intro_reject', 'member_id': 'd',
             'event': 'introduction_response', 'value': 'no',
             'occurred_day': 3, 'observed_day': 3, 'missing_reason': None},
            # intro_pending: only one response so far, deadline not passed
        ]

        results = bandit.classify_introductions(introductions, feedback,
                                                 current_day=8)

        labels = {r['introduction_id']: r for r in results}

        self.assertEqual(labels['intro_success']['status'], 'success')
        self.assertEqual(labels['intro_success']['label'], 1.0)

        self.assertEqual(labels['intro_reject']['status'], 'rejection')
        self.assertEqual(labels['intro_reject']['label'], 0.0)

        self.assertEqual(labels['intro_pending']['status'], 'pending')
        self.assertGreater(labels['intro_pending']['label'], 0.0)
        self.assertLess(labels['intro_pending']['label'], 1.0)

    def test_serialization_roundtrip(self):
        """Survival bandit state should survive serialization and deserialization."""
        from src.models.survival_bandit import SurvivalWeightedBandit
        import json

        bandit = SurvivalWeightedBandit(feature_dim=7)
        bandit.mu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        bandit.survival.success_times = [10, 15]
        bandit.survival.rejection_times = [5, 8]

        data = bandit.to_dict()
        # Ensure it's JSON-serializable (required for memory transport)
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)

        restored = SurvivalWeightedBandit.from_dict(restored_data)
        self.assertEqual(restored.mu, bandit.mu)
        self.assertEqual(restored.survival.success_times, [10, 15])
        self.assertEqual(restored.survival.rejection_times, [5, 8])

    def test_thompson_sampling_produces_varied_weights(self):
        """Thompson sampling should produce diverse weight samples."""
        import random
        from src.models.survival_bandit import SurvivalWeightedBandit

        bandit = SurvivalWeightedBandit(feature_dim=7)
        rng = random.Random(42)
        samples = [bandit.sample_weights(rng) for _ in range(10)]

        # Check that at least some weights differ across samples
        first_key = sorted(samples[0].keys())[0]
        values = [s[first_key] for s in samples]
        self.assertGreater(max(values) - min(values), 0.01)


if __name__ == '__main__':
    unittest.main()

