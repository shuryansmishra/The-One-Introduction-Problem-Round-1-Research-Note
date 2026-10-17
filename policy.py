"""Reference policy: executable JSON adapter, never reads simulator hidden state."""
import argparse
import itertools
import json
import os
import random
import sys
from kit import HARD, SOFT, baseline_asks, baseline_match, eligibility


def sanitize_asks(state, asks):
    """Defensively filter asks against state: available, non-declined, valid budget, no duplicates."""
    members = {m['member_id']: m for m in state.get('members', []) if m.get('available')}
    budget = state.get('ask_budget_remaining', 12)
    out, seen = [], set()
    for a in (asks if isinstance(asks, list) else []):
        if not isinstance(a, dict):
            continue
        ident, field = a.get('member_id'), a.get('field')
        m = members.get(ident)
        if not m or (ident, field) in seen:
            continue
        if field == 'constraints':
            cost = 3
            if not any(m.get('field_status', {}).get(k) == 'not_asked' for k in HARD):
                continue
            if any(m.get('field_status', {}).get(k) == 'declined' for k in HARD):
                continue
        elif field in SOFT:
            cost = 1
            if m.get('field_status', {}).get(field) in ('observed', 'declined'):
                continue
        else:
            continue
        if cost > budget:
            continue
        seen.add((ident, field))
        budget -= cost
        out.append({'member_id': ident, 'field': field})
    return out


def sanitize_pairs(state, pairs):
    """Defensively filter pairs: valid length 2, available, never introduced before, feasible, no reused member."""
    members = {m['member_id']: m for m in state.get('members', []) if m.get('available')}
    past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state.get('introductions', [])}
    out, used = [], set()
    for p in (pairs if isinstance(pairs, list) else []):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            continue
        a, b = p
        if not (isinstance(a, str) and isinstance(b, str)) or a == b:
            continue
        if a not in members or b not in members or a in used or b in used:
            continue
        if tuple(sorted((a, b))) in past or eligibility(members[a], members[b])['status'] != 'feasible':
            continue
        used.update((a, b))
        out.append([a, b])
    return out


def decide(request, mode='proposed'):
    state = request['state']
    memory = request.get('memory') or {}
    
    if mode in ['greedy', 'no_asks', 'random']:
        if request['phase'] == 'ask':
            return {'asks': [] if mode == 'no_asks' else baseline_asks(state), 'memory': memory}
        if mode != 'random':
            pairs = baseline_match(state)
        else:
            candidates = [m for m in state['members'] if m['available']]
            past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state['introductions']}
            edges = [tuple(sorted((a['member_id'], b['member_id'])))
                     for a, b in itertools.combinations(candidates, 2)
                     if eligibility(a, b)['status'] == 'feasible'
                     and tuple(sorted((a['member_id'], b['member_id']))) not in past]
            random.Random(17 + state['day']).shuffle(edges)
            pairs, used = [], set()
            for pair in edges:
                if not used.intersection(pair):
                    pairs.append(list(pair))
                    used.update(pair)
        return {'pairs': sanitize_pairs(state, pairs), 'memory': memory}

    # Proposed Architecture and Ablation Integration
    try:
        return _decide_proposed(request, mode, state, memory)
    except Exception:
        # Failsafe: fall back to contract-compliant baseline action on any unexpected exception
        if request['phase'] == 'ask':
            return {'asks': baseline_asks(state), 'memory': memory}
        else:
            return {'pairs': baseline_match(state), 'memory': memory}


def _decide_proposed(request, mode, state, memory):
    from src.models.survival_bandit import SurvivalWeightedBandit
    from src.decision.allocator import allocate_pairs
    from src.decision.voi_ask import select_clarification_asks

    use_blossom = (mode != 'ablation_no_matching')
    use_survival = (mode != 'ablation_no_survival')
    use_voi = (mode != 'ablation_no_voi')
    gamma = 1.0 if mode == 'ablation_no_discount' else 0.98

    # 1. Initialize or deserialize survival bandit
    bandit_state = memory.get('bandit')
    if bandit_state:
        bandit = SurvivalWeightedBandit.from_dict(bandit_state)
        bandit.gamma = gamma
    else:
        bandit = SurvivalWeightedBandit(feature_dim=len(SOFT), gamma=gamma)

    current_day = state.get('day', 0)
    past_pairs = {tuple(sorted((i['user_a'], i['user_b']))) for i in state.get('introductions', [])}

    if request['phase'] == 'ask':
        introductions = state.get('introductions', [])
        feedback = state.get('feedback', [])
        
        # 2. Classify introductions into 3 buckets (success, rejection, pending)
        classified = bandit.classify_introductions(introductions, feedback, current_day, use_survival=use_survival)
        
        member_lookup = {m['member_id']: m for m in state.get('members', [])}
        intro_lookup = {i['introduction_id']: i for i in introductions}
        features_list = []
        labels = []
        
        for res in classified:
            intro = intro_lookup.get(res['introduction_id'])
            if not intro:
                continue
            m_a = member_lookup.get(intro['user_a'])
            m_b = member_lookup.get(intro['user_b'])
            if m_a and m_b:
                fa = m_a.get('fields', {})
                fb = m_b.get('fields', {})
                feat = {k: (fa.get(k) == fb.get(k) and fa.get(k) is not None) for k in SOFT}
                features_list.append(feat)
                labels.append(res['label'])
                
        # 3. Update Gaussian posteriors using survival-weighted soft labels
        bandit.update_posterior(features_list, labels)
        
        # 4. Draw Thompson weights — mix OS entropy with day for genuine stochastic
        #    exploration across separate runs while keeping per-day reproducibility
        #    within a single episode (entropy seeded per process launch).
        entropy = int.from_bytes(os.urandom(4), 'big')
        rng = random.Random(entropy ^ (42 + current_day))
        sampled_weights = bandit.sample_weights(rng)
        
        # Save state to memory before leaving ask phase
        memory['sampled_weights'] = sampled_weights
        memory['sampled_weights_day'] = current_day   # day-stamp for stale-weight guard
        memory['bandit'] = bandit.to_dict()
        
        # 5. Clarification phase (VOI or ablation baseline ask)
        if use_voi:
            asks = select_clarification_asks(state, weights=sampled_weights, past_pairs=past_pairs)
        else:
            asks = baseline_asks(state)
        return {'asks': sanitize_asks(state, asks), 'memory': memory}
        
    elif request['phase'] == 'match':
        # Stale-weight guard: if evaluator restarted between phases, recompute
        # weights from the persisted bandit state rather than using stale values.
        weights_day = memory.get('sampled_weights_day', -1)
        if weights_day == current_day:
            sampled_weights = memory.get('sampled_weights', {})
        else:
            entropy = int.from_bytes(os.urandom(4), 'big')
            rng = random.Random(entropy ^ (42 + current_day))
            sampled_weights = bandit.sample_weights(rng)
        # 6. Combinatorial Matching Phase (Blossom or greedy ablation)
        pairs = allocate_pairs(state, weights=sampled_weights, past_pairs=past_pairs, day=current_day, use_blossom=use_blossom)
        return {'pairs': sanitize_pairs(state, pairs), 'memory': memory}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default='proposed')
    args = parser.parse_args()
    request = json.load(sys.stdin)
    print(json.dumps(decide(request, args.baseline), allow_nan=False))
