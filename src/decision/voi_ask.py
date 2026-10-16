"""Value of Information (VOI) clarification engine.

Selects clarifications that maximize decision impact per budget unit:
Priority(q) = VOI(q) / Cost(q)
"""
from kit import HARD, SOFT, eligibility
from src.models.dyadic_model import estimate_expected_msmi, DEFAULT_WEIGHTS

def compute_member_bundle_voi(member: dict, available_members: list, past_pairs: set, weights: dict) -> float:
    """Compute the Value of Information for requesting member's hard constraint bundle (cost 3).
    
    VOI is high if clarifying the bundle can convert near-feasible pairs into valid, high-utility matches.
    VOI is 0 if no partner is eligible or all possible partners already violate known hard constraints.
    """
    m_id = member['member_id']
    f = member['fields']
    # If all hard constraints are already observed, VOI is 0
    if not any(f.get(k) is None for k in HARD):
        return 0.0
    # If any hard constraint was declined, bundle cannot be completed
    if any(member.get('field_status', {}).get(k) == 'declined' for k in HARD):
        return 0.0
        
    potential_utility = 0.0
    viable_partner_count = 0
    
    for other in available_members:
        o_id = other['member_id']
        if o_id == m_id:
            continue
        key = tuple(sorted((m_id, o_id)))
        if key in past_pairs:
            continue
            
        el = eligibility(member, other)
        # If already known infeasible, clarification cannot save this pair
        if el['status'] == 'infeasible':
            continue
            
        # If needs_clarification, check potential utility
        if el['status'] == 'needs_clarification':
            # Estimate potential expected MSMI if the pair becomes feasible
            u = estimate_expected_msmi(member, other, weights=weights)
            if u > potential_utility:
                potential_utility = u
            viable_partner_count += 1
            
    # VOI is proportional to best potential match utility discounted by probability of passing hard checks
    estimated_pass_prob = 0.55
    voi = potential_utility * estimated_pass_prob * min(2.0, 1.0 + 0.1 * viable_partner_count)
    return voi

def compute_soft_field_voi(member: dict, field: str, available_members: list, weights: dict) -> float:
    """Compute VOI for clarifying a specific soft field (cost 1)."""
    f = member['fields']
    if f.get(field) is not None or member.get('field_status', {}).get(field) == 'declined':
        return 0.0
        
    w = weights.get(field, 0.1)
    # Higher soft weight => higher variance reduction in utility estimate
    return w * 0.40

def select_clarification_asks(state: dict, weights: dict = None, past_pairs: set = None) -> list:
    """Select the optimal portfolio of asks under the 12-unit daily budget.
    Hard bundle costs 3 units; soft field costs 1 unit.
    """
    w = weights or DEFAULT_WEIGHTS
    budget = state.get('ask_budget_remaining', 12)
    available = [m for m in state.get('members', []) if m.get('available')]
    past = past_pairs or set()
    
    candidates = []
    
    # 1. Evaluate hard bundles
    for m in available:
        # Check if bundle has missing hard fields and no declined fields
        f = m.get('fields', {})
        st = m.get('field_status', {})
        if any(f.get(k) is None for k in HARD) and not any(st.get(k) == 'declined' for k in HARD):
            voi = compute_member_bundle_voi(m, available, past, w)
            roi = voi / 3.0 # cost 3
            if roi > 0:
                candidates.append({
                    'ask': {'member_id': m['member_id'], 'field': 'constraints'},
                    'cost': 3,
                    'roi': roi
                })
                
    # Sort candidates by cost-normalized ROI
    candidates.sort(key=lambda c: c['roi'], reverse=True)
    
    chosen_asks = []
    spent = 0
    seen = set()
    
    for item in candidates:
        ask = item['ask']
        cost = item['cost']
        key = (ask['member_id'], ask['field'])
        if key in seen:
            continue
        if spent + cost <= budget:
            chosen_asks.append(ask)
            seen.add(key)
            spent += cost
            if spent >= budget:
                break
                
    return chosen_asks
