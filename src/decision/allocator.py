"""Global Combinatorial Allocation Engine.

Solves Maximum-Weight Independent Matching across the feasible graph,
avoiding the greedy starvation failure mode.
"""
import itertools
from kit import eligibility
from src.models.dyadic_model import estimate_expected_msmi
from src.decision.waiting_policy import get_reservation_threshold

def max_weight_matching_heuristic(edges: list) -> list:
    """Solve Maximum-Weight Matching over general graph edges [(w, u, v), ...].
    
    Uses Edmonds' blossom algorithm via NetworkX to find the global optimum.
    """
    import networkx as nx
    
    G = nx.Graph()
    for weight, u, v in edges:
        G.add_edge(u, v, weight=weight)
        
    # networkx max_weight_matching returns a set of tuples representing matched edges
    matching = nx.max_weight_matching(G, maxcardinality=False)
    
    # Convert set of tuples back to list of lists
    return [list(pair) for pair in matching]


def allocate_pairs(state: dict, weights: dict = None, past_pairs: set = None, day: int = None, use_blossom: bool = True) -> list:
    """Computes optimal non-overlapping feasible pair introductions for the current day.
    
    If use_blossom=True, uses Edmonds' Blossom algorithm for global maximum-weight matching.
    If use_blossom=False, uses greedy edge selection by descending weight (for ablation H1).
    """
    current_day = day if day is not None else state.get('day', 0)
    available_members = [m for m in state.get('members', []) if m.get('available')]
    past = past_pairs or set()
    tau = get_reservation_threshold(current_day)
    
    # Build candidate feasible edge list
    edges = []
    for a, b in itertools.combinations(available_members, 2):
        key = tuple(sorted((a['member_id'], b['member_id'])))
        if key in past:
            continue
        el = eligibility(a, b)
        if el['status'] != 'feasible':
            continue
            
        u_expected = estimate_expected_msmi(a, b, weights=weights)
        if u_expected >= tau:
            edges.append((u_expected, a['member_id'], b['member_id']))
            
    if use_blossom:
        # Solve global combinatorial matching via Edmonds' blossom algorithm
        pairs = max_weight_matching_heuristic(edges)
    else:
        # Greedy heuristic: sort edges descending and greedily match non-conflicting pairs
        edges.sort(key=lambda x: x[0], reverse=True)
        pairs, used = [], set()
        for _, u, v in edges:
            if u not in used and v not in used:
                pairs.append([u, v])
                used.add(u)
                used.add(v)
    return pairs
