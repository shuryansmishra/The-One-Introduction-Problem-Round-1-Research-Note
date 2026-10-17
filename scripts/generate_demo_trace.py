"""Generate precomputed simulation traces for the interactive Web Demo.

Runs the proposed policy against the kit simulator across variants and records
per-day events: asks, Blossom matches, delayed feedback, and cumulative metrics.
Outputs a self-contained JS/JSON file so demo.html works instantly offline with file:// URLs.
"""
import copy
import json
import math
from pathlib import Path
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kit import Simulator, generate, VERSION, eligibility
import policy

def run_trace(seed=101, variant='development', max_days=60):
    world = generate(seed, 200, 'evaluation', variant)
    sim = Simulator(world)
    memory = None
    
    # Pre-assign 2D positions for members in a force-directed layout ring
    member_positions = {}
    all_members = world['members']
    n = len(all_members)
    for idx, m in enumerate(all_members):
        angle = (2 * math.pi * idx) / n
        radius = 280 + (idx % 5) * 20
        # cluster slightly by gender and zone
        gx = 120 if m['gender'] == 'f' else -120
        member_positions[m['member_id']] = {
            'x': round(400 + gx + radius * 0.7 * math.cos(angle), 1),
            'y': round(350 + radius * 0.7 * math.sin(angle), 1),
            'age': m['age'],
            'gender': m['gender'],
            'zone': m['zone'],
            'pool_id': m['pool_id'],
            'goal': m['truth'].get('relationship_goal', 'unknown'),
            'pace': m['truth'].get('relationship_pace', 'unknown')
        }
    
    days_data = []
    
    for day in range(max_days):
        day_record = {
            'day': day,
            'arrived_today': [m['member_id'] for m in sim.members.values() if m['arrived_day'] == day],
            'available_count': len([m for m in sim.members.values() if sim.available(m)]),
            'asks': [],
            'matches': [],
            'events_today': [],
            'metrics': {}
        }
        
        # 1. Ask phase
        req_ask = {'schema_version': VERSION, 'phase': 'ask', 'state': sim.observe(), 'memory': memory}
        res_ask = policy.decide(req_ask, mode='proposed')
        memory = res_ask['memory']
        asks = res_ask.get('asks', [])
        
        ask_resolutions = sim.resolve_asks(asks)
        for ask_res in ask_resolutions:
            day_record['asks'].append({
                'member_id': ask_res['member_id'],
                'field': ask_res['field'],
                'values': ask_res.get('values', {}),
                'statuses': ask_res.get('statuses', {})
            })
            
        # 2. Match phase
        req_match = {'schema_version': VERSION, 'phase': 'match', 'state': sim.observe(), 'memory': memory}
        res_match = policy.decide(req_match, mode='proposed')
        memory = res_match['memory']
        pairs = res_match.get('pairs', [])
        
        # Record feasible candidate count for context
        day_record['matches'] = [{'user_a': p[0], 'user_b': p[1]} for p in pairs]
        
        sim.advance(pairs)
        
        # Feedback observed today
        observed_today = [e for e in sim.events if e['observed_day'] == day]
        for e in observed_today:
            day_record['events_today'].append({
                'event': e['event'],
                'value': e.get('value'),
                'intro_id': e.get('introduction_id'),
                'member_id': e.get('member_id')
            })
            
        cur_metrics = sim.metrics()
        arrived_so_far = len([m for m in sim.members.values() if m['arrived_day'] <= day])
        day_record['metrics'] = {
            'arrived': arrived_so_far,
            'introductions': len(sim.introductions),
            'mutual_acceptances': cur_metrics['mutual_acceptances'],
            'dates': cur_metrics['dates'],
            'msmi': cur_metrics['mutual_second_meeting_intention'],
            'ask_spent': sim.ask_spent
        }
        
        days_data.append(day_record)
        
    # Follow-up 40 days
    for _ in range(40):
        sim.advance([])
        
    final_metrics = sim.metrics()
    
    return {
        'seed': seed,
        'variant': variant,
        'member_positions': member_positions,
        'days': days_data,
        'final_summary': {
            'msmi': final_metrics['mutual_second_meeting_intention'],
            'mutual_acceptances': final_metrics['mutual_acceptances'],
            'dates': final_metrics['dates'],
            'total_introductions': len(sim.introductions),
            'ask_spent': sim.ask_spent
        }
    }

def main():
    variants = ['development', 'delayed', 'cold_start', 'drift']
    all_traces = {}
    print("Generating simulation traces for interactive demo...")
    for v in variants:
        print(f"  Simulating variant '{v}' (seed 101)...")
        all_traces[v] = run_trace(101, v)
        print(f"    Done: MSMI = {all_traces[v]['final_summary']['msmi']}, Intros = {all_traces[v]['final_summary']['total_introductions']}")
        
    out_dir = Path("demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_path = out_dir / "demo_traces.json"
    json_path.write_text(json.dumps(all_traces, indent=2))
    
    # Also save as JS global variable so file:/// works seamlessly without CORS
    js_path = out_dir / "demo_data.js"
    js_path.write_text("window.DEMO_TRACES = " + json.dumps(all_traces) + ";\n")
    print(f"Traces written to {json_path} and {js_path}")

if __name__ == '__main__':
    main()
