"""Metric calculation and funnel diagnostic tools."""
from collections import defaultdict

def evaluate_funnel(introductions: list, feedback: list, arrived_count: int = 200) -> dict:
    """Computes standard sequential matching funnel metrics from logs."""
    events_by_id = defaultdict(list)
    for e in feedback:
        events_by_id[e['introduction_id']].append(e)
        
    mutual_acceptances = 0
    dates = 0
    msmi = 0
    missing = 0
    
    for intro in introductions:
        evs = events_by_id[intro['introduction_id']]
        # Responses
        responses = [e for e in evs if e['event'] == 'introduction_response']
        if len(responses) == 2 and all(e.get('value') == 'yes' for e in responses):
            mutual_acceptances += 1
            
        # Dates
        d_evs = [e for e in evs if e['event'] == 'date_happened' and e.get('value') is True]
        if d_evs:
            dates += 1
            
        # MSMI
        s_evs = [e for e in evs if e['event'] == 'second_meeting_intention']
        if d_evs and len(s_evs) == 2 and all(e.get('value') == 'yes' for e in s_evs):
            d_day = d_evs[0]['occurred_day']
            if (d_day - intro['assigned_day'] <= 30) and all(e['occurred_day'] - d_day <= 3 for e in s_evs):
                msmi += 1
                
        missing += sum(e.get('missing_reason') == 'no_response' for e in evs)
        
    return {
        'introductions': len(introductions),
        'mutual_acceptances': mutual_acceptances,
        'mutual_rate': mutual_acceptances / max(1, len(introductions)),
        'dates': dates,
        'date_rate': dates / max(1, mutual_acceptances) if mutual_acceptances else 0.0,
        'msmi': msmi,
        'msmi_per_100': 100 * msmi / max(1, arrived_count),
        'missing_feedback': missing
    }
