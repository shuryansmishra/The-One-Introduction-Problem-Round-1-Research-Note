#!/usr/bin/env python3
"""Interactive Terminal & Web Demo for the Sequential Reciprocal Matching Policy.

Demonstrates the policy's core components:
1. Edmonds' Blossom Combinatorial Allocation
2. Kaplan-Meier Survival-Weighted Delay Correction
3. Cost-Normalized Value-of-Information (VOI) Clarification
4. Full Funnel Metrics Tracking (Arrived -> Intro -> Mutual -> Date -> MSMI)

Usage:
    python demo.py                     # Run terminal demo on standard development scenario
    python demo.py --variant delayed   # Run on delayed-feedback scenario
    python demo.py --web               # Launch interactive browser dashboard
"""
import argparse
import sys
import time
from pathlib import Path
from kit import Simulator, generate, VERSION
import policy

# ANSI styling helpers
BOLD = '\033[1m'
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
MAGENTA = '\033[95m'
DIM = '\033[2m'
RESET = '\033[0m'

def print_banner():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════════════╗
║               THE SEQUENTIAL RECIPROCAL MATCHING DEMO                    ║
║   Survival-Weighted Dyadic Bandit with Edmonds' Blossom Optimization     ║
║                  Vouchsafe × IITM Hackathon · Team Zero                  ║
╚══════════════════════════════════════════════════════════════════════════╝{RESET}
""")

def run_terminal_demo(seed=101, variant='development', num_days=15):
    print_banner()
    print(f"Initializing simulation: {BOLD}Seed {seed}{RESET} | {BOLD}Variant: {variant}{RESET} | Horizon: {num_days} days\n")
    
    world = generate(seed, 200, 'evaluation', variant)
    sim = Simulator(world)
    memory = None
    
    for day in range(num_days):
        print(f"{BOLD}{MAGENTA}── Day {day:02d} / 60 ────────────────────────────────────────────────────────{RESET}")
        
        # 1. Ask Phase (VOI Clarifications)
        req_ask = {'schema_version': VERSION, 'phase': 'ask', 'state': sim.observe(), 'memory': memory}
        res_ask = policy.decide(req_ask, mode='proposed')
        memory = res_ask['memory']
        asks = res_ask.get('asks', [])
        resolutions = sim.resolve_asks(asks)
        
        bundle_asks = [a for a in resolutions if a['field'] == 'constraints']
        soft_asks = [a for a in resolutions if a['field'] != 'constraints']
        
        print(f"  {CYAN}[PHASE 1: ASK]{RESET} Spent {sim.ask_spent % 12 or (12 if sim.ask_spent else 0)} budget units:")
        if bundle_asks:
            members_str = ", ".join(a['member_id'] for a in bundle_asks[:3])
            more = f" (+{len(bundle_asks)-3} more)" if len(bundle_asks) > 3 else ""
            print(f"    • Hard Bundles (cost 3): {BOLD}{members_str}{more}{RESET}")
        if soft_asks:
            fields_str = ", ".join(f"{a['member_id']}:{a['field']}" for a in soft_asks[:3])
            more = f" (+{len(soft_asks)-3} more)" if len(soft_asks) > 3 else ""
            print(f"    • Soft Queries (cost 1): {DIM}{fields_str}{more}{RESET}")
            
        # 2. Match Phase (Edmonds' Blossom Allocation)
        req_match = {'schema_version': VERSION, 'phase': 'match', 'state': sim.observe(), 'memory': memory}
        res_match = policy.decide(req_match, mode='proposed')
        memory = res_match['memory']
        pairs = res_match.get('pairs', [])
        sim.advance(pairs)
        
        print(f"  {GREEN}[PHASE 2: MATCH]{RESET} Edmonds' Blossom matched {BOLD}{len(pairs)}{RESET} optimal non-overlapping pairs:")
        for p in pairs[:3]:
            print(f"    • Pair: {p[0]} ⇄ {p[1]}")
        if len(pairs) > 3:
            print(f"    • ... and {len(pairs)-3} more non-overlapping pairs")
            
        # 3. Observed Feedback Events Today
        events_today = [e for e in sim.events if e['observed_day'] == day]
        acceptances = [e for e in events_today if e['event'] == 'introduction_response' and e.get('value') == 'yes']
        dates = [e for e in events_today if e['event'] == 'date_happened' and e.get('value') is True]
        
        m = sim.metrics()
        print(f"  {YELLOW}[STATUS & DELAYED FEEDBACK]{RESET} Total Intros: {len(sim.introductions)} | Mutual Acc: {m['mutual_acceptances']} | Dates: {m['dates']} | MSMI: {BOLD}{m['mutual_second_meeting_intention']}{RESET}")
        print()
        time.sleep(0.15)
        
    print(f"\n{BOLD}Running follow-up horizon for right-censored date feedback...{RESET}")
    for _ in range(40):
        sim.advance([])
        
    final = sim.metrics()
    arrived = len([m for m in sim.members.values() if m['arrived_day'] <= 59])
    score = 100 * final['mutual_second_meeting_intention'] / arrived
    
    print(f"""
{BOLD}{GREEN}═══════════════════════════ FINAL FUNNEL RESULTS ═══════════════════════════{RESET}
  Arrived Members:                  {BOLD}{arrived}{RESET}
  Total Introductions Assigned:     {BOLD}{len(sim.introductions)}{RESET}
  Mutual Acceptances Achieved:      {BOLD}{final['mutual_acceptances']}{RESET}
  Dates Completed:                  {BOLD}{final['dates']}{RESET}
  Mutual 2nd Meeting Intentions:    {BOLD}{GREEN}{final['mutual_second_meeting_intention']}{RESET}
  -------------------------------------------------------------------------
  {BOLD}Primary Score (MSMI / 100 Members): {GREEN}{score:.3f}{RESET}
════════════════════════════════════════════════════════════════════════════
""")

def launch_web_demo():
    import webbrowser
    import http.server
    import socketserver
    
    html_path = Path(__file__).parent / "demo.html"
    if not html_path.exists():
        print(f"Error: {html_path} does not exist.")
        return
        
    port = 8080
    handler = http.server.SimpleHTTPRequestHandler
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            url = f"http://localhost:{port}/demo.html"
            print(f"\n{BOLD}{GREEN}✓ Interactive Web Demo running at:{RESET} {BOLD}{url}{RESET}")
            print(f"{DIM}Press Ctrl+C to stop server{RESET}\n")
            webbrowser.open(url)
            httpd.serve_forever()
    except OSError:
        # Fallback to direct file open if port in use
        print(f"\nOpening {html_path.resolve()} directly in default browser...")
        webbrowser.open(f"file://{html_path.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Sequential Matching Interactive Demo")
    parser.add_argument('--variant', default='development', choices=['development', 'delayed', 'cold_start', 'drift', 'shift', 'sparse'])
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--days', type=int, default=15)
    parser.add_argument('--web', action='store_true', help="Open the interactive Web UI Dashboard")
    args = parser.parse_args()
    
    if args.web:
        launch_web_demo()
    else:
        run_terminal_demo(seed=args.seed, variant=args.variant, num_days=args.days)

if __name__ == '__main__':
    main()
