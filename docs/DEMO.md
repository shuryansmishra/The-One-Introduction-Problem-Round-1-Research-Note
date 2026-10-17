# Interactive Demo Guide · Team Zero
## The Sequential Reciprocal Matching Machine

This repository includes both a **Rich Web Dashboard** and an **Interactive Terminal CLI** to demonstrate the policy's real-time matching and clarification mechanics.

---

### Option 1: Interactive Browser Dashboard (`demo.html`)

The web demo provides an interactive visualization of the 60-day reciprocal matching process:
- **Visual Network Graph**: Displays the 200 members in an organic constellation, with pulsing halos for available candidates and glowing laser arcs for daily Edmonds' Blossom matches.
- **Scenario Selector**: Toggle between `Standard (Dev)`, `Delayed Dates`, `Cold Start`, and `Distribution Drift`.
- **Playback Controls**: Play / Pause, Step Day +1, Reset, Speed multipliers (1x, 2x, 5x), and Day Scrubber.
- **5-Stage Funnel Scoreboard**: Live counters for Arrived Members ($N$), Blossom Matches ($I$), Mutual Acceptances ($A$), Dates ($D$), and MSMI Outcomes ($M$).
- **Decision Engine Breakdown**:
  - *Phase 1 (VOI Asks)*: Shows cost-normalized information queries (hard bundles & soft preferences).
  - *Phase 2 (Blossom Matches)*: Shows non-overlapping pair assignments with Thompson-sampled expected utilities.
  - *Phase 3 (Delay Correction)*: Live Kaplan-Meier survival curves predicting pending dates.
  - *Benchmark Comparison*: Side-by-side comparison with Greedy, Random, and No-Clarification baselines.

#### How to Launch:
```bash
# Option A: Directly open in any web browser
open demo.html

# Option B: Launch via Python demo script
python demo.py --web
```

---

### Option 2: Terminal Interactive Walkthrough (`demo.py`)

Run the terminal demo to step through the simulation day-by-day with colored ASCII output:

```bash
# Run on default development scenario (15 days)
python demo.py

# Run on the delayed feedback scenario
python demo.py --variant delayed

# Run for custom number of days
python demo.py --variant cold_start --days 30
```

#### Terminal Walkthrough Output Example:
```text
╔══════════════════════════════════════════════════════════════════════════╗
║               THE SEQUENTIAL RECIPROCAL MATCHING DEMO                    ║
║   Survival-Weighted Dyadic Bandit with Edmonds' Blossom Optimization     ║
║                  Vouchsafe × IITM Hackathon · Team Zero                  ║
╚══════════════════════════════════════════════════════════════════════════╝

── Day 00 / 60 ────────────────────────────────────────────────────────
  [PHASE 1: ASK] Spent 12 budget units:
    • Soft Queries (cost 1): syn_ed6dc367...:relationship_pace (+9 more)
  [PHASE 2: MATCH] Edmonds' Blossom matched 8 optimal non-overlapping pairs:
    • Pair: syn_9281244a... ⇄ syn_fdf10ee9...
    • Pair: syn_447148bc... ⇄ syn_9a1797ab...
  [STATUS & DELAYED FEEDBACK] Total Intros: 8 | Mutual Acc: 0 | Dates: 0 | MSMI: 0
```
