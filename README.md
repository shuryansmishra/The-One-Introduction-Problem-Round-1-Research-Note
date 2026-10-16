# The Sequential Matching Problem

**Final participant release 1.0.0 | Vouchsafe | 5 October 2026**

Build a policy that decides who to introduce, when to wait and what to clarify, using incomplete reciprocal preferences and delayed feedback. Every person, conversation and outcome in this kit is synthetic.

## Start here

1. Read the [final problem statement](PROBLEM_STATEMENT.md) or [download the PDF](docs/PROBLEM_STATEMENT.pdf).
2. Download this repository with **Code → Download ZIP**, or clone it. The full dataset is already in `data/`; no Supabase account, API key or separate download is required.
3. Install Python 3.10 or later. Open a terminal in the extracted repository folder and run:

```bash
python -m unittest -v
python verify_data.py
python evaluate.py --seeds 101 --output results/first_run.json
```

On Windows, use `py` instead of `python` if needed. On macOS/Linux, use `python3` if your Python executable has that name. The starter has no external Python dependencies. Results are written to the requested JSON file; scores measure this invented simulator only.

## Round 2 Final Build (Selected Build)

Our Round 1 proposal was officially selected for Round 2! The final production build and comprehensive report are available at:
- **Final Technical Report:** **[docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)**
- **Round 1 Research Note:** **[docs/RESEARCH_NOTE.md](docs/RESEARCH_NOTE.md)**
- **Primary Results:** **[results/final.json](results/final.json)**

### Final Build Architecture
1. **Dyadic Reciprocal Compatibility Model:** Models joint pair outcomes directly, capturing latent chemistry covariance.
2. **Survival-Weighted Delay Correction:** Kaplan-Meier survival curves calculate exact conditional probability for pending dates $\tilde{y}(t) = \Pr(Y=1 \mid \text{pending at } t)$, eliminating delayed feedback bias.
3. **Bayesian Thompson Sampling:** Balances exploration vs. exploitation over soft preferences with exponential recency discounting ($\gamma = 0.98$).
4. **Edmonds' Blossom Allocation:** Polynomial-time Maximum-Weight Combinatorial Matching (`networkx.max_weight_matching`), solving the greedy starvation/blocking trap.
5. **Cost-Normalized VOI Clarification:** Allocates the 12-unit daily budget strictly where information can alter matching decisions.

### Full Reproducible Evaluation Sequence
```bash
# 1. Run unit test suite (31 tests)
python -m unittest -v

# 2. Verify synthetic dataset checksums (2,000 members, 10 pools)
python verify_data.py

# 3. Evaluate Proposed policy across all 6 scenarios and seeds
python evaluate.py --seeds 101,102,103 --variants all --output results/final.json

# 4. Run baseline comparisons
python evaluate.py --baseline greedy --seeds 101,102,103 --variants all --output results/baseline_greedy.json
python evaluate.py --baseline no_asks --seeds 101,102,103 --variants all --output results/baseline_no_asks.json
python evaluate.py --baseline random --seeds 101,102,103 --variants all --output results/baseline_random.json

# 5. Run ablation studies
python evaluate.py --baseline ablation_no_matching --seeds 101 --variants all --output results/ablation_matching.json
python evaluate.py --baseline ablation_no_survival --seeds 101 --variants all --output results/ablation_survival.json
python evaluate.py --baseline ablation_no_voi --seeds 101 --variants all --output results/ablation_voi.json
```

---

## Research Submission (Round 1)

The complete Round 1 research submission note is available at:
**[docs/RESEARCH_NOTE.md](docs/RESEARCH_NOTE.md)**

It provides a comprehensive, publication-ready analysis covering:
- Problem interpretation, 60-day horizon, and MSMI objective formulation.
- Symmetrical reciprocal hard constraints and missing data tri-state mechanics (`observed`, `not_asked`, `declined`).
- Dyadic reciprocal compatibility and why $P(\text{Mutual Yes}) > P(A \to B) \times P(B \to A)$ due to latent chemistry covariance.
- Decision-Theoretic Value of Information (VOI) clarification engine ($\text{VOI}(q) / \text{Cost}(q)$).
- Global Maximum-Weight Combinatorial Allocation solving the greedy starvation failure mode.
- Bayesian Thompson Sampling exploration under selective labels and non-stationary drift ($\gamma = 0.98$).
- Systematic benchmark comparison against all three baselines across all six scenario families.

---

## System Architecture & File Layout

```text
The-Sequential-Matching-Problem/
├── docs/
│   ├── RESEARCH_NOTE.md         # Master Round 1 Research Submission
│   ├── DATA_CONTRACT.md         # Data dictionary & constraint rules
│   ├── POLICY_INTERFACE.md      # Executable JSON protocol (stdin/stdout)
│   ├── SUBMISSION.md            # Hackathon schedule & submission guide
│   └── PROBLEM_STATEMENT.md     # Official problem specification
├── src/
│   ├── models/
│   │   ├── dyadic_model.py      # Reciprocal compatibility & multi-stage MSMI funnel
│   │   └── thompson_sampler.py  # Bayesian Thompson Sampling with recency discount
│   ├── decision/
│   │   ├── voi_ask.py           # Value of Information (VOI) clarification engine
│   │   ├── allocator.py         # Global Maximum-Weight Matching allocator
│   │   └── waiting_policy.py    # Dynamic reservation threshold under finite horizon
│   └── utils/
│       └── metrics.py           # Evaluation helpers & funnel analytics
├── scripts/
│   ├── analyze_pools.py         # Empirical data analysis across train/val/test splits
│   └── run_baselines.py         # Automated runner for comparing all three baselines
├── data/                        # 10 synthetic pools (200 members each)
├── policy.py                    # JSON adapter executable entrypoint
├── evaluate.py                  # Evaluation harness (60 decision days + 40 follow-up)
└── test_system_modules.py       # Unit tests for research and decision modules
```

---

## Core Mathematical Formulations

### 1. Reciprocal Dyadic Compatibility Model
Directional acceptance probabilities share latent chemistry $\xi_{ij} \sim \mathcal{N}(0, \sigma^2)$:
$$\text{logit}(P(i \to j \mid \xi_{ij})) = \beta_0 + \alpha_i + \text{fit}(i, j) + \xi_{ij} + \delta_t$$
Because $\xi_{ij}$ is shared by both partners, Chebyshev's association inequality implies positive covariance $\text{Cov}(P_{i \to j}, P_{j \to i}) > 0$. The joint probability is evaluated via Gauss-Hermite integration:
$$P(\text{Mutual Acceptance}_{ij}) = \mathbb{E}_{\xi_{ij}}\left[ \sigma(\text{logit}_i(\xi_{ij})) \cdot \sigma(\text{logit}_j(\xi_{ij})) \right] > \mathbb{E}[P(i \to j)] \times \mathbb{E}[P(j \to i)]$$

### 2. Decision-Theoretic Value of Information (VOI)
For candidate clarification query $q \in \{\text{bundle}_i, \text{soft}_{i, k}\}$ with cost $c(q) \in \{3, 1\}$:
$$\text{VOI}(q) = \mathbb{E}_{Z_q}\left[ \max_{\mathcal{P}} \sum_{(u,v) \in \mathcal{P}} U_{uv}(Z_q) \right] - \max_{\mathcal{P}} \sum_{(u,v) \in \mathcal{P}} U_{uv}$$
$$\text{Priority}(q) = \frac{\text{VOI}(q)}{c(q)}$$

### 3. Global Combinatorial Matching (Max-Weight Independent Set)
$$\max_x \sum_{(i,j) \in \mathcal{E}_{\text{feasible}}} \tilde{U}_{ij} \cdot x_{ij} \quad \text{subject to} \quad \sum_{j} x_{ij} \le 1, \; x_{ij} \in \{0, 1\}$$

### 4. Bayesian Exploration under Selective Labels (Thompson Sampling)
Feature weights $\boldsymbol{w} \sim \mathcal{N}(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)$ are updated using observed introductions, with exponential recency discounting $\gamma = 0.98$ for non-stationary drift:
$$\boldsymbol{\Sigma}_{t+1}^{-1} = \gamma \boldsymbol{\Sigma}_t^{-1} + \boldsymbol{X}^\top \boldsymbol{X}$$

---

## What is included

| File / folder | Purpose |
|---|---|
| [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md) | Final challenge, ranking, resources, schedule and ownership rules |
| [data/](data/) | 2,000 synthetic adults, ten disjoint pools and day-30 observable histories |
| [data_manifest.json](data_manifest.json) | Pool counts, source seeds and train/validation/development-test split |
| [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) | Every field, constraint and feedback rule |
| [docs/POLICY_INTERFACE.md](docs/POLICY_INTERFACE.md) | Executable JSON request/response protocol |
| [policy.py](policy.py) | Runnable greedy, no-clarification and random-feasible baselines |
| [evaluate.py](evaluate.py) | 60-day episodes, 40-day follow-up, metrics and scenario-level score |
| [build_public_data.py](build_public_data.py) | Rebuild public synthetic tables in a new folder |
| [kit.py](kit.py) | Reproducible public simulator and reciprocal eligibility checks |
| [organise.py](organise.py) | Prepare and grade 120 private episodes outside the public repo |
| [docs/ORGANISER_RUNBOOK.md](docs/ORGANISER_RUNBOOK.md) | Complete assessment commands and ranking procedure |
| [Dockerfile](Dockerfile) | Offline container packaging for assessed inference |
| [docs/SUBMISSION.md](docs/SUBMISSION.md) | Complete research and build submission instructions |
| [docs/INTEGRATION.md](docs/INTEGRATION.md) | Reusable adapter boundary for later Vouchsafe evaluation |
| [docs/FAQ.md](docs/FAQ.md) | Common questions and communication policy |
| [examples/baseline_results/](examples/baseline_results/) | Checked reference runs for all three baselines |
| [examples/REPORT_GUIDE.md](examples/REPORT_GUIDE.md) | What a complete participant technical report must cover |

## Develop and compare

Edit `policy.py`, keeping the JSON interface. Observe only the input state. Refreshing observations after clarification is handled by the evaluator. Policy memory is passed explicitly between calls and reset between episodes.

```bash
python evaluate.py --baseline greedy --seeds 101,102,103 --variants all --output results/greedy.json
python evaluate.py --baseline no_asks --seeds 101,102,103 --variants all --output results/no_asks.json
python evaluate.py --baseline random --seeds 101,102,103 --variants all --output results/random.json
```

The full commands take longer than the one-episode quick start. Public variants are `development`, `sparse`, `cold_start`, `delayed`, `shift`, and `drift`. Use your own declared training seeds. Keep the supplied six training pools, two validation pools and two development-test pools disjoint. Day-30 snapshots are not observations from earlier decisions.

To check container execution after installing Docker:

```bash
docker build -t sequential-policy:1.0 .
python evaluate.py --image sequential-policy:1.0 --seeds 101 --output results/container.json
```

Local subprocess mode is only for trusted development code. Assessed submissions use offline containers with no host evaluation files mounted. Private organiser worlds are intentionally absent from this public repository.

To regenerate the public tables without changing the supplied data:

```bash
python build_public_data.py --output ../rebuilt-public-data
```

The new folder must not already exist. The generator uses only invented data and public seeds. Do not use seed reconstruction inside an evaluated policy.

## Submit and ask questions

Teams: **1–4 participants**.

| Milestone | Date and time (IST) |
|---|---|
| Round 1 research submission closes | 9 October 2026, 23:59 |
| Round 1 results announced by | 11 October 2026, 22:00 |
| Round 2 build starts | 12 October 2026 |
| Final build submission closes | 18 October 2026, 23:59 |

**Round 1:** submit the research note through a Google Form. The organisers will release the link soon through [Discord](https://discord.gg/GwZdY54Gq) and [the submission guide](docs/SUBMISSION.md#research-submission). GitHub Issues are not the research submission route.

**Round 2:** use the [Final submission](https://github.com/RomeoJulietLove/The-Sequential-Matching-Problem/issues/new?template=final_submission.yml) issue form by the final deadline. Provide the immutable submitted version and required report/results. Use [Question](https://github.com/RomeoJulietLove/The-Sequential-Matching-Problem/issues/new?template=question.yml) for technical questions. Never post personal data, access codes or credentials in public issues.

This repository and public Issues are the shared technical communication hub. Event announcements and reminders are shared through [the IITM Discord server](https://discord.gg/GwZdY54Gq). Questions answered here are visible to every team. See [CHANGELOG.md](CHANGELOG.md) for release corrections.

The [starter-code permission](LICENSE) and [synthetic-data permission](DATA_LICENSE.md) permit reuse of the supplied kit. Students retain ownership of their own work; participation does not grant Vouchsafe a commercial licence to submissions. Later use requires a separate agreement.
