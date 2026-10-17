# Final Submission Issue Draft
**Template:** Final submission (`.github/ISSUE_TEMPLATE/final_submission.yml`)  
**Repository:** [RomeoJulietLove/The-Sequential-Matching-Problem](https://github.com/RomeoJulietLove/The-Sequential-Matching-Problem/issues/new?template=final_submission.yml)  
**Submission Target:** Round 2 Final Build (v1.0.0)  

---

### Field 1: Team name and size
```text
Team Zero (1 participant: Shuryans Mishra)
```

### Field 2: Public source URL and immutable version
```text
Repository URL: https://github.com/shuryansmishra/The-One-Introduction-Problem-Round-1-Research-Note.git
40-Character Commit SHA: 38cfc5b9aa6d8b2412233bab812a4e4f93ecf965

Archive URL: https://github.com/shuryansmishra/The-One-Introduction-Problem-Round-1-Research-Note/raw/main/submission_round2.zip
Archive SHA-256 Checksum: 9524f47871c317a117ee869ccfe64d66b6b3b68f730e3120cda4e72ebb7ba2c2
```

### Field 3: Report and results URLs
```text
Technical Report: https://github.com/shuryansmishra/The-One-Introduction-Problem-Round-1-Research-Note/blob/main/docs/TECHNICAL_REPORT.md
Primary Evaluation Results: https://github.com/shuryansmishra/The-One-Introduction-Problem-Round-1-Research-Note/blob/main/results/final.json
```

### Field 4: Build and evaluation commands
```bash
# 1. Verification of unit tests (31 tests pass)
python -m unittest -v

# 2. Synthetic dataset checksum verification
python verify_data.py

# 3. Local multi-seed, multi-scenario evaluation
python evaluate.py --seeds 101,102,103 --variants all --output results/final.json

# 4. Container build & isolated offline CPU evaluation
docker build -t sequential-policy:submission .
python evaluate.py --image sequential-policy:submission --seeds 101 --output results/container_check.json

# Confirmed Constraints:
# - CPU: 2 cores, no GPU
# - RAM: Strictly <= 1 GiB
# - Timeout: Strictly <= 10.0s per invocation (actual latency: ~0.18s)
# - State: Serialized JSON memory <= 150 KiB (limit: 1 MiB)
# - Security: Runs as non-root user 65534:65534 with network=none and read-only filesystem
```

### Field 5: Assets, licences and tools
```text
- Core Runtime: Python 3.12.7-slim
- External Libraries: networkx==3.2.1 (BSD-3-Clause License)
- Algorithms: Edmonds' Blossom Algorithm (max-weight combinatorial matching), Kaplan-Meier survival analysis for right-censored delayed feedback, Bayesian Thompson Sampling with Laplace posterior approximation.
- Declared Seeds: Public experiment seeds 101, 102, 103; Daily inference exploration seed (42 + day).
- Datasets: Standard synthetic dataset v1.0.0 (2,000 adult profiles, 10 disjoint pools); zero external participant data.
- Development Assistance: Developed with Google DeepMind Antigravity AI pair programming assistant.
```

### Field 6: Validation and comparisons
```text
Confirmed:
1. Evaluated across all 3 required public baselines: Greedy, No-Clarification, and Random-Feasible.
2. Evaluated across all 6 scenario families: Standard (development), Sparse Geography (sparse), Cold Start (cold_start), Delayed Dates (delayed), Shifted Weights (shift), and Changing Response Drift (drift).
3. Evaluated hypothesis-driven ablations:
   - Ablation 1 (-matching): Replaced Blossom with greedy ranking (confirms H1).
   - Ablation 2 (-survival): Removed Kaplan-Meier delay soft labels (confirms H2).
   - Ablation 3 (-voi): Replaced VOI clarify with uniform heuristic asking (confirms H4).
4. Container passes all sandbox requirements with 100% valid episodes, 0 errors, and zero non-finite JSON outputs.
```
