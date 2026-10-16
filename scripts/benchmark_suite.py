"""Comprehensive benchmark suite for Round 2 Final Build.

Runs proposed policy, required baselines (greedy, no_asks, random), and ablations
across all 6 scenario families and specified seeds, saving machine-readable outputs
and printing markdown-compatible comparative tables.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def run_eval(baseline: str, seeds: str, variants: str, output_path: str):
    cmd = [
        sys.executable,
        str(ROOT / "evaluate.py"),
        "--baseline", baseline,
        "--seeds", seeds,
        "--variants", variants,
        "--output", output_path
    ]
    print(f"\n[BENCHMARK] Running mode '{baseline}' on variants [{variants}] seeds [{seeds}]...")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run complete Round 2 benchmark suite")
    parser.add_argument("--seeds", default="101", help="Comma-separated seeds, e.g. 101 or 101,102,103")
    parser.add_argument("--variants", default="all", help="Comma-separated variants or 'all'")
    parser.add_argument("--mode", choices=["all", "proposed", "baselines", "ablations"], default="all")
    parser.add_argument("--outdir", default="results", help="Output directory")
    args = parser.parse_args()

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    configs = []
    if args.mode in ["all", "proposed"]:
        configs.append(("proposed", "final.json"))
    if args.mode in ["all", "baselines"]:
        configs.append(("greedy", "baseline_greedy.json"))
        configs.append(("no_asks", "baseline_no_asks.json"))
        configs.append(("random", "baseline_random.json"))
    if args.mode in ["all", "ablations"]:
        configs.append(("ablation_no_matching", "ablation_matching.json"))
        configs.append(("ablation_no_survival", "ablation_survival.json"))
        configs.append(("ablation_no_voi", "ablation_voi.json"))

    results = {}
    for baseline, filename in configs:
        target = str(outdir / filename)
        run_eval(baseline, args.seeds, args.variants, target)
        with open(target, "r") as f:
            data = json.load(f)
            results[baseline] = data.get("summary", {})

    print("\n" + "=" * 80)
    print("ROUND 2 COMPARATIVE EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Configuration':<24} | {'MSMI/100':<10} | {'Coverage':<10} | {'Mutual/100':<12} | {'Ask Cost':<10} | {'Runtime (s)':<12}")
    print("-" * 80)
    for name, summ in results.items():
        ov = summ.get("overall", {})
        msmi = ov.get("msmi_per_100_arrived_members", 0.0)
        cov = ov.get("coverage", 0.0) * 100.0
        mut = ov.get("mutual_acceptances_per_100", 0.0)
        cost = ov.get("ask_cost", 0.0)
        t = ov.get("inference_seconds", 0.0)
        print(f"{name:<24} | {msmi:<10.2f} | {cov:<9.1f}% | {mut:<12.2f} | {cost:<10.1f} | {t:<12.2f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
