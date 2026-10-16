"""Benchmark runner for comparing all three public baselines."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def run_evaluation(baseline: str, seeds: str, variants: str, output_path: str):
    cmd = [
        sys.executable,
        str(ROOT / "evaluate.py"),
        "--baseline", baseline,
        "--seeds", seeds,
        "--variants", variants,
        "--output", output_path
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run full baseline suite")
    parser.add_argument("--seeds", default="101", help="Comma-separated seeds")
    parser.add_argument("--variants", default="development", help="Comma-separated variants or 'all'")
    parser.add_argument("--outdir", default="results", help="Output directory")
    args = parser.parse_args()
    
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    
    baselines = ["greedy", "no_asks", "random"]
    summaries = {}
    
    for b in baselines:
        out_file = str(outdir / f"{b}_benchmark.json")
        run_evaluation(b, args.seeds, args.variants, out_file)
        with open(out_file) as f:
            data = json.load(f)
            summaries[b] = data.get("summary", {}).get("overall", {})
            
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 60)
    header = f"{'Baseline':<12} | {'MSMI/100':<10} | {'Coverage':<10} | {'Mutual/100':<12} | {'Ask Cost':<10}"
    print(header)
    print("-" * len(header))
    for b, s in summaries.items():
        print(f"{b:<12} | {s.get('msmi_per_100_arrived_members', 0):<10.2f} | {s.get('coverage', 0):<10.2%} | {s.get('mutual_acceptances_per_100', 0):<12.2f} | {s.get('ask_cost', 0):<10.1f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
