"""Analyze synthetic pools to extract empirical distributions, missingness rates, and outcome funnels."""
import json
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def analyze_pool(pool_name):
    pool_path = DATA_DIR / pool_name
    if not pool_path.exists():
        return None
    
    with open(pool_path / "members.jsonl") as f:
        members = [json.loads(line) for line in f]
    
    with open(pool_path / "introductions.jsonl") as f:
        introductions = [json.loads(line) for line in f]
        
    with open(pool_path / "feedback.jsonl") as f:
        feedback = [json.loads(line) for line in f]

    # Compute missingness
    missingness = defaultdict(lambda: Counter())
    for m in members:
        for k, status in m.get("field_status", {}).items():
            missingness[k][status] += 1
            
    # Compute funnel counts
    events_by_intro = defaultdict(list)
    for e in feedback:
        events_by_intro[e["introduction_id"]].append(e)
        
    mutual_yes = 0
    dates = 0
    msmi = 0
    missing_resp = 0
    
    for intro in introductions:
        evs = events_by_intro[intro["introduction_id"]]
        resp = [e for e in evs if e["event"] == "introduction_response"]
        if len(resp) == 2 and all(e.get("value") == "yes" for e in resp):
            mutual_yes += 1
        d_evs = [e for e in evs if e["event"] == "date_happened" and e.get("value") is True]
        if d_evs:
            dates += 1
        s_evs = [e for e in evs if e["event"] == "second_meeting_intention"]
        if d_evs and len(s_evs) == 2 and all(e.get("value") == "yes" for e in s_evs):
            d_day = d_evs[0]["occurred_day"]
            if (d_day - intro["assigned_day"] <= 30) and all(e["occurred_day"] - d_day <= 3 for e in s_evs):
                msmi += 1
        missing_resp += sum(e.get("missing_reason") == "no_response" for e in evs)

    return {
        "pool": pool_name,
        "members": len(members),
        "introductions": len(introductions),
        "mutual_yes": mutual_yes,
        "dates": dates,
        "msmi": msmi,
        "missing_resp": missing_resp,
        "missingness": dict(missingness)
    }

def main():
    manifest_path = DATA_DIR.parent / "data_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    splits = manifest["splits"]
    results = {}
    for split_name, pool_list in splits.items():
        results[split_name] = [analyze_pool(p) for p in pool_list]
        
    print(f"=== Empirical Pool Analysis Across Splits ===")
    for split_name, pool_results in results.items():
        total_intros = sum(r["introductions"] for r in pool_results)
        total_mutual = sum(r["mutual_yes"] for r in pool_results)
        total_dates = sum(r["dates"] for r in pool_results)
        total_msmi = sum(r["msmi"] for r in pool_results)
        print(f"\nSplit: {split_name} ({len(pool_results)} pools)")
        print(f"  Total Introductions: {total_intros}")
        print(f"  Mutual Acceptance: {total_mutual} ({total_mutual/max(1, total_intros)*100:.1f}%)")
        print(f"  Dates Occurred:     {total_dates} ({total_dates/max(1, total_intros)*100:.1f}%)")
        print(f"  MSMI Achieved:      {total_msmi} ({total_msmi/max(1, total_intros)*100:.1f}%)")
        
    train_pools = results["train"]
    agg_missing = defaultdict(lambda: Counter())
    for r in train_pools:
        for k, counts in r["missingness"].items():
            for status, count in counts.items():
                agg_missing[k][status] += count
                
    print("\nTraining Split Field Status Rates (Sample of 1200 members):")
    for k in sorted(agg_missing):
        tot = sum(agg_missing[k].values())
        obs = agg_missing[k]["observed"] / tot * 100
        na = agg_missing[k]["not_asked"] / tot * 100
        dec = agg_missing[k]["declined"] / tot * 100
        print(f"  {k:25s}: Observed: {obs:4.1f}% | Not Asked: {na:4.1f}% | Declined: {dec:4.1f}%")

if __name__ == "__main__":
    main()
