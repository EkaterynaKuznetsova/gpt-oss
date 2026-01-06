import json
import statistics
from pathlib import Path

results_path = "data/ips/stage1_results_full.jsonl"
with open(results_path) as f:
    results = [json.loads(line) for line in f]

short = [r for r in results if r['prompt_token_len'] < 70]
medium = [r for r in results if 70 <= r['prompt_token_len'] < 120]
long = [r for r in results if r['prompt_token_len'] >= 120]

for name, group in [("Short (<70 tok)", short), ("Medium (70-120)", medium), ("Long (≥120)", long)]:
    if not group:
        continue
    L_vals = [r['L_val'] for r in group]
    quality_rate = sum(r['quality_passed'] for r in group) / len(group) * 100
    print(f"{name}: n={len(group)}, mean ℒ_val={statistics.mean(L_vals):.4f}, quality_pass={quality_rate:.1f}%")
