import json
import statistics

results_path = "data/ips/stage1_results_full.jsonl"
with open(results_path) as f:
    results = [json.loads(line) for line in f]

low_copy = [r for r in results if r['similarity'] < 0.3]
mid_copy = [r for r in results if 0.3 <= r['similarity'] < 0.7]
high_copy = [r for r in results if r['similarity'] >= 0.7]

for name, group in [("Low copy (<0.3)", low_copy), ("Mid copy (0.3-0.7)", mid_copy), ("High copy (≥0.7)", high_copy)]:
    if not group:
        continue
    L_vals = [r['L_val'] for r in group]
    quality_rate = sum(r['quality_passed'] for r in group) / len(group) * 100
    print(f"{name}: n={len(group)}, mean ℒ_val={statistics.mean(L_vals):.4f}, quality_pass={quality_rate:.1f}%")
