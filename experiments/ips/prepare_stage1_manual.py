from __future__ import annotations

import json
from pathlib import Path

from .dataset import sample_n, write_jsonl


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sample targets and create a Stage 1 manual prompts file")
    parser.add_argument("--targets", type=Path, default=Path("data/ips/targets.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/ips/manual_prompts.jsonl"))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample = sample_n(args.targets, n=args.n, seed=args.seed)
    rows = []
    for t in sample:
        rows.append({
            "id": t.id,
            "language": t.language,
            "code": t.code,
            "metadata": t.metadata,
            "prompt": "",  # fill manually with a strong instruction
            "notes": "",
        })
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} manual prompt rows to {args.output}")


if __name__ == "__main__":
    main()
