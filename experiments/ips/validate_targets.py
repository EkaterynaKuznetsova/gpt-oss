from __future__ import annotations

import json
from pathlib import Path
from typing import Counter

from gpt_oss.tokenizer import get_tokenizer


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate IPS targets JSONL")
    parser.add_argument("--path", type=Path, default=Path("data/ips/targets.jsonl"))
    parser.add_argument("--min-tok", type=int, default=200)
    parser.add_argument("--max-tok", type=int, default=800)
    args = parser.parse_args()

    tok = get_tokenizer()
    total = 0
    bad_json = 0
    out_of_range = 0
    mismatched_len = 0
    langs: dict[str, int] = {}
    lic: dict[str, int] = {}
    min_len = 10**9
    max_len = 0
    sum_len = 0

    with args.path.open("r", encoding="utf-8") as f:
        for i, ln in enumerate(f, 1):
            if not ln.strip():
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                bad_json += 1
                continue
            total += 1
            code = obj.get("code", "")
            lang = obj.get("language", "")
            langs[lang] = langs.get(lang, 0) + 1
            meta = obj.get("metadata", {})
            lic_name = meta.get("license", "UNKNOWN")
            lic[lic_name] = lic.get(lic_name, 0) + 1

            l = len(tok.encode(code))
            min_len = min(min_len, l)
            max_len = max(max_len, l)
            sum_len += l
            if not (args.min_tok <= l <= args.max_tok):
                out_of_range += 1
            if isinstance(meta.get("len_tokens"), int) and meta["len_tokens"] != l:
                mismatched_len += 1

    print(f"total={total} bad_json={bad_json} out_of_range={out_of_range} mismatched_len={mismatched_len}")
    if total:
        print(f"tokens: min={min_len} max={max_len} avg={sum_len/total:.1f}")
    print("languages:", langs)
    print("licenses:", lic)


if __name__ == "__main__":
    main()
