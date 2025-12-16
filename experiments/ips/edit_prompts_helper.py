"""
Helper to merge user-edited prompts back into proper JSONL format.
Usage: manually edit prompts in a simple text file, then merge them back.
"""
from __future__ import annotations

import json
from pathlib import Path


def extract_prompts_for_editing(jsonl_path: Path, txt_path: Path):
    """Extract prompts into an editable text file with separators."""
    with jsonl_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    
    with txt_path.open("w", encoding="utf-8") as out:
        for i, ln in enumerate(lines):
            try:
                obj = json.loads(ln)
                tid = obj.get("id", f"entry-{i}")
                prompt = obj.get("prompt", "")
                out.write(f"=== PROMPT {i+1}: {tid} ===\n")
                out.write(prompt.strip() + "\n")
                out.write("=== END ===\n\n")
            except Exception as e:
                print(f"Skip line {i+1}: {e}")
    print(f"Extracted prompts to {txt_path}. Edit them, then run merge.")


def merge_edited_prompts(jsonl_path: Path, txt_path: Path, output_path: Path):
    """Merge edited prompts back into JSONL."""
    # Read original JSONL
    with jsonl_path.open("r", encoding="utf-8") as f:
        objects = [json.loads(ln) for ln in f if ln.strip()]
    
    # Read edited prompts
    text = txt_path.read_text(encoding="utf-8")
    sections = text.split("=== PROMPT ")
    prompts = []
    for sec in sections[1:]:
        if "=== END ===" in sec:
            header, rest = sec.split("\n", 1)
            prompt_text, _ = rest.split("=== END ===", 1)
            prompts.append(prompt_text.strip())
    
    if len(prompts) != len(objects):
        print(f"Warning: {len(prompts)} prompts vs {len(objects)} objects")
    
    # Merge
    for i, obj in enumerate(objects):
        if i < len(prompts):
            obj["prompt"] = prompts[i]
    
    # Write proper JSONL
    with output_path.open("w", encoding="utf-8") as out:
        for obj in objects:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    
    print(f"Merged {len(objects)} objects to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["extract", "merge"])
    parser.add_argument("--jsonl", type=Path, default=Path("data/ips/manual_prompts_backup.jsonl"))
    parser.add_argument("--txt", type=Path, default=Path("data/ips/prompts_editable.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/ips/manual_prompts.jsonl"))
    args = parser.parse_args()
    
    if args.action == "extract":
        extract_prompts_for_editing(args.jsonl, args.txt)
    else:
        merge_edited_prompts(args.jsonl, args.txt, args.output)


if __name__ == "__main__":
    main()
