from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import sys

# Ensure local imports when running directly
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ips.dataset import iter_jsonl


def _format_signature(args: ast.arguments) -> str:
    parts = []
    # positional-only
    if getattr(args, 'posonlyargs', None):
        parts += [a.arg for a in args.posonlyargs]
        if parts:
            parts[-1] = parts[-1] + ", /"
    # positional/kw
    parts += [a.arg for a in args.args]
    # varargs
    if args.vararg:
        parts.append("*" + args.vararg.arg)
    # kwonly
    if args.kwonlyargs:
        if not args.vararg:
            parts.append("*")
        parts += [a.arg for a in args.kwonlyargs]
    # varkw
    if args.kwarg:
        parts.append("**" + args.kwarg.arg)
    return "(" + ", ".join(parts) + ")"


def _first_top_level_entity(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ("function", node)
        if isinstance(node, ast.ClassDef):
            return ("class", node)
    return None


def draft_prompt_for(code: str, language: str) -> str:
    if language.lower() != "python":
        return (
            "Implement the requested code artifact exactly and deterministically. "
            "Follow the specified names and signatures. No extra output."
        )
    ent = _first_top_level_entity(code)
    if not ent:
        return (
            "Implement the described Python module deterministically with the required API. "
            "Keep names and behavior exact; do not add extraneous text."
        )
    kind, node = ent
    if kind == "function":
        sig = _format_signature(node.args)
        name = node.name
        return (
            f"Write a Python function `{name}{sig}` that reproduces the intended behavior exactly, "
            "including edge cases and error handling. Use only the standard library. Keep the exact "
            "signature and names. Deterministic output, no prints or logs other than required. "
            "Add a concise docstring."
        )
    else:
        # class
        name = node.name
        methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        methods_text = ", ".join(f"`{m}()`" for m in methods[:6])
        return (
            f"Create a Python class `{name}` implementing the same public API and behavior. "
            f"Include methods: {methods_text}. Keep exact method names/signatures and deterministic behavior. "
            "Use only the standard library, no extra output. Provide minimal docstrings."
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Draft human-like instruction prompts for Stage 1")
    parser.add_argument("--targets", type=Path, default=Path("data/ips/targets.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/ips/manual_prompts.jsonl"))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    out = []
    for i, t in enumerate(iter_jsonl(args.targets)):
        if i >= args.limit:
            break
        prompt = draft_prompt_for(t.code, t.language)
        out.append({
            "id": t.id,
            "language": t.language,
            "code": t.code,
            "metadata": t.metadata,
            "prompt": prompt + "\n\nDetails: [briefly specify behavior, inputs, outputs, constraints; no code]",
            "notes": "Review and refine this prompt manually for fidelity.",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} drafted prompts to {args.output}")


if __name__ == "__main__":
    main()
