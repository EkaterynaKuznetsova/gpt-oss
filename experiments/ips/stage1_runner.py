from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from .infer import DeterministicInference, InferenceConfig
from .metrics import normalized_similarity, python_ast_equal
from .dataset import write_jsonl


def load_manual_prompts(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            rows.append(json.loads(ln))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Stage 1: run manual prompts deterministically and score")
    parser.add_argument("--checkpoint", required=True, help="Path to GPT-OSS checkpoint (e.g., int4 safetensors)")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "triton", "torch"], help="Inference backend")
    parser.add_argument("--manual-prompts", type=Path, default=Path("data/ips/manual_prompts.jsonl"), help="Path to manual prompts JSONL")
    parser.add_argument("--output", type=Path, default=Path("data/ips/stage1_results.jsonl"), help="Where to write results JSONL")
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="Tensor parallel size (vLLM)")
    parser.add_argument("--context-length", type=int, default=4096, help="Context length for Triton backend")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p (vLLM)")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k (vLLM); use -1 for None")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (vLLM)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens to sample per prompt")
    args = parser.parse_args()

    config = InferenceConfig(
        checkpoint=args.checkpoint,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
        context_length=args.context_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=None if args.top_k < 0 else args.top_k,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )
    runner = DeterministicInference(config)

    rows = load_manual_prompts(args.manual_prompts)
    results = []
    ast_match = 0
    for row in rows:
        prompt = row.get("prompt", "")
        if not prompt:
            print(f"Skipping {row.get('id')} because prompt is empty")
            continue
        target_code = row.get("code", "")
        t0 = perf_counter()
        generated_text, prompt_tok_len, gen_tok_len = runner.generate(prompt)
        dt = perf_counter() - t0

        sim = normalized_similarity(generated_text, target_code)
        ast_eq = python_ast_equal(generated_text, target_code)
        ast_match += int(ast_eq)

        results.append(
            {
                "id": row.get("id"),
                "prompt": prompt,
                "target_code": target_code,
                "generated_code": generated_text,
                "similarity": sim,
                "ast_equal": ast_eq,
                "prompt_token_len": prompt_tok_len,
                "generated_token_len": gen_tok_len,
                "latency_sec": dt,
                "backend": args.backend,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": None if args.top_k < 0 else args.top_k,
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
            }
        )

    write_jsonl(args.output, results)
    total = len(results)
    print(f"Ran {total} prompts. AST match {ast_match}/{total}. Wrote {args.output}")


if __name__ == "__main__":
    main()
