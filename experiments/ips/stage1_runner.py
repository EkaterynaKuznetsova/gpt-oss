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
    parser = argparse.ArgumentParser(description="Stage 1 (IPE Verification): run manual prompts deterministically and score")
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
    parser.add_argument("--quality-threshold", type=float, default=0.1, help="Quality threshold τ_quality for ℒ_val (IPE Verification)")
    parser.add_argument("--length-factor", type=float, default=0.15, help="Max prompt length factor: len(P) < factor × len(T)")
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
    quality_passed = 0
    length_passed = 0
    ast_match = 0
    
    for row in rows:
        prompt = row.get("prompt", "")
        if not prompt:
            print(f"Skipping {row.get('id')} because prompt is empty")
            continue
        target_code = row.get("code", "")
        
        # Token counts
        prompt_tokens = len(runner.tokenizer.encode(prompt))
        target_tokens = len(runner.tokenizer.encode(target_code))
        
        # Length constraint check
        max_prompt_len = int(args.length_factor * target_tokens)
        length_ok = prompt_tokens <= max_prompt_len
        length_passed += int(length_ok)
        
        t0 = perf_counter()
        # Generate with log-probs for auxiliary metrics
        generated_text, prompt_tok_len, gen_tok_len, gen_logprobs = runner.generate(
            prompt, return_logprobs=True
        )
        
        # Compute ℒ_val (IPE Verification metric)
        L_val = runner.compute_target_loss(prompt, target_code)
        quality_ok = L_val <= args.quality_threshold
        quality_passed += int(quality_ok)
        
        dt = perf_counter() - t0

        # Auxiliary metrics
        sim = normalized_similarity(generated_text, target_code)
        ast_eq = python_ast_equal(generated_text, target_code)
        ast_match += int(ast_eq)

        results.append(
            {
                "id": row.get("id"),
                "prompt": prompt,
                "target_code": target_code,
                "generated_code": generated_text,
                # IPE Verification metrics
                "L_val": L_val,
                "quality_passed": quality_ok,
                "length_constraint_passed": length_ok,
                # Auxiliary metrics
                "ast_equal": ast_eq,
                "similarity": sim,
                "prompt_token_len": prompt_tokens,
                "target_token_len": target_tokens,
                "generated_token_len": gen_tok_len,
                "latency_sec": dt,
                # Config
                "backend": args.backend,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": None if args.top_k < 0 else args.top_k,
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
                "quality_threshold": args.quality_threshold,
                "length_factor": args.length_factor,
            }
        )

    write_jsonl(args.output, results)
    total = len(results)
    print(f"IPE Verification complete: {total} prompts")
    print(f"  Quality passed (ℒ_val ≤ {args.quality_threshold}): {quality_passed}/{total}")
    print(f"  Length constraint passed: {length_passed}/{total}")
    print(f"  AST match: {ast_match}/{total}")
    print(f"Results → {args.output}")


if __name__ == "__main__":
    main()
