# Inverse Prompt Search (IPS) plan for GPT-OSS-20B (int4)

Goal: find short, diverse instructions `P` (7–10× shorter than `T`) such that deterministic inference with GPT-OSS-20B reproduces complex reference code `T`. Cover hypotheses H1–H4 under fixed inference (temperature=0, top_k=1, top_p=1, seed fixed) and limited VRAM (~10–12 GB, ~3.7B active params).

## Model + inference setup (fixed)
- Model: `openai/gpt-oss-20b` int4. Prefer vLLM or Metal backend; ensure Harmony format and stop tokens are used. Sampling strictly deterministic: `temperature=0`, `top_k=1`, `top_p=1`, `seed` fixed.
- Context: allow ≥5k tokens to fit `prefill(T) + P` and responses. Target `|T|=200–800` tokens, `|P|<=|T|/7..|T|/10`.
- Batch eval: expose a thin inference wrapper that accepts `(prompt_tokens, stop_tokens)` and returns completion tokens; record logprobs for loss metrics.

## Data: code targets T
- Source: GitHub trending repos, stars >1k, licenses MIT/Apache/BSD. Languages: Python first, then JS/Rust.
- Unit: function, class, small module, or single-file commit diff (later). Length 200–800 tokens after tokenization.
- Storage format (JSONL):
  ```json
  {"id": "owner/repo:path:commit", "language": "python", "code": "def ...", "metadata": {"license": "MIT", "len_tokens": 432}}
  ```
- No instructions included; comments allowed as-is.

## Metrics for matching M(P) ≈ T
- Token-level LCS / character edit distance; optionally normalized by |T|.
- AST equality (for Python: `ast.dump` canonicalization) and selective execution tests when safe.
- Logprob-based loss over T tokens given M(P) (teacher-forcing style if supported by backend).
- Copy penalty: similarity(P, T) via embedding cosine or n-gram overlap.
- Length penalty: `alpha * |P|`.

## Stage 1 — Manual sanity check (H1 smoke test)
- Sample 20 targets T. Hand-write strong instructions per T.
- Run deterministic inference once per P. Collect: exact/AST match, LCS, pass/fail on execution probes, stability across 3 repeated runs.
- If success rate is very low, deprioritize IPS for those T or simplify T.

## Stage 2 — GA-based IPS
- Representation: P is plain-text instruction; max_len = |T|/7..|T|/10.
- Init population: 100–300 prompts. Seeds include structured templates (“Write a Python function that …”, “Implement the following logic: …”, “Create a module that …”) plus noisy/shuffled text.
- Fitness(P):
  - `-L_code(M(P), T)` using AST/token LCS/exec tests.
  - `- alpha * len(P)` length penalty.
  - `- beta * similarity(P, T)` to discourage copy/paste.
- Selection: top 10–20%. Tournament or truncation.
- Variation:
  - Mutation: paraphrase via synonym swaps, deletions, shuffles, token-drop noise; constrain to length.
  - Crossover: splice halves/segments of two prompts; keep within max_len.
- Stopping: success if `L_code <= τ` or AST/execution match; else max generations/forward passes.
- Logged per T: generations to success, forward passes, best fitness, |P|, final loss; archive all P with scores.

## Stage 3 — RL-based IPS
- Environment: state = current prompt tokens; actions = add token, remove token, replace token; episode length 50–100 steps.
- Reward: `R = reward_match(M(P), T) - alpha*len(P) - beta*similarity(P, T)` with terminal bonus on exact/AST/execution match.
- Algorithms: PPO (Stable-Baselines3) and REINFORCE baseline; optional DPO for preference pairs of prompts.
- Rollout policy calls deterministic model; cache model evals to save compute when P repeats.
- Compare to GA: convergence speed, stability, diversity of discovered P, sensitivity to hyperparams.

## Stage 4 — Entropy/diversity analysis (H3)
- For fixed T, run GA/RL N times with different seeds/miners → {P_i}.
- Metrics: BLEU between prompts; embedding cosine distances; clustering (e.g., k-means/silhouette); manual inspection of top-k diverse prompts.
- Goal: demonstrate multiple semantically distinct P that all yield T.

## Stage 5 — Dataset quality (H4)
- Build instruct dataset of (P, T). Fine-tune a small model (e.g., 7B) and compare to baselines (Alpaca, Self-Instruct, GPT-generated).
- Eval: instruction following, generalization to held-out T, robustness to overfitting (train/val gap, memorization checks).

## Stage 6 — Negative tests
- Prompts that are: too short, random noise, copy/paste of T, explicit “print this code”. They must fail the matcher and not be accepted.

## Immediate next steps
- Confirm inference backend for int4 (vLLM or Metal) and prepare a deterministic wrapper.
- Build minimal metrics library (token LCS + AST match) and a small target sample loader.
- Run Stage 1 manual prompts on 20 T to gauge feasibility and set thresholds τ for GA/RL.
- Prototype GA search on 1–2 T with small population to measure compute/forward-pass costs.
