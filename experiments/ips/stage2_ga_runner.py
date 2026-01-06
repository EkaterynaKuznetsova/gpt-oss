from __future__ import annotations
import random
import json
from pathlib import Path
from typing import List, Dict, Any
from copy import deepcopy

from experiments.ips.dataset import iter_jsonl, write_jsonl
from experiments.ips.metrics import normalized_similarity, python_ast_equal
from experiments.ips.infer import InferenceConfig, DeterministicInference

# --- GA Hyperparameters ---
POPULATION_SIZE = 50
GENERATIONS = 20
MUTATION_RATE = 0.3
ELITE_FRAC = 0.1
QUALITY_THRESHOLD = 0.3

# --- Utility functions ---
def load_targets(path: Path) -> List[Dict[str, Any]]:
    return [t.__dict__ for t in iter_jsonl(path)]

def load_prompts(path: Path) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def save_population(path: Path, population: List[Dict[str, Any]]):
    write_jsonl(path, population)

# --- GA Operators ---
def mutate_prompt(prompt: str) -> str:
    # Simple mutation: random word drop, swap, or insert
    words = prompt.split()
    if not words:
        return prompt
    op = random.choice(['drop', 'swap', 'insert'])
    if op == 'drop' and len(words) > 3:
        idx = random.randrange(len(words))
        del words[idx]
    elif op == 'swap' and len(words) > 3:
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    elif op == 'insert':
        idx = random.randrange(len(words)+1)
        words.insert(idx, random.choice(words))
    return ' '.join(words)

def crossover_prompt(p1: str, p2: str) -> str:
    # Single-point crossover
    w1, w2 = p1.split(), p2.split()
    if not w1 or not w2:
        return p1
    cut1 = random.randint(1, len(w1))
    cut2 = random.randint(1, len(w2))
    return ' '.join(w1[:cut1] + w2[cut2:])

def evaluate_prompt(prompt: str, target: Dict[str, Any], infer: DeterministicInference) -> Dict[str, Any]:
    # Run model generation and compute metrics
    # For simplicity, only use the first target (single-task GA)
    input_data = {
        'prompt': prompt,
        'language': target['language'],
        'metadata': target.get('metadata', {})
    }
    # Generate code
    generated, *_ = infer.generate(prompt)
    # Compute metrics
    sim = normalized_similarity(generated, target['code'])
    ast_eq = python_ast_equal(generated, target['code'])
    l_val = 1.0 - sim  # Placeholder: replace with real loss if available
    quality_passed = l_val <= QUALITY_THRESHOLD
    return {
        'prompt': prompt,
        'generated_code': generated,
        'L_val': l_val,
        'quality_passed': quality_passed,
        'similarity': sim,
        'ast_equal': ast_eq
    }

def select_elite(population: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return sorted(population, key=lambda x: x['L_val'])[:n]

# --- Main GA Loop ---
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2: GA Prompt Optimizer")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--backend', type=str, default='vllm', help='Model backend: vllm, torch, triton, etc.')
    parser.add_argument('--manual-prompts', type=Path, default=Path('data/ips/manual_prompts.jsonl'), help='Path to manual prompts JSONL')
    parser.add_argument('--targets', type=Path, default=Path('data/ips/targets.jsonl'))
    parser.add_argument('--population-size', type=int, default=POPULATION_SIZE)
    parser.add_argument('--generations', type=int, default=GENERATIONS)
    parser.add_argument('--mutation-rate', type=float, default=MUTATION_RATE)
    parser.add_argument('--quality-threshold', type=float, default=QUALITY_THRESHOLD)
    parser.add_argument('--output', type=Path, default=Path('data/ips/stage2_ga_results.jsonl'))
    parser.add_argument('--max-new-tokens', type=int, default=512, help='Maximum number of new tokens to generate')
    args = parser.parse_args()

    # Load targets and initialize inference
    targets = load_targets(args.targets)
    target = targets[0]  # For now, optimize for the first target only
    infer_cfg = InferenceConfig(
        checkpoint=args.checkpoint,
        backend=args.backend,
        max_new_tokens=args.max_new_tokens
    )
    infer = DeterministicInference(infer_cfg)

    # Initialize population (randomly mutate manual prompts)
    manual_prompts = load_prompts(args.manual_prompts)
    base_prompts = [p['prompt'] for p in manual_prompts if p['prompt'].strip()]
    population = [{'prompt': p} for p in random.choices(base_prompts, k=args.population_size)]

    for gen in range(args.generations):
        print(f"=== Generation {gen+1}/{args.generations} ===")
        # Evaluate all
        evaluated = [evaluate_prompt(ind['prompt'], target, infer) for ind in population]
        # Select elite
        n_elite = max(1, int(args.population_size * ELITE_FRAC))
        elite = select_elite(evaluated, n_elite)
        print(f"Best L_val: {elite[0]['L_val']:.4f} | Quality passed: {elite[0]['quality_passed']}")
        # Generate next population
        next_pop = deepcopy(elite)
        while len(next_pop) < args.population_size:
            if random.random() < args.mutation_rate:
                parent = random.choice(elite)
                mutated = mutate_prompt(parent['prompt'])
                next_pop.append({'prompt': mutated})
            else:
                p1, p2 = random.sample(elite, 2)
                child = crossover_prompt(p1['prompt'], p2['prompt'])
                next_pop.append({'prompt': child})
        population = next_pop
    # Final evaluation and save
    final_evaluated = [evaluate_prompt(ind['prompt'], target, infer) for ind in population]
    save_population(args.output, final_evaluated)
    print(f"Saved final population to {args.output}")

if __name__ == "__main__":
    main()
