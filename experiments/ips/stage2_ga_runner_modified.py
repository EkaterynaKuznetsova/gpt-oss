from __future__ import annotations
import random
import nltk
from nltk.corpus import wordnet
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

successful_mutations = []  # Global list of successful mutations

# --- GA Operators ---
def synonymize_word(word: str) -> str:
    synsets = wordnet.synsets(word)
    if not synsets:
        return word
    lemmas = [lemma.name().replace('_', ' ') for syn in synsets for lemma in syn.lemmas() if lemma.name().lower() != word.lower()]
    if not lemmas:
        return word
    return random.choice(lemmas)

def mutate_prompt(prompt: str) -> str:
    words = prompt.split()
    if not words:
        return prompt
    original_prompt = prompt
    if len(words) > 120:
        n_remove = max(1, len(words) // 10)
        idxs = random.sample(range(len(words)), n_remove)
        words = [w for i, w in enumerate(words) if i not in idxs]
    elif len(words) < 50:
        templates = [
            "No explanation.",
            "Output exactly:",
            "No text before/after. Only code.",
            "Use only the standard library.",
            "Provide minimal docstrings."
        ]
        words += random.choices(templates, k=2)
    else:
        op = random.choice(['drop', 'swap', 'insert', 'synonymize'])
        if op == 'drop' and len(words) > 3:
            idx = random.randrange(len(words))
            del words[idx]
        elif op == 'swap' and len(words) > 3:
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
        elif op == 'insert':
            idx = random.randrange(len(words)+1)
            words.insert(idx, random.choice(words))
        elif op == 'synonymize' and len(words) > 3:
            idx = random.randrange(len(words))
            words[idx] = synonymize_word(words[idx])
    mutated = ' '.join(words)
    # Repair step: if the prompt became too short (<10 words), contains meaningless repetitions, or many templates — return the original
    if len(words) < 10 or not any(c.isalpha() for c in mutated):
        return original_prompt
    # Check for excessive repetition of a single word (>3 times)
    from collections import Counter
    word_counts = Counter(words)
    if any(count > 3 for count in word_counts.values()):
        return original_prompt
    # Check for excessive templates (>5 from the list)
    template_words = {'explanation', 'output', 'code', 'docstring', 'only', 'exactly', 'before', 'after', 'text'}
    template_count = sum(1 for w in words if w.lower() in template_words)
    if template_count > 5:
        return original_prompt
    return mutated

def crossover_prompt(p1: str, p2: str) -> str:
    # Crossover by task type: find task keywords and mix only compatible fragments
    TASK_KEYWORDS = [
        "class", "function", "method", "property", "session", "cookie", "serialize", "header", "python code"
    ]
    def extract_task_chunks(words):
        chunks = []
        current = []
        for w in words:
            current.append(w)
            if any(kw in w.lower() for kw in TASK_KEYWORDS):
                if current:
                    chunks.append(list(current))
                    current = []
        if current:
            chunks.append(current)
        return chunks if chunks else [words]

    w1, w2 = p1.split(), p2.split()
    if not w1 or not w2:
        return p1
    chunks1 = extract_task_chunks(w1)
    chunks2 = extract_task_chunks(w2)
    # Take 1-2 compatible fragments from each
    selected = []
    if chunks1:
        selected.append(random.choice(chunks1))
    if chunks2:
        selected.append(random.choice(chunks2))
    # Add random fragments if few
    all_chunks = chunks1 + chunks2
    while len(selected) < random.randint(2, 4) and all_chunks:
        selected.append(random.choice(all_chunks))
    child_words = []
    for chunk in selected:
        child_words.extend(chunk)
    return ' '.join(child_words)

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
    # L_val: combination of similarity and AST match — if AST matches, L_val is low
    if ast_eq:
        l_val = 1.0 - sim  # If AST matches, use similarity
    else:
        l_val = 1.0 - (sim * 0.3)  # If AST does not match, penalize more (min 0.7)
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

    # Initialize population: take manual prompts + their variants
    manual_prompts = load_prompts(args.manual_prompts)
    base_prompts = [p['prompt'] for p in manual_prompts if p['prompt'].strip()]
    population = []
    # 50% — original prompts
    population.extend([{'prompt': p} for p in random.choices(base_prompts, k=args.population_size//2)])
    # 50% — slightly mutated variants
    for _ in range(args.population_size - len(population)):
        base = random.choice(base_prompts)
        mutated = mutate_prompt(base)
        population.append({'prompt': mutated})

    for gen in range(args.generations):
        print(f"=== Generation {gen+1}/{args.generations} ===")
        # Evaluate all
        evaluated = [evaluate_prompt(ind['prompt'], target, infer) for ind in population]
        # Log successful mutations (quality_passed)
        for ind in evaluated:
            if ind['quality_passed']:
                successful_mutations.append(ind['prompt'])
        # Select elite
        n_elite = max(1, int(args.population_size * ELITE_FRAC))
        elite = select_elite(evaluated, n_elite)
        print(f"Best L_val: {elite[0]['L_val']:.4f} | Quality passed: {elite[0]['quality_passed']}")
        # Generate next population
        next_pop = deepcopy(elite)
        while len(next_pop) < args.population_size:
            if random.random() < args.mutation_rate:
                parent = random.choice(elite)
                # 20% chance to take a successful mutation
                if successful_mutations and random.random() < 0.2:
                    mutated = random.choice(successful_mutations)
                else:
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
