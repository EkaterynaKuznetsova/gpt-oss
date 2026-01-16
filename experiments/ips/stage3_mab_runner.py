from __future__ import annotations
import random
import nltk
from nltk.corpus import wordnet
import json
from pathlib import Path
from typing import List, Dict, Any
from copy import deepcopy
from collections import defaultdict, Counter

from experiments.ips.dataset import iter_jsonl, write_jsonl
from experiments.ips.metrics import normalized_similarity, python_ast_equal
from experiments.ips.infer import InferenceConfig, DeterministicInference

# --- MAB Hyperparameters ---
ITERATIONS = 1000  
EPSILON = 0.1  # Epsilon-greedy: 10% random, 90% greedy
QUALITY_THRESHOLD = 0.3
INITIAL_TEMP = 0.0  # Initial temp for epsilon-greedy

# --- Utility functions (from stage2_ga_runner.py) ---
def load_targets(path: Path) -> List[Dict[str, Any]]:
    return [t.__dict__ for t in iter_jsonl(path)]

def load_prompts(path: Path) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def save_results(path: Path, results: List[Dict[str, Any]]):
    write_jsonl(path, results)

# --- Action Functions ---
def synonymize_word(word: str) -> str:
    synsets = wordnet.synsets(word)
    if not synsets:
        return word
    lemmas = [lemma.name().replace('_', ' ') for syn in synsets for lemma in syn.lemmas() if lemma.name().lower() != word.lower()]
    if not lemmas:
        return word
    return random.choice(lemmas)

def apply_action(prompt: str, action: str) -> str:
    """Apply one action to the prompt"""
    words = prompt.split()
    if len(words) < 3:
        return prompt
    
    original_prompt = prompt
    
    if action == 'drop' and len(words) > 3:
        idx = random.randrange(len(words))
        del words[idx]
    elif action == 'swap' and len(words) > 3:
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    elif action == 'insert':
        idx = random.randrange(len(words) + 1)
        words.insert(idx, random.choice(words))
    elif action == 'synonymize' and len(words) > 3:
        idx = random.randrange(len(words))
        words[idx] = synonymize_word(words[idx])
    else:
        return prompt
    
    mutated = ' '.join(words)
    
    # Repair step
    if len(words) < 10 or not any(c.isalpha() for c in mutated):
        return original_prompt
    word_counts = Counter(words)
    if any(count > 3 for count in word_counts.values()):
        return original_prompt
    template_words = {'explanation', 'output', 'code', 'docstring', 'only', 'exactly', 'before', 'after', 'text'}
    template_count = sum(1 for w in words if w.lower() in template_words)
    if template_count > 5:
        return original_prompt
    
    return mutated

def evaluate_prompt(prompt: str, target: Dict[str, Any], infer: DeterministicInference) -> Dict[str, Any]:
    """Evaluate prompt"""
    generated, *_ = infer.generate(prompt)
    sim = normalized_similarity(generated, target['code'])
    ast_eq = python_ast_equal(generated, target['code'])
    
    # L_val: combination of similarity and AST match
    if ast_eq:
        l_val = 1.0 - sim
    else:
        l_val = 1.0 - (sim * 0.5)  # More strict penalty for incorrect code (min 0.5)
    
    # Two-stage filtering: require AST match + low L_val
    quality_passed = (ast_eq == True) and (l_val <= QUALITY_THRESHOLD)
    
    return {
        'prompt': prompt,
        'generated_code': generated,
        'L_val': l_val,
        'quality_passed': quality_passed,
        'similarity': sim,
        'ast_equal': ast_eq
    }

# --- MAB Algorithm ---
class MABOptimizer:
    def __init__(self, actions: List[str], epsilon: float = 0.1):
        self.actions = actions
        self.epsilon = epsilon
        
        # Track for each action: number of trials and sum of rewards
        self.action_counts = defaultdict(int)
        self.action_rewards = defaultdict(float)
        self.action_successes = defaultdict(int)  # Number of successful applications
    
    def get_action_reward(self, action: str) -> float:
        """Average reward for an action (avoiding division by 0)"""
        if self.action_counts[action] == 0:
            return 0.0
        return self.action_rewards[action] / self.action_counts[action]
    
    def select_action(self) -> str:
        """Epsilon-greedy: 90% best action, 10% random"""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        # Greedy: choose the action with the best average reward
        best_action = max(self.actions, key=lambda a: self.get_action_reward(a))
        return best_action
    
    def update(self, action: str, reward: float, success: bool = False):
        """Update action statistics"""
        self.action_counts[action] += 1
        self.action_rewards[action] += reward
        if success:
            self.action_successes[action] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics for all actions"""
        stats = {}
        for action in self.actions:
            count = self.action_counts[action]
            if count > 0:
                avg_reward = self.action_rewards[action] / count
                success_rate = self.action_successes[action] / count
            else:
                avg_reward = 0.0
                success_rate = 0.0
            stats[action] = {
                'count': count,
                'avg_reward': avg_reward,
                'success_rate': success_rate,
                'successes': self.action_successes[action]
            }
        return stats

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3: MAB Prompt Optimizer")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--backend', type=str, default='vllm', help='Model backend: vllm, torch, triton, etc.')
    parser.add_argument('--prompts', type=Path, default=Path('data/ips/manual_prompts.jsonl'), help='Path to initial prompts JSONL')
    parser.add_argument('--targets', type=Path, default=Path('data/ips/targets.jsonl'))
    parser.add_argument('--iterations', type=int, default=ITERATIONS)
    parser.add_argument('--epsilon', type=float, default=EPSILON)
    parser.add_argument('--quality-threshold', type=float, default=QUALITY_THRESHOLD)
    parser.add_argument('--output', type=Path, default=Path('data/ips/stage3_mab_results.jsonl'))
    parser.add_argument('--max-new-tokens', type=int, default=512, help='Maximum number of new tokens to generate')
    args = parser.parse_args()

    # Load targets and initialize inference
    targets = load_targets(args.targets)
    target = targets[0]  # Optimize for the first target only
    infer_cfg = InferenceConfig(
        checkpoint=args.checkpoint,
        backend=args.backend,
        max_new_tokens=args.max_new_tokens
    )
    infer = DeterministicInference(infer_cfg)

    # Load initial prompts
    prompt_data = load_prompts(args.prompts)
    initial_prompts = [p['prompt'] for p in prompt_data if p['prompt'].strip()]
    
    # Initialize MAB
    actions = ['drop', 'swap', 'insert', 'synonymize']
    mab = MABOptimizer(actions, epsilon=args.epsilon)
    
    # Start with random initial prompt
    current_prompt = random.choice(initial_prompts)
    current_eval = evaluate_prompt(current_prompt, target, infer)
    best_eval = deepcopy(current_eval)
    
    results = [current_eval]
    
    print(f"Starting MAB optimization for {args.iterations} iterations")
    print(f"Initial L_val: {current_eval['L_val']:.4f}")
    
    for iteration in range(args.iterations):
        # Select action
        action = mab.select_action()
        
        # Apply action
        new_prompt = apply_action(current_prompt, action)
        
        # Evaluate
        new_eval = evaluate_prompt(new_prompt, target, infer)
        
        # Compute reward: improvement in L_val (negative = good)
        reward = current_eval['L_val'] - new_eval['L_val']  # Positive if improved
        success = reward > 0
        
        # Update MAB
        mab.update(action, reward, success=success)
        
        # Update best
        if new_eval['L_val'] < best_eval['L_val']:
            best_eval = deepcopy(new_eval)
            results.append(best_eval)
            print(f"Iter {iteration+1}/{args.iterations}: {action:10} | New best L_val: {best_eval['L_val']:.4f} | Quality: {best_eval['quality_passed']}")
        
        # Move to new prompt (with some probability, else stay)
        if success or random.random() < 0.3:
            current_prompt = new_prompt
            current_eval = new_eval
        
        # Log every 100 iterations
        if (iteration + 1) % 100 == 0:
            stats = mab.get_stats()
            print(f"\nStats at iteration {iteration+1}:")
            for act, st in stats.items():
                print(f"  {act:10}: count={st['count']:3d}, avg_reward={st['avg_reward']:+.4f}, success_rate={st['success_rate']:.2%}")
    
    # Save results
    save_results(args.output, results)
    
    # Final stats
    stats = mab.get_stats()
    print(f"\n=== Final MAB Statistics ===")
    for act, st in stats.items():
        print(f"{act:10}: count={st['count']:4d}, avg_reward={st['avg_reward']:+.4f}, success_rate={st['success_rate']:.2%}, successes={st['successes']}")
    
    print(f"\nBest L_val achieved: {best_eval['L_val']:.4f}")
    print(f"Best quality_passed: {best_eval['quality_passed']}")
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
