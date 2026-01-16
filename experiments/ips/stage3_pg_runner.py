from __future__ import annotations
import random
import nltk
from nltk.corpus import wordnet
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from copy import deepcopy
from collections import Counter
import numpy as np

from experiments.ips.dataset import iter_jsonl, write_jsonl
from experiments.ips.metrics import normalized_similarity, python_ast_equal
from experiments.ips.infer import InferenceConfig, DeterministicInference

# --- PG Hyperparameters ---
ITERATIONS = 5000
LEARNING_RATE = 0.01
GAMMA = 0.99  # Discount factor
QUALITY_THRESHOLD = 0.3
EMBEDDING_DIM = 32  # Size of prompt embedding

# --- Utility functions ---
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

def prompt_to_embedding(prompt: str, vocab_size: int = 1000, embedding_dim: int = EMBEDDING_DIM) -> np.ndarray:
    """
    Simple embedding for the prompt:
    - Take the hash of each word modulo vocab_size
    - Sum the embeddings of all words
    - Normalize
    """
    words = prompt.split()
    embedding = np.zeros(embedding_dim)
    
    for word in words:
        word_hash = hash(word) % vocab_size
        # Generate fixed embedding for this hash
        np.random.seed(word_hash)
        embedding += np.random.randn(embedding_dim) * 0.1
    
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding

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

# --- Policy Network (simple linear) ---
class PolicyNetwork:
    def __init__(self, input_dim: int = EMBEDDING_DIM, num_actions: int = 4, learning_rate: float = LEARNING_RATE):
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        
        # Weights: input_dim x num_actions
        self.weights = np.random.randn(input_dim, num_actions) * 0.01
        self.bias = np.zeros(num_actions)
        
        # History for REINFORCE
        self.trajectory = []  # List of (state_embedding, action_idx, reward, log_prob)
    
    def forward(self, state_embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass: state -> action logits -> probabilities
        Returns: (action_probs, logits)
        """
        logits = np.dot(state_embedding, self.weights) + self.bias
        # Softmax
        logits = logits - np.max(logits)  # Numerical stability
        exp_logits = np.exp(logits)
        action_probs = exp_logits / np.sum(exp_logits)
        return action_probs, logits
    
    def select_action(self, state_embedding: np.ndarray) -> Tuple[int, float]:
        """
        Select action according to policy
        Returns: (action_idx, log_prob)
        """
        action_probs, _ = self.forward(state_embedding)
        action_idx = np.random.choice(self.num_actions, p=action_probs)
        log_prob = np.log(action_probs[action_idx] + 1e-8)
        return action_idx, log_prob
    
    def save_trajectory_step(self, state_embedding: np.ndarray, action_idx: int, log_prob: float, reward: float):
        self.trajectory.append((state_embedding, action_idx, reward, log_prob))
    
    def update_policy(self):
        """
        REINFORCE update:
        For each step in the trajectory: loss = -log_prob * G
        where G = discounted cumulative reward
        """
        if len(self.trajectory) == 0:
            return
        
        # Compute discounted cumulative rewards in reverse order
        returns = []
        G = 0
        for state_emb, action_idx, reward, log_prob in reversed(self.trajectory):
            G = reward + GAMMA * G
            returns.insert(0, G)
        
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # Update weights
        for i, (state_emb, action_idx, reward, log_prob) in enumerate(self.trajectory):
            G = returns[i]
            
            # Gradient: -log_prob * G * state_embedding
            gradient = -G * state_emb  # Shape: (input_dim,)
            
            # Update weights for the selected action
            self.weights[:, action_idx] += self.learning_rate * gradient
        
        self.trajectory = []

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3: Policy Gradient Prompt Optimizer")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--backend', type=str, default='vllm', help='Model backend: vllm, torch, triton, etc.')
    parser.add_argument('--prompts', type=Path, default=Path('data/ips/manual_prompts.jsonl'), help='Path to initial prompts JSONL')
    parser.add_argument('--targets', type=Path, default=Path('data/ips/targets.jsonl'))
    parser.add_argument('--iterations', type=int, default=ITERATIONS)
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--quality-threshold', type=float, default=QUALITY_THRESHOLD)
    parser.add_argument('--output', type=Path, default=Path('data/ips/stage3_pg_results.jsonl'))
    parser.add_argument('--max-new-tokens', type=int, default=1024, help='Maximum number of new tokens to generate')
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
    
    # Initialize policy network
    policy = PolicyNetwork(input_dim=EMBEDDING_DIM, num_actions=4, learning_rate=args.learning_rate)
    actions = ['drop', 'swap', 'insert', 'synonymize']
    
    # Start with random initial prompt
    current_prompt = initial_prompts[0] if initial_prompts else random.choice(initial_prompts)
    current_eval = evaluate_prompt(current_prompt, target, infer)
    best_eval = deepcopy(current_eval)
    
    results = [current_eval]
    
    print(f"Starting Policy Gradient optimization for {args.iterations} iterations")
    print(f"Initial L_val: {current_eval['L_val']:.4f}")
    
    for iteration in range(args.iterations):
        # Get state embedding
        state_embedding = prompt_to_embedding(current_prompt)
        
        # Select action using policy
        action_idx, log_prob = policy.select_action(state_embedding)
        action = actions[action_idx]
        
        # Apply action
        new_prompt = apply_action(current_prompt, action)
        
        # Evaluate
        new_eval = evaluate_prompt(new_prompt, target, infer)
        
        # Compute reward
        reward = current_eval['L_val'] - new_eval['L_val']  # Positive if improved
        
        # Save to trajectory
        policy.save_trajectory_step(state_embedding, action_idx, log_prob, reward)
        
        # Update best
        if new_eval['L_val'] < best_eval['L_val']:
            best_eval = deepcopy(new_eval)
            results.append(best_eval)
            print(f"Iter {iteration+1}/{args.iterations}: {action:10} | New best L_val: {best_eval['L_val']:.4f} | Quality: {best_eval['quality_passed']}")
        
        # Move to new prompt (with some probability, else stay)
        if reward > 0 or random.random() < 0.3:
            current_prompt = new_prompt
            current_eval = new_eval
        
        # Update policy every N steps
        update_interval = 10
        if (iteration + 1) % update_interval == 0:
            policy.update_policy()
            if (iteration + 1) % 100 == 0:
                print(f"Updated policy at iteration {iteration+1}")
    
    # Save results
    save_results(args.output, results)
    
    print(f"\n=== Final Results ===")
    print(f"Best L_val achieved: {best_eval['L_val']:.4f}")
    print(f"Best quality_passed: {best_eval['quality_passed']}")
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
