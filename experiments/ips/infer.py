from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from gpt_oss.tokenizer import get_tokenizer


@dataclass
class InferenceConfig:
    checkpoint: str
    backend: str = "vllm"
    tensor_parallel_size: int = 2
    context_length: int = 4096
    temperature: float = 0.0
    top_p: float | None = 1.0
    top_k: int | None = None
    seed: int | None = 0
    max_new_tokens: int = 512


class DeterministicInference:
    """Small helper to run deterministic-ish generation across backends."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.tokenizer = get_tokenizer()
        self.eot_token = getattr(self.tokenizer, "eot_token", None)
        self.generator = self._init_generator()

    def _init_generator(self):
        match self.config.backend:
            case "vllm":
                try:
                    from gpt_oss.vllm.token_generator import TokenGenerator
                except ModuleNotFoundError as e:
                    raise ModuleNotFoundError(
                        "Backend 'vllm' requested but package 'vllm' is not installed. "
                        "Install vllm or choose --backend torch|triton."
                    ) from e

                return TokenGenerator(
                    self.config.checkpoint,
                    tensor_parallel_size=self.config.tensor_parallel_size,
                )
            case "triton":
                from gpt_oss.torch.utils import init_distributed
                from gpt_oss.triton.model import TokenGenerator

                device = init_distributed()
                return TokenGenerator(
                    self.config.checkpoint,
                    context=self.config.context_length,
                    device=device,
                )
            case "torch":
                from gpt_oss.torch.utils import init_distributed
                from gpt_oss.torch.model import TokenGenerator

                device = init_distributed()
                return TokenGenerator(self.config.checkpoint, device=device)
            case _:
                raise ValueError(f"Unsupported backend: {self.config.backend}")

    def generate(
        self,
        prompt: str,
        stop_tokens: Optional[List[int]] = None,
        return_logprobs: bool = False,
    ) -> Tuple[str, int, int, Optional[List[float]]]:
        """Generate text from prompt.
        
        Returns:
            text: generated text
            prompt_len: prompt token count
            gen_len: generated token count
            logprobs: per-token log-probabilities (if return_logprobs=True)
        """
        tokens = self.tokenizer.encode(prompt)
        stop = stop_tokens or ([] if self.eot_token is None else [self.eot_token])
        kwargs: dict = {
            "stop_tokens": stop,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_new_tokens,
            "return_logprobs": return_logprobs,
        }

        if self.config.backend == "vllm":
            kwargs.update(
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                seed=self.config.seed,
            )

        generated: list[int] = []
        logprobs: list[float] = []
        for out in self.generator.generate(tokens, **kwargs):
            if isinstance(out, tuple):
                tok_id, logprob = out[0], out[1] if len(out) > 1 else None
            else:
                tok_id, logprob = out, None
            generated.append(tok_id)
            if return_logprobs and logprob is not None:
                logprobs.append(logprob)
            if stop and tok_id in stop:
                break

        text = self.tokenizer.decode(generated)
        return text, len(tokens), len(generated), (logprobs if return_logprobs else None)

    def compute_target_loss(
        self,
        prompt: str,
        target_code: str,
    ) -> float:
        """Compute ℒ_val = -1/N Σ log P(t_i | t_<i, P; θ_fixed) for target code.
        
        Args:
            prompt: instruction prompt P
            target_code: target code T
            
        Returns:
            Negative log-likelihood (lower is better)
        """
        prompt_tokens = self.tokenizer.encode(prompt)
        target_tokens = self.tokenizer.encode(target_code)
        
        # Concatenate prompt + target for teacher-forced evaluation
        full_tokens = prompt_tokens + target_tokens
        stop = []
        
        kwargs: dict = {
            "stop_tokens": stop,
            "temperature": self.config.temperature,
            "max_tokens": len(target_tokens),
            "return_logprobs": True,
        }

        if self.config.backend == "vllm":
            kwargs.update(
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                seed=self.config.seed,
            )

        # Collect log-probabilities for target tokens
        logprobs: list[float] = []
        for out in self.generator.generate(prompt_tokens, **kwargs):
            if isinstance(out, tuple) and len(out) > 1:
                logprobs.append(out[1])
            if len(logprobs) >= len(target_tokens):
                break
        
        if not logprobs:
            return float('inf')  # Failed to compute
        
        # ℒ_val = -1/N Σ log P(t_i | ...)
        return -sum(logprobs) / len(logprobs)
