from __future__ import annotations

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
    ) -> Tuple[str, int, int]:
        tokens = self.tokenizer.encode(prompt)
        stop = stop_tokens or ([] if self.eot_token is None else [self.eot_token])
        kwargs: dict = {
            "stop_tokens": stop,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_new_tokens,
        }

        if self.config.backend == "vllm":
            kwargs.update(
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                seed=self.config.seed,
            )

        generated: list[int] = []
        for out in self.generator.generate(tokens, **kwargs):
            tok_id = out[0] if isinstance(out, tuple) else out
            generated.append(tok_id)
            if stop and tok_id in stop:
                break

        text = self.tokenizer.decode(generated)
        return text, len(tokens), len(generated)
