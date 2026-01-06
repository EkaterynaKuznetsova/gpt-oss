import requests
import os
from typing import Iterable, List, Optional
from gpt_oss.tokenizer import get_tokenizer

class TokenGenerator:
    def __init__(self, model_path: Optional[str] = None, tensor_parallel_size: int = 1, api_base: Optional[str] = None):
        self.api_base = (api_base or os.environ.get("VLLM_API_BASE") or "http://localhost:8001/v1").rstrip("/")
        self.model = os.environ.get("VLLM_MODEL", "openai/gpt-oss-20b")
        self.timeout = float(os.environ.get("VLLM_TIMEOUT", "300"))
        self.tokenizer = get_tokenizer()

    def _decode_stops(self, stop_tokens: Optional[List[int]]) -> Optional[List[str]]:
        if not stop_tokens:
            return None
        return [self.tokenizer.decode([tid]) for tid in stop_tokens]

    def _extract_text(self, data: dict) -> Optional[str]:
        # Chat Completions schema
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            text = msg.get("content")
            if isinstance(text, str) and text.strip():
                return text
            # Sometimes 'text' on choice or 'reasoning_content'
            text = choices[0].get("text") or msg.get("reasoning_content") or choices[0].get("reasoning_content")
            if isinstance(text, str) and text.strip():
                return text
        # Responses API aggregate
        text = data.get("output_text")
        if isinstance(text, str) and text.strip():
            return text
        return None

    def generate(
        self,
        prompt_tokens: List[int],
        stop_tokens: Optional[List[int]] = None,
        temperature: float = 1.0,
        max_tokens: int = 0,
        return_logprobs: bool = False,
        top_p: Optional[float] = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Iterable[int]:
        prompt = self.tokenizer.decode(prompt_tokens)
        url = f"{self.api_base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens or 512,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "stop": self._decode_stops(stop_tokens),
            "tools": [],
            "tool_choice": "none",
            "stream": False,
            "logprobs": return_logprobs,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        text = self._extract_text(data)

        # If logprobs requested and available, yield (token_id, logprob) pairs
        if return_logprobs:
            choices = data.get("choices") or []
            lp = None
            if choices:
                lp = choices[0].get("logprobs")
            if isinstance(lp, dict):
                content = lp.get("content") or []
                logprob_vals = []
                for item in content:
                    val = item.get("logprob")
                    if isinstance(val, (int, float)):
                        logprob_vals.append(float(val))
                if isinstance(text, str) and text:
                    token_ids = self.tokenizer.encode(text)
                    for tid, lpv in zip(token_ids, logprob_vals):
                        yield (tid, lpv)
                    for tid in token_ids[len(logprob_vals):]:
                        yield (tid, None)
                    return

        # Fallback: yield token ids only
        if not isinstance(text, str) or not text:
            snippet = str(data)[:500]
            raise ValueError(
                f"Empty text in vLLM response (tool_choice=none). Response snippet: {snippet}"
            )
        for tid in self.tokenizer.encode(text):
            yield tid