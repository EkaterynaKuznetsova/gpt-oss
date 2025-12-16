from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional


@dataclass
class Target:
    id: str
    language: str
    code: str
    metadata: dict


def iter_jsonl(path: Path | str) -> Generator[Target, None, None]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for i, ln in enumerate(f, 1):
            if not ln.strip():
                continue
            obj = json.loads(ln)
            yield Target(
                id=obj["id"],
                language=obj.get("language", ""),
                code=obj["code"],
                metadata=obj.get("metadata", {}),
            )


def sample_n(path: Path | str, n: int, seed: int = 0) -> list[Target]:
    items = list(iter_jsonl(path))
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n]


def write_jsonl(path: Path | str, items: Iterable[dict]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
