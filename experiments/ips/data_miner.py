import ast
import json
import os
import re
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Iterable, Tuple

# Ensure local package imports work when running the script directly from the repo
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_oss.tokenizer import get_tokenizer

EXCLUDE_DIR_HINTS = (
    "test",
    "tests",
    "example",
    "examples",
    "venv",
    "env",
    "node_modules",
    "build",
    "dist",
    "vendor",
)

MIN_TOK = 200
MAX_TOK = 800

POPULAR_REPOS = [
    # Safe, popular Python repos with permissive licenses
    "https://github.com/pallets/flask.git",        # BSD-3-Clause
    "https://github.com/psf/requests.git",        # Apache-2.0
    "https://github.com/pydantic/pydantic.git",   # MIT
    "https://github.com/tiangolo/fastapi.git",    # MIT
]


def _looks_excluded(path_str: str) -> bool:
    low = path_str.lower()
    return any(h in low for h in EXCLUDE_DIR_HINTS)


def _safe_read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def _license_guess(repo_url: str) -> str:
    # Best-effort static mapping for the POPULAR_REPOS above
    if "flask" in repo_url:
        return "BSD-3-Clause"
    if "requests" in repo_url:
        return "Apache-2.0"
    if "pydantic" in repo_url:
        return "MIT"
    if "fastapi" in repo_url:
        return "MIT"
    return "UNKNOWN"


def count_tokens(text: str) -> int:
    tok = get_tokenizer()
    return len(tok.encode(text))


def iter_python_targets(root: Path, min_tok: int = MIN_TOK, max_tok: int = MAX_TOK) -> Iterable[Tuple[Path, str, int]]:
    for path in root.rglob("*.py"):
        pstr = str(path)
        if _looks_excluded(pstr):
            continue
        src = _safe_read(path)
        if not src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            # Skip non-parseable
            continue
        # Whole file candidate
        tok_len = count_tokens(src)
        if min_tok <= tok_len <= max_tok:
            yield path, src, tok_len
        # Functions/classes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                segment = ast.get_source_segment(src, node)
                if not segment:
                    continue
                tok = count_tokens(segment)
                if min_tok <= tok <= max_tok:
                    yield path, segment, tok


def collect_repo(repo_url: str, out_jsonl: Path) -> int:
    """Clone repo shallowly, extract Python targets, append to JSONL. Returns count."""
    license_id = _license_guess(repo_url)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--depth", "1", repo_url, tmp], check=True)
        root = Path(tmp)
        entries = []
        for path, code, tok in iter_python_targets(root):
            rel = path.relative_to(root)
            repo = repo_url.split("/")[-1].removesuffix(".git")
            entries.append({
                "id": f"{repo}:{rel}",
                "language": "python",
                "code": code,
                "metadata": {"license": license_id, "len_tokens": tok},
            })
        with out_jsonl.open("a", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        return len(entries)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect IPS targets from popular repos")
    parser.add_argument("--output", type=Path, default=Path("data/ips/targets.jsonl"))
    parser.add_argument("--min-tok", type=int, default=MIN_TOK)
    parser.add_argument("--max-tok", type=int, default=MAX_TOK)
    parser.add_argument("--extra-repo", action="append", default=[], help="Additional repo URLs to include")
    args = parser.parse_args()

    # Ensure output dir exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    for url in POPULAR_REPOS + args.extra_repo:
        try:
            total += collect_repo(url, args.output)
        except subprocess.CalledProcessError as e:
            print(f"Skip {url}: {e}")
    print(f"Wrote {total} targets to {args.output}")


if __name__ == "__main__":
    main()
