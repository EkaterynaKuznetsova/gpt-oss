from __future__ import annotations

import ast
import difflib
from typing import Optional


def normalized_similarity(a: str, b: str) -> float:
    """Quick similarity proxy in [0,1] using SequenceMatcher ratio."""
    return difflib.SequenceMatcher(None, a, b).ratio()


def python_ast_equal(a: str, b: str) -> bool:
    """Return True if Python code parses and AST dumps match (rough structural equality)."""
    try:
        a_ast = ast.parse(a)
        b_ast = ast.parse(b)
    except SyntaxError:
        return False
    dump_a = ast.dump(a_ast, include_attributes=False)
    dump_b = ast.dump(b_ast, include_attributes=False)
    return dump_a == dump_b


def copy_similarity_penalty(p: str, t: str) -> float:
    """Penalty in [0,1] approximating copy overlap between prompt P and target T.
    Here we reuse normalized_similarity as a proxy; callers can scale by beta.
    """
    return normalized_similarity(p, t)


def length_penalty(p: str, alpha: float) -> float:
    return alpha * len(p)
