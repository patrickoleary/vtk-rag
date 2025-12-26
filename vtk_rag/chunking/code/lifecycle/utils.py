"""Shared utility functions for lifecycle processing.

Used by:
    - builder.py (in this directory)
    - ../semantic_chunk.py (parent directory)
"""

from __future__ import annotations

from typing import TypeVar

from .models import MethodCall

T = TypeVar('T')


def dedupe_preserve_order(items: list[T]) -> list[T]:
    """Deduplicate a list while preserving order."""
    seen: set[T] = set()
    return [x for x in items if not (x in seen or seen.add(x))]


def dedupe_method_calls(calls: list[MethodCall]) -> list[MethodCall]:
    """Deduplicate method calls by name, keeping first occurrence."""
    seen: set[str] = set()
    result: list[MethodCall] = []
    for mc in calls:
        if mc["name"] not in seen:
            seen.add(mc["name"])
            result.append(mc)
    return result
