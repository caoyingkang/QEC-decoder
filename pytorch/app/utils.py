"""Utility functions used in benchmark_app.py."""
from typing import Iterable


def is_unique(items: Iterable) -> bool:
    """Check if an iterable contains unique elements."""
    seen = set()
    for x in items:
        if x in seen:
            return False
        seen.add(x)
    return True
