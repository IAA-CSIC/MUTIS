# Licensed under a 3-clause BSD style license - see LICENSE
import numpy as np

__all__ = ["get_grid", "memoize"]


def memoize(f):
    """Decorator for recursive memoization."""
    memo = {}

    def helper(a, b):
        x = np.array([a, b], dtype='object')
        y = bytes(x)
        if y not in memo:
            memo[y] = f(a, b)
        return memo[y]
    return helper


@memoize
def get_grid(x, y):
    """Compute a meshgrid with memoization."""
    return np.meshgrid(x, y)
