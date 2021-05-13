# Licensed under a 3-clause BSD style license - see LICENSE
"""Utility functions used in mutis."""

import numpy as np

__all__ = ["get_grid", "memoize", "nindcf"]


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


def nindcf(t1, s1, t2, s2):
    """Implement normalization and interpolation over numpy correlate function."""
    # numpy correlate function is not designed for unevenly spaced data
    # correlation C(tau), where tau goes from
    # -( np.max([t1.max(),t2.max()]) - np.min([t1.min(),t2.min()]) )
    # to
    # +( np.min([t1.max(),t2.max()]) - np.max([t1.min(),t2.min()]) )

    s1i = np.interp(np.linspace(t1.min(), t1.max(), t1.size), t1, s1)
    s2i = np.interp(np.linspace(t2.min(), t2.max(), t2.size), t2, s2)
    x = (s1i - np.mean(s1i)) / np.std(s1i) / len(s1i)
    y = (s2i - np.mean(s2i)) / np.std(s2i)
    return np.correlate(x, y, 'full')


