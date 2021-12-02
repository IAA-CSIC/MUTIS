# Licensed under a 3-clause BSD style license - see LICENSE
"""Utility functions used in mutis."""

import numpy as np
import scipy as sp

__all__ = ["get_grid", "memoize"]


def memoize(f):
    """Decorator for recursive memoization."""
    memo = {}

    def helper(a, b):
        x = np.array([a, b], dtype="object")
        y = bytes(x)
        if y not in memo:
            memo[y] = f(a, b)
        return memo[y]

    return helper


@memoize
def get_grid(x, y):
    """Compute a meshgrid with memoization."""
    return np.meshgrid(x, y)


def interp_smooth_curve(x, y, s, N=None):
    """Return an interpolated and smoothed curve of len N. A gaussian kernel of std = s (in units of x) is used for smoothing.
    If N is None, an array of the same length is returned (but interpolated so it is equispaced).
    """
    
    s = s/np.ptp(x)*len(x)
    
    if N is None:
        N = len(x)
        
    spl = sp.interpolate.splrep(x, y)
    xs = np.linspace(min(x), max(x), N)
    ys = sp.interpolate.splev(xs, spl)
        
    ys = sp.ndimage.gaussian_filter1d(ys, s)
    
    return xs, ys