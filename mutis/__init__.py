# Licensed under a 3-clause BSD style license - see LICENSE
"""MUTIS: A Python package for muti-wavelength time series analysis."""

__all__ = ["__version__"]

import pkg_resources

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    # package is not installed
    pass

from mutis.signal import Signal
from mutis.correlation import Correlation