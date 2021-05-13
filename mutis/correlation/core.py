# Licensed under a 3-clause BSD style license - see LICENSE
"""Analysis of correlation of light curves."""

import logging
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from mutis.correlation.methods import *


__all__ = ["Correlation"]

log = logging.getLogger(__name__)


class Correlation:
    """Analysis of the correlation of two signals.

    Parameters
    ----------
    signal1 : `~numpy.ndarray`
        Values of the time axis.
    signal2 : `~numpy.ndarray`
        Values of the signal axis.
    method : str
        Method used to correlate the signals.
    """

    def __init__(self, signal1, signal2, method):
        self.signal1 = signal1
        self.signal1 = signal2
        self.method = method
    pass

