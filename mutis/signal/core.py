# Licensed under a 3-clause BSD style license - see LICENSE
"""Synthetic generation of light curves."""

import logging
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from mutis.signal.methods import *


__all__ = ["Signal"]

log = logging.getLogger(__name__)


class Signal:
    """Analysis and generation of a signal.

    Parameters
    ----------
    times : `~numpy.ndarray`
        Values of the time axis.
    signs : `~numpy.ndarray`
        Values of the signal axis.
    method : str
        Method used to generate the synthetic signal.
    """

    def __init__(self, times, signs, method):
        self.times = times
        self.signs = signs
        self.method = method
        self.synth = None
        self.theta = None
        self.mu = None
        self.sigma = None

    def synth_gen(self, N):
        self.synth = np.empty((N, self.times.size))

        for n in range(0, N):
            if self.method == 'lc_gen_samp':
                self.synth[n] = lc_gen_samp(self.signs)

            if self.method == 'lc_gen_psd':
                self.synth[n] = lc_gen_psd(self.signs)

            if self.method == 'lc_gen_ou':
                if self.theta is None or self.mu is None or self.sigma is None:
                    raise Exception('You need to set the parameters for the signal')
                else:
                    self.synth[n] = lc_gen_ou(self.theta, self.mu, self.sigma, self.times)
