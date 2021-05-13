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
        self.signal2 = signal2
        self.method = method
        self.times = np.array([])
        self.dts = np.array([])
        self.nb = np.array([])
        self.N = None
        self.l1s = None
        self.l2s = None
        self.l3s = None
        self.signs = None

    def synth_gen(self, N):
        self.N = N
        self.signal1.synth_gen(N)
        self.signal2.synth_gen(N)

    def gen_corr(self):
        mc_corr = np.empty((self.N, self.times.size))

        for n in range(0, self.N):
            if self.method == 'welsh_ab':
                mc_corr[n] = welsh_ab(self.signal1.times, self.signal1.synth[n], self.signal2.times, self.signal2.synth[n], self.times, self.dts)
            elif self.method == 'kroedel_ab':
                mc_corr[n] = kroedel_ab(self.signal1.times, self.signal1.synth[n], self.signal2.times, self.signal2.synth[n], self.times,
                                        self.dts)
            elif self.method == 'numpy':
                mc_corr[n] = nindcf(self.signal1.times, self.signal1.synth[n], self.signal2.times, self.signal2.synth[n])

        self.l3s = np.percentile(mc_corr, [0.135, 99.865], axis=0)
        self.l2s = np.percentile(mc_corr, [2.28, 97.73], axis=0)
        self.l1s = np.percentile(mc_corr, [15.865, 84.135], axis=0)

        if self.method == 'welsh_ab':
            self.signs = welsh_ab(self.signal1.times, self.signal1.signs, self.signal2.times, self.signal2.signs, self.times, self.dts)
        elif self.method == 'kroedel_ab':
            self.signs = kroedel_ab(self.signal1.times, self.signal1.signs, self.signal2.times, self.signal2.signs, self.times, self.dts)
        elif self.method == 'numpy':
            self.signs = nindcf(self.signal1.times, self.signal1.signs, self.signal2.times, self.signal2.signs, self.times, self.dts)

    def plot_corr(self):
        plt.figure()

        plt.plot(self.times, self.l1s[0], 'c-.')
        plt.plot(self.times, self.l1s[1], 'c-.', label='$1\sigma$')
        plt.plot(self.times, self.l2s[0], 'k--')
        plt.plot(self.times, self.l2s[1], 'k--', label='$2\sigma$')
        plt.plot(self.times, self.l3s[0], 'r-')
        plt.plot(self.times, self.l3s[1], 'r-', label='$3\sigma$')
        plt.plot(self.times, self.signs, 'b.--', lw=1)

        t1, t2 = self.signal1.times, self.signal2.times

        # SAME LIMIT
        plt.axvline(x=-(np.max([t1.max() - t1.min(), t2.max() - t2.min()])) / 2, ymin=-1, ymax=+1, color='black',
                    linewidth=4, alpha=0.6)
        plt.axvline(x=+(np.max([t1.max() - t1.min(), t2.max() - t2.min()])) / 2, ymin=-1, ymax=+1, color='black',
                    linewidth=4, alpha=0.6)

        # VALID LIMIT
        plt.axvline(x=-(np.max([t1.max() - t1.min(), t2.max() - t2.min()]) - np.min(
            [t1.max() - t1.min(), t2.max() - t2.min()])) / 2, ymin=-1, ymax=+1, color='blue', linewidth=1, alpha=0.6)
        plt.axvline(x=+(np.max([t1.max() - t1.min(), t2.max() - t2.min()]) - np.min(
            [t1.max() - t1.min(), t2.max() - t2.min()])) / 2, ymin=-1, ymax=+1, color='blue', linewidth=1, alpha=0.6)

        plt.legend()
        plt.show()

    def gen_times(self, dt0=None, ndtmax=0.9, nbinsmin=121):
        """
        Returns times and bins to use with adaptative binning methods.
        Sensible values for these parameters must be found by hand, and depend
        on the characteristics of input data.

        dt0:
            minimum bin size, also used as step in a.b.
                default: dt0 = 0.25*(tmax -tmin)/np.sqrt(t1.size*t2.size+1)
            (more or less a statistically reasonable binning,
            to increase precision)
        ndtmax:
            Maximum size of bins (in units of dt0).
                0 < ndtmax < 1: fixed dt (=dt0) (no a.b)
                1 < ndtmax: allow adaptative time binning
            default: 0.9
        nbinsmin:
            if the data has a lot of error, higher values are needed
            to soften the correlation beyond spurious variability.
                default: 121 (11x11)
        """

        t1 = self.signal1.times
        t2 = self.signal2.times

        # tmin = -(np.min([t1.max(),t2.max()]) - np.max([t1.min(),t2.min()]))
        tmax = +(np.max([t1.max(), t2.max()]) - np.min([t1.min(), t2.min()]))
        tmin = -tmax

        if dt0 is None:
            # dt0 = 1*(tmax-tmin)/(t1.size+t2.size-1)
            # dt0 = 1.0*(tmax-tmin)/np.sqrt(t1.size*t2.size+1)
            # dt0 = 30/365; ndtmax=2.9; nbinsmin=12*12
            # dt0 = 0.25; ndtmax=10; nbinsmin=5
            dt0 = 0.25 * (tmax - tmin) / np.sqrt(t1.size * t2.size + 1)

        t1m, t2m = np.meshgrid(t1, t2)
        ti = tmin
        tf = ti + dt0

        while tf < tmax:
            tm = (ti + tf) / 2
            dtm = (tf - ti) / 2
            nbins = np.sum((((tm - dtm / 2) < (t2m - t1m)) & ((t2m - t1m) < (tm + dtm / 2))))
            if dtm < dt0 * ndtmax:
                if nbins > nbinsmin:
                    self.times = np.append(self.times, tm)
                    self.dts = np.append(self.dts, dtm)
                    self.nb = np.append(self.nb, nbins)
                    ti, tf = tf, tf + dt0
                else:
                    tf = tf + dt0
            else:
                ti, tf = tf, tf + dt0
