# Licensed under a 3-clause BSD style license - see LICENSE
"""Analysis of correlation of light curves."""

import logging

import matplotlib.pyplot as plt
import numpy as np

from mutis.lib.correlation import *

__all__ = ["Correlation"]

log = logging.getLogger(__name__)


class Correlation:
    """Analysis of the correlation of two signals.

    Parameters
    ----------
    signal1 : :class:`~mutis.signal.Signal`
        Values of the time axis.
    signal2 : :class:`~mutis.signal.Signal`
        Values of the signal axis.
    method : :py:class:`~str`
        Method used to correlate the signals.
    """

    def __init__(self, signal1, signal2, method):
        self.signal1 = signal1
        self.signal2 = signal2
        self.method = method
        self.times = np.array([])
        self.dts = np.array([])
        self.nb = np.array([])

        # TODO: have a much smaller set of attributes
        self.N = None
        self.l1s = None
        self.l2s = None
        self.l3s = None
        self.signs = None

        t1, t2 = self.signal1.times, self.signal2.times
        self.tmin_full = t2.min() - t1.max()
        self.tmax_full = t2.max() - t1.min()
        self.t0_full = (self.tmax_full + self.tmin_full) / 2
        self.tmin_same = -(np.max([t1.max() - t1.min(), t2.max() - t2.min()])) / 2 + self.t0_full
        self.tmax_same = (np.max([t1.max() - t1.min(), t2.max() - t2.min()])) / 2 + self.t0_full
        self.tmin_valid = -(np.max([t1.max() - t1.min(), t2.max() - t2.min()]) - np.min([t1.max() - t1.min(), t2.max() - t2.min()]))/ 2 + self.t0_full
        self.tmax_valid = +(np.max([t1.max() - t1.min(), t2.max() - t2.min()]) - np.min([t1.max() - t1.min(), t2.max() - t2.min()]))/ 2 + self.t0_full

    def gen_synth(self, N):
        """Description goes here.

        Parameters
        ----------
        N : :py:class:`~int`
            Parameter description.
        """

        self.N = N
        self.signal1.gen_synth(N)
        self.signal2.gen_synth(N)

    def gen_corr(self):
        """Description goes here."""

        if not len(self.times) or not len(self.dts):
            raise Exception(
                "You need to define the times on which to calculate the correlation."
                "Please use gen_times() or manually set them."
            )

        # TODO: refactor if/elif with an helper function
        mc_corr = np.empty((self.N, self.times.size))
        if self.method == "welsh_ab":
            for n in range(self.N):
                mc_corr[n] = welsh_ab(
                    self.signal1.times,
                    self.signal1.synth[n],
                    self.signal2.times,
                    self.signal2.synth[n],
                    self.times,
                    self.dts,
                )
            self.signs = welsh_ab(
                self.signal1.times,
                self.signal1.signs,
                self.signal2.times,
                self.signal2.signs,
                self.times,
                self.dts,
            )
        elif self.method == "kroedel_ab":
            for n in range(self.N):
                mc_corr[n] = kroedel_ab(
                    self.signal1.times,
                    self.signal1.synth[n],
                    self.signal2.times,
                    self.signal2.synth[n],
                    self.times,
                    self.dts,
                )
                self.signs = kroedel_ab(
                    self.signal1.times,
                    self.signal1.signs,
                    self.signal2.times,
                    self.signal2.signs,
                    self.times,
                    self.dts,
                )
        elif self.method == "numpy":
            for n in range(self.N):
                mc_corr[n] = nindcf(
                    self.signal1.times,
                    self.signal1.synth[n],
                    self.signal2.times,
                    self.signal2.synth[n],
                )
            self.signs = nindcf(
                self.signal1.times,
                self.signal1.signs,
                self.signal2.times,
                self.signal2.signs,
            )
        else:
            raise Exception("Unknown method " + self.method + " for correlation.")

        self.l3s = np.percentile(mc_corr, [0.135, 99.865], axis=0)
        self.l2s = np.percentile(mc_corr, [2.28, 97.73], axis=0)
        self.l1s = np.percentile(mc_corr, [15.865, 84.135], axis=0)

    def gen_times(self, ftimes="canopy", *args, **kwargs):
        """Sets times and bins using the method defined by ftimes parameter.

        Parameters
        ----------
        ftimes : :py:class:`~str`
            Method used to bin the time interval of the correlation. Possible values are:
                - "canopy": Computes a binning as dense as possible, with variable bin width and (with a minimum and a maximum resolution) and a minimum statistic.
                - "rawab": Computes a binning with variable bin width, a given step, maximum bin size and a minimum statistic.
                - "uniform": Computes a binning with uniform bin width and a minimum statistic.
                - "numpy": Computes a binning suitable for method='numpy'.
        """
        if ftimes == "canopy":
            self.times, self.dts, self.nb = gen_times_canopy(self.signal1.times, self.signal2.times, *args, **kwargs)
        elif ftimes == "rawab":
            self.times, self.dts, self.nb = gen_times_rawab(self.signal1.times, self.signal2.times, *args, **kwargs)
        elif ftimes == "uniform":
            self.times, self.dts, self.nb = gen_times_uniform(self.signal1.times, self.signal2.times, *args, **kwargs)
        elif ftimes == "numpy":
            t1, t2 = self.signal1.times, self.signal1.times
            dt = np.max([(t1.max() - t1.min()) / t1.size, (t2.max() - t2.min()) / t2.size])
            n1 = np.int(np.ptp(t1) / dt * 10.0)
            n2 = np.int(np.ptp(t1) / dt * 10.0)
            self.times = np.linspace(self.tmin_full, self.tmax_full, n1 + n2 - 1)
            self.dts = np.full(self.times.size, (self.tmax_full - self.tmin_full) / (n1 + n2))
        else:
            raise Exception("Unknown method " + ftimes + ", please indicate how to generate times.")

    def plot_corr(self, ax=None, legend=None):
        """Description goes here."""

        # TODO: develop a plotting object for plots
        #       this will considerably shorten the
        #       number of attributes of this class

        # plt.figure()
        if ax is None:
            ax = plt.gca()

        ax.plot(self.times, self.l1s[0], "c-.")
        ax.plot(self.times, self.l1s[1], "c-.", label="$1\sigma$")
        ax.plot(self.times, self.l2s[0], "k--")
        ax.plot(self.times, self.l2s[1], "k--", label="$2\sigma$")
        ax.plot(self.times, self.l3s[0], "r-")
        ax.plot(self.times, self.l3s[1], "r-", label="$3\sigma$")
        ax.plot(self.times, self.signs, "b.--", lw=1)

        # full limit
        ax.axvline(x=self.tmin_full, ymin=-1, ymax=+1, color="red", linewidth=4, alpha=0.5)
        ax.axvline(x=self.tmax_full, ymin=-1, ymax=+1, color="red", linewidth=4, alpha=0.5)
        # same limit
        ax.axvline(x=self.tmin_same, ymin=-1, ymax=+1, color="black", linewidth=2, alpha=0.5)
        ax.axvline(x=self.tmax_same, ymin=-1, ymax=+1, color="black", linewidth=2, alpha=0.5)
        # valid limit
        ax.axvline(x=self.tmin_valid, ymin=-1, ymax=+1, color="cyan", linewidth=1, alpha=0.5)
        ax.axvline(x=self.tmax_valid, ymin=-1, ymax=+1, color="cyan", linewidth=1, alpha=0.5)

        if legend is not None:
            ax.legend()

        # plt.show()
        return ax

    def plot_times(self, rug=False):
        """Description goes here."""

        # TODO: develop a plotting object for plots
        #       this will considerably shorten the
        #       number of attributes of this class

        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        tab, dtab, nab = self.times, self.dts, self.nb

        fig.suptitle("Total bins: {:d}".format(self.times.size))
        ax[0].plot(tab, nab, "b.")
        ax[0].errorbar(x=tab, y=nab, xerr=dtab / 2, fmt="none")
        ax[0].set_ylabel("$n_i$")
        ax[0].grid()
        ax[0].axvline(x=self.tmin_full, ymin=-1, ymax=+1, color="red", linewidth=4, alpha=0.5)
        ax[0].axvline(x=self.tmax_full, ymin=-1, ymax=+1, color="red", linewidth=4, alpha=0.5)
        ax[0].axvline(x=self.tmin_same, ymin=-1, ymax=+1, color="black", linewidth=2, alpha=0.5)
        ax[0].axvline(x=self.tmax_same, ymin=-1, ymax=+1, color="black", linewidth=2, alpha=0.5)
        ax[0].axvline(x=self.tmin_valid, ymin=-1, ymax=+1, color="cyan", linewidth=1, alpha=0.5)
        ax[0].axvline(x=self.tmax_valid, ymin=-1, ymax=+1, color="cyan", linewidth=1, alpha=0.5)
        ax[1].plot(tab, dtab, "b.")
        ax[1].set_ylabel("$dt_i$")
        # ax[1].grid()
        ax[1].axvline(x=self.tmin_full, ymin=-1, ymax=+1, color="red", linewidth=4, alpha=0.5)
        ax[1].axvline(x=self.tmax_full, ymin=-1, ymax=+1, color="red", linewidth=4, alpha=0.5)
        ax[1].axvline(x=self.tmin_same, ymin=-1, ymax=+1, color="black", linewidth=2, alpha=0.5)
        ax[1].axvline(x=self.tmax_same, ymin=-1, ymax=+1, color="black", linewidth=2, alpha=0.5)
        ax[1].axvline(x=self.tmin_valid, ymin=-1, ymax=+1, color="cyan", linewidth=1, alpha=0.5)
        ax[1].axvline(x=self.tmax_valid, ymin=-1, ymax=+1, color="cyan", linewidth=1, alpha=0.5)

        if rug is True:
            for t in self.times:
                ax[1].axvline(x=t, ymin=0, ymax=0.2, color="black", linewidth=0.8, alpha=1.0)
            # ax[1].plot(self.t, ax[1].get_ylim()[0]+np.zeros(self.t.size), 'k|', alpha=0.8, lw=1)

        ax[1].grid()
        # fig.show()

    def plot_signals(self, ax=None):
        """Description goes here."""

        # TODO: develop a plotting object for plots
        #       this will considerably shorten the
        #       number of attributes of this class

        if ax is None:
            ax = plt.gca()

        ax.plot(self.signal1.times, self.signal1.signs, "b.-", lw=1, alpha=0.4)
        ax.tick_params(axis="y", labelcolor="b")
        ax.set_ylabel("sig 1", color="b")

        ax2 = ax.twinx()
        ax2.plot(self.signal2.times, self.signal2.signs, "r.-", lw=1, alpha=0.4)
        ax2.tick_params(axis="y", labelcolor="r")
        ax2.set_ylabel("sig 2", color="r")
