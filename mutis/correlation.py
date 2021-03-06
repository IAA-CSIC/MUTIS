# Licensed under a 3-clause BSD style license - see LICENSE
"""Analysis of correlation of light curves."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from mutis.lib.correlation import *

from mutis.lib.utils import interp_smooth_curve


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
    fcorr : :py:class:`~str`
        Method used to correlate the signals.
    """

    def __init__(self, signal1, signal2, fcorr):
        self.signal1 = signal1
        self.signal2 = signal2
        self.fcorr = fcorr

        self.times = np.array([])
        self.dts = np.array([])
        self.nb = np.array([])
        self.values = None

        # TODO: have a much smaller set of attributes
        self.samples = None
        # storage of the significance limits of the correlation
        self.l1s = None
        self.l2s = None
        self.l3s = None
        # storage of the uncertainties of the correlation
        self.s1s = None
        self.s2s = None
        self.s3s = None

        # attributes indicating the ranges where the correlations are defined
        t1, t2 = self.signal1.times, self.signal2.times
        self.tmin_full = t2.min() - t1.max()
        self.tmax_full = t2.max() - t1.min()
        self.t0_full = (self.tmax_full + self.tmin_full) / 2
        self.tmin_same = -(np.max([t1.max() - t1.min(), t2.max() - t2.min()])) / 2 + self.t0_full
        self.tmax_same = (np.max([t1.max() - t1.min(), t2.max() - t2.min()])) / 2 + self.t0_full
        self.tmin_valid = (
            -(
                np.max([t1.max() - t1.min(), t2.max() - t2.min()])
                - np.min([t1.max() - t1.min(), t2.max() - t2.min()])
            )
            / 2
            + self.t0_full
        )
        self.tmax_valid = (
            +(
                np.max([t1.max() - t1.min(), t2.max() - t2.min()])
                - np.min([t1.max() - t1.min(), t2.max() - t2.min()])
            )
            / 2
            + self.t0_full
        )

    def peak_find(self, smooth=False, smooth_std=None, Ninterp=1000):
        """Find the peaks of the correlation, optionally smoothing with a kernel of standard deviation `s`.
        Returns dict with peak positions and significances, ordered from closest to farthest from zero.
        """
        x, y = self.times, self.values

        if smooth_std is None:
            dt1 = np.mean(self.signal1.times[1:]-self.signal1.times[:-1])
            std1 = np.std(self.signal1.times[1:]-self.signal1.times[:-1])
            dt2 = np.mean(self.signal2.times[1:]-self.signal2.times[:-1])
            std2 = np.std(self.signal2.times[1:]-self.signal2.times[:-1])

            smooth_std = 1*np.max([dt1,dt2])
                
        if smooth:
            xs, ys = interp_smooth_curve(x, y, smooth_std, Ninterp)
        else:
            xs, ys = x, y

        idx, props = sp.signal.find_peaks(ys)
        
        if smooth:
            s1s_x, s1s_y = interp_smooth_curve(x, self.l1s[1], smooth_std, Ninterp)
        else:
            s1s_x, s1s_y = x, self.l1s[1]

        peak_idx = idx[np.argsort(np.abs(xs[idx]))]
        peak_x = xs[peak_idx]
        peak_y = ys[peak_idx]
        peak_signf1s = ys[peak_idx]/s1s_y[peak_idx]
        
        peak_signif_percent = list()
        for i in range(len(peak_x)):
            f = sp.interpolate.interp1d(self.times, self.mc_corr, axis=-1)
            peak_signif_percent.append( sp.stats.percentileofscore(f(peak_x[i]), peak_y[i], kind='strict') )

        return {'x':peak_x, 's':smooth_std, 'y':peak_y, 'signf1s':peak_signf1s, 'signif_percent':np.array(peak_signif_percent)}

    def gen_synth(self, samples):
        """Generates the synthetic light curves.

        Generates the specified number `samples` of synthetic light
        curves for each signal, to be used to compute the significance
        the correlation.

        Parameters
        ----------
        samples : :py:class:`~int`
            Number of synthetic light curves to be generated for each signal.
        """

        self.samples = samples
        self.signal1.gen_synth(samples)
        self.signal2.gen_synth(samples)

    def gen_corr(self, uncert=True, dsamples=500):
        """Generates the correlation of the signals.

        Generates the correlation of the signals, and computes their
        confidence level from the synthetic light curves, which must
        have been generated before.
        """

        if uncert and self.signal1.dvalues is None:
            log.error(
                "uncert is True but no uncertainties for Signal 1 were specified, setting uncert to False"
            )
            uncert = False
        if uncert and self.signal2.dvalues is None:
            log.error(
                "uncert is True but no uncertainties for Signal 2 were specified, setting uncert to False"
            )
            uncert = False

        if len(self.times) == 0 or len(self.dts) == 0:
            raise Exception(
                "You need to define the times on which to calculate the correlation."
                "Please use gen_times() or manually set them."
            )

        # TODO: refactor if/elif with a helper function
        mc_corr = np.empty((self.samples, self.times.size))
        if uncert:
            mc_sig = np.empty((dsamples, self.times.size))

        if self.fcorr == "welsh":
            for idx in range(self.samples):
                mc_corr[idx] = welsh(
                    self.signal1.times,
                    self.signal1.synth[idx],
                    self.signal2.times,
                    self.signal2.synth[idx],
                    self.times,
                    self.dts,
                )
            if uncert:
                for idx in range(dsamples):
                    mc_sig[idx] = welsh(
                        self.signal1.times,
                        self.signal1.values
                        + self.signal1.dvalues * np.random.randn(self.signal1.values.size),
                        self.signal2.times,
                        self.signal2.values
                        + self.signal2.dvalues * np.random.randn(self.signal2.values.size),
                        self.times,
                        self.dts,
                    )
            self.values = welsh(
                self.signal1.times,
                self.signal1.values,
                self.signal2.times,
                self.signal2.values,
                self.times,
                self.dts,
            )
        elif self.fcorr == "kroedel":
            for idx in range(self.samples):
                mc_corr[idx] = kroedel(
                    self.signal1.times,
                    self.signal1.synth[idx],
                    self.signal2.times,
                    self.signal2.synth[idx],
                    self.times,
                    self.dts,
                )
            if uncert:
                for idx in range(dsamples):
                    mc_sig[idx] = kroedel(
                        self.signal1.times,
                        self.signal1.values
                        + self.signal1.dvalues * np.random.randn(self.signal1.values.size),
                        self.signal2.times,
                        self.signal2.values
                        + self.signal2.dvalues * np.random.randn(self.signal2.values.size),
                        self.times,
                        self.dts,
                    )
            self.values = kroedel(
                self.signal1.times,
                self.signal1.values,
                self.signal2.times,
                self.signal2.values,
                self.times,
                self.dts,
            )
        elif self.fcorr == "welsh_old": # should produce the exactly same results, but we keep it for debugs and testcov
            for idx in range(self.samples):
                mc_corr[idx] = welsh_old(
                    self.signal1.times,
                    self.signal1.synth[idx],
                    self.signal2.times,
                    self.signal2.synth[idx],
                    self.times,
                    self.dts,
                )
            if uncert:
                for idx in range(dsamples):
                    mc_sig[idx] = welsh_old(
                        self.signal1.times,
                        self.signal1.values
                        + self.signal1.dvalues * np.random.randn(self.signal1.values.size),
                        self.signal2.times,
                        self.signal2.values
                        + self.signal2.dvalues * np.random.randn(self.signal2.values.size),
                        self.times,
                        self.dts,
                    )
            self.values = welsh_old(
                self.signal1.times,
                self.signal1.values,
                self.signal2.times,
                self.signal2.values,
                self.times,
                self.dts,
            )
        elif self.fcorr == "kroedel_old": # should produce the exactly same results, but we keep it for debugs and testcov
            for idx in range(self.samples):
                mc_corr[idx] = kroedel_old(
                    self.signal1.times,
                    self.signal1.synth[idx],
                    self.signal2.times,
                    self.signal2.synth[idx],
                    self.times,
                    self.dts,
                )
            if uncert:
                for idx in range(dsamples):
                    mc_sig[idx] = kroedel_old(
                        self.signal1.times,
                        self.signal1.values
                        + self.signal1.dvalues * np.random.randn(self.signal1.values.size),
                        self.signal2.times,
                        self.signal2.values
                        + self.signal2.dvalues * np.random.randn(self.signal2.values.size),
                        self.times,
                        self.dts,
                    )
            self.values = kroedel_old(
                self.signal1.times,
                self.signal1.values,
                self.signal2.times,
                self.signal2.values,
                self.times,
                self.dts,
            )
        elif self.fcorr == "numpy":
            for idx in range(self.samples):
                mc_corr[idx] = nindcf(
                    self.signal1.times,
                    self.signal1.synth[idx],
                    self.signal2.times,
                    self.signal2.synth[idx],
                )
            if uncert:
                for idx in range(dsamples):
                    mc_sig[idx] = nindcf(
                        self.signal1.times,
                        self.signal1.values
                        + self.signal1.dvalues * np.random.randn(self.signal1.values.size),
                        self.signal2.times,
                        self.signal2.values
                        + self.signal2.dvalues * np.random.randn(self.signal2.values.size),
                    )
            self.values = nindcf(
                self.signal1.times,
                self.signal1.values,
                self.signal2.times,
                self.signal2.values,
            )
        else:
            raise Exception("Unknown method " + self.fcorr + " for correlation.")

        self.l3s = np.percentile(mc_corr, [0.135, 99.865], axis=0)
        self.l2s = np.percentile(mc_corr, [2.28, 97.73], axis=0)
        self.l1s = np.percentile(mc_corr, [15.865, 84.135], axis=0)

        self.mc_corr = mc_corr # save them to be able to compute exact significance later...
        
        if uncert:
            self.s3s = np.percentile(mc_sig, [0.135, 99.865], axis=0)
            self.s2s = np.percentile(mc_sig, [2.28, 97.73], axis=0)
            self.s1s = np.percentile(mc_sig, [15.865, 84.135], axis=0)

    def gen_times(self, ftimes="canopy", *args, **kwargs):
        """Sets times and bins using the method defined by ftimes parameter.

        Parameters
        ----------
        ftimes : :py:class:`~str`
            Method used to bin the time interval of the correlation.
            Possible values are:
            - "canopy": Computes a binning as dense as possible, with
            variable bin width and (with a minimum and a maximum
            resolution) and a minimum statistic.
            - "rawab": Computes a binning with variable bin width,
            a given step, maximum bin size and a minimum statistic.
            - "uniform": Computes a binning with uniform bin width
            and a minimum statistic.
            - "numpy": Computes a binning suitable for method='numpy'.
        """
        if ftimes == "canopy":
            self.times, self.dts, self.nb = gen_times_canopy(
                self.signal1.times, self.signal2.times, *args, **kwargs
            )
        elif ftimes == "rawab":
            self.times, self.dts, self.nb = gen_times_rawab(
                self.signal1.times, self.signal2.times, *args, **kwargs
            )
        elif ftimes == "uniform":
            self.times, self.dts, self.nb = gen_times_uniform(
                self.signal1.times, self.signal2.times, *args, **kwargs
            )
        elif ftimes == "numpy":
            t1, t2 = self.signal1.times, self.signal1.times
            dt = np.max([(t1.max() - t1.min()) / t1.size, (t2.max() - t2.min()) / t2.size])
            n1 = int(np.ptp(t1) / dt * 10.0)
            n2 = int(np.ptp(t1) / dt * 10.0)
            self.times = np.linspace(self.tmin_full, self.tmax_full, n1 + n2 - 1)
            self.dts = np.full(self.times.size, (self.tmax_full - self.tmin_full) / (n1 + n2))
        else:
            raise Exception("Unknown method " + ftimes + ", please indicate how to generate times.")

    def plot_corr(self, uncert=True, ax=None, legend=False):
        """Plots the correlation of the signals.

        Plots the correlation of the signal, and the confidence limits
        computed from the synthetic curves.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            Axes to be used (default None, it creates a new axes).
        legend : :py:class:`~bool`
            Whether to add a legend indicating the confidence levels.
        """

        # TODO: develop a plotting object for plots
        #       this will considerably shorten the
        #       number of attributes of this class

        if uncert and self.signal1.dvalues is None:
            log.error(
                "uncert is True but no uncertainties for Signal 1 were specified, setting uncert to False"
            )
            uncert = False
        if uncert and self.signal2.dvalues is None:
            log.error(
                "uncert is True but no uncertainties for Signal 2 were specified, setting uncert to False"
            )
            uncert = False

        # plt.figure()
        if ax is None:
            ax = plt.gca()

        ax.plot(self.times, self.l1s[0], "c-.")
        ax.plot(self.times, self.l1s[1], "c-.", label=r"$1\sigma$")
        ax.plot(self.times, self.l2s[0], "k--")
        ax.plot(self.times, self.l2s[1], "k--", label=r"$2\sigma$")
        ax.plot(self.times, self.l3s[0], "r-")
        ax.plot(self.times, self.l3s[1], "r-", label=r"$3\sigma$")
        ax.plot(self.times, self.values, "b.--", lw=1)

        # full limit
        ax.axvline(x=self.tmin_full, ymin=-1, ymax=+1, color="red", linewidth=4, alpha=0.5)
        ax.axvline(x=self.tmax_full, ymin=-1, ymax=+1, color="red", linewidth=4, alpha=0.5)
        # same limit
        ax.axvline(x=self.tmin_same, ymin=-1, ymax=+1, color="black", linewidth=2, alpha=0.5)
        ax.axvline(x=self.tmax_same, ymin=-1, ymax=+1, color="black", linewidth=2, alpha=0.5)
        # valid limit
        ax.axvline(x=self.tmin_valid, ymin=-1, ymax=+1, color="cyan", linewidth=1, alpha=0.5)
        ax.axvline(x=self.tmax_valid, ymin=-1, ymax=+1, color="cyan", linewidth=1, alpha=0.5)

        if uncert:
            ax.fill_between(x=self.times, y1=self.s1s[0], y2=self.s1s[1], color="b", alpha=0.5)
            ax.fill_between(x=self.times, y1=self.s2s[0], y2=self.s2s[1], color="b", alpha=0.3)
            ax.fill_between(x=self.times, y1=self.s3s[0], y2=self.s3s[1], color="b", alpha=0.1)
            
            
        if legend:
            ax.legend()

        # plt.show()
        return ax

    def plot_times(self, rug=False):
        """Plots the time binning generated previously.

        Plots the number of total bins, their distribution and the
        number of points in each bin for the generated time binning,
        previously generated with Correlation().gen_times(...).

        Parameters
        ----------
        rug : :py:class:`~bool`
            Whether to make a rug plot just below the binning, to make
            it easier to visually understand the density and distribution
            of the generated bins.

        """

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

        if rug:
            for time in self.times:
                ax[1].axvline(x=time, ymin=0, ymax=0.2, color="black", linewidth=0.8, alpha=1.0)
            # ax[1].plot(self.t, ax[1].get_ylim()[0]+np.zeros(self.t.size), 'k|', alpha=0.8, lw=1)

        ax[1].grid()
        # fig.show()

    def plot_signals(self, ax=None):
        """Plots the signals involved in this correlation.

        Plots the signals involved in this correlation, in the same window
        but with different twin y-axes and different colors.

        Parameters
        ----------
        ax : :py:class:`~matplotlib.axes.Axes`
            Axes to be used for plotting.
        """

        # TODO: develop a plotting object for plots
        #       this will considerably shorten the
        #       number of attributes of this class

        if ax is None:
            ax = plt.gca()

        ax.plot(self.signal1.times, self.signal1.values, "b.-", lw=1, alpha=0.4)
        ax.tick_params(axis="y", labelcolor="b")
        ax.set_ylabel("sig 1", color="b")

        ax2 = ax.twinx()
        ax2.plot(self.signal2.times, self.signal2.values, "r.-", lw=1, alpha=0.4)
        ax2.tick_params(axis="y", labelcolor="r")
        ax2.set_ylabel("sig 2", color="r")
