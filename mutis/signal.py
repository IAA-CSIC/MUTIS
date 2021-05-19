# Licensed under a 3-clause BSD style license - see LICENSE
"""Synthetic generation of light curves."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as scipy_optimize
import scipy.special as scipy_special
import scipy.stats as scipy_stats
from matplotlib.offsetbox import AnchoredText

from mutis.lib.signal import *

__all__ = ["Signal"]

log = logging.getLogger(__name__)



class Signal:
    """Analysis and generation of a signal.

    Description goes here.

    Parameters
    ----------
    times : :class:`numpy.ndarray` or :class:`pandas.Series`
        Values of the time axis.
    values : :class:`numpy.ndarray` or :class:`pandas.Series`
        Values of the signal axis.
    fgen : :class:`str`
        Method used to generate the synthetic signal.
    """

    def __init__(self, times, values, fgen):
        self.times = np.array(times)
        self.values = np.array(values)
        self.fgen = fgen
        self.synth = None

        # TODO make attributes below specific of OU method / not the entire class
        self.theta = None
        self.mu = None
        self.sigma = None

    def gen_synth(self, samples):
        """Description goes here."""

        self.synth = np.empty((samples, self.times.size))
        for n in range(samples):
            if self.fgen == "lc_gen_samp":
                self.synth[n] = lc_gen_samp(self.values)
            elif self.fgen == "lc_gen_psd_nft":
                self.synth[n] = lc_gen_psd_nft(self.times, self.values)
            elif self.fgen == "lc_gen_psd_lombscargle":
                self.synth[n] = lc_gen_psd_lombscargle(self.times, self.values)
            elif self.fgen == "lc_gen_psd_fft":
                self.synth[n] = lc_gen_psd_fft(self.values)
            elif self.fgen == "lc_gen_ou":
                if self.theta is None or self.mu is None or self.sigma is None:
                    raise Exception("You need to set the parameters for the signal")
                self.synth[n] = lc_gen_ou(self.theta, self.mu, self.sigma, self.times)
            else:
                raise(f'Unknown fgen method {self.fgen}')

    def OU_fit(self, bins=None, rang=None, a=1e-5, b=100):
        """Description goes here."""

        # TODO: make a generic fit method
        res = dict()

        # estimate sigma
        try:
            # dy = np.diff(self.values)
            # dt = np.diff(self.times)
            sigma_est = (np.nanmean(np.diff(self.values) ** 2 / self.values[:-1] ** 2 / np.diff(self.times))) ** 0.5
            res["sigma_est"] = sigma_est
        except Exception as e:
            raise Exception(f"Could not estimate sigma: {e}")

        # plot histogram
        if bins is None:
            bins = np.int(self.values.size ** 0.5 / 1.5)  # bins='auto'
        if rang is None:
            rang = (np.percentile(self.values, 0), np.percentile(self.values, 99))
        p, x = np.histogram(self.values, density=True, bins=bins, range=rang)  # bins='sqrt')
        x = (x + np.roll(x, -1))[:-1] / 2.0

        plt.subplots()
        plt.hist(self.values, density=True, alpha=0.75, bins=bins, range=rang)
        plt.plot(x, p, "r-", alpha=0.5)
        anchored_text = AnchoredText(
            f"mean    {np.mean(self.values):.2g} \n "
            "median  {np.median(self.values):.2g} \n "
            "mode    {scipy_stats.mode(self.values)[0][0]:.2g} \n "
            "std     {np.std(self.values):.2g} \n "
            "var     {np.var(self.values):.2g}",
            loc="upper right",
        )
        plt.gca().add_artist(anchored_text)

        # curve_fit
        try:
            popt, pcov = scipy_optimize.curve_fit(f=self.pdf, xdata=x, ydata=p)
            # print('curve_fit: (l, mu)')
            # print('popt: ')
            # print(popt)
            # print('pcov: ')
            # print(np.sqrt(np.diag(pcov)))
            x_c = np.linspace(1e-5, 1.1 * np.max(x), 1000)
            plt.plot(x_c, self.pdf(x_c, *popt), "k-", label="curve_fit", alpha=0.8)
            res["curve_fit"] = (popt, np.sqrt(np.diag(pcov)))
        except Exception as e:
            raise Exception(f"Some error fitting with curve_fit {e}")

        # TODO: place outside as a helper class
        #       and understand how it is used
        #       is it mu really needed ?
        #       mu here is different than self.mu
        # fit pdf with MLE
        class OU(scipy_stats.rv_continuous):
            def _pdf(self, x, l, mu):
                return (l * mu) ** (1 + l) / scipy_special.gamma(1 + l) * np.exp(-l * mu / x) / x ** (l + 2)

        try:
            fit = OU(a=a, b=np.percentile(self.values, b)).fit(self.values, 1, 1, floc=0, fscale=1)
            # print('MLE fit: (l, mu)')
            # print(fit)
            x_c = np.linspace(0, 1.1 * np.max(x), 1000)
            plt.plot(x_c, self.pdf(x_c, fit[0], fit[1]), "k-.", label="MLE", alpha=0.8)
            res["MLE_fit"] = fit[:-2]
        except Exception as e:
            raise Exception(f"Some error fitting with MLE {e}")
        plt.legend(loc="lower right")
        plt.show()

        # estimate theta
        res["th_est1"] = fit[0] * sigma_est ** 2 / 2
        res["th_est2"] = popt[0] * sigma_est ** 2 / 2

        return res

    def OU_check_gen(self, theta, mu, sigma):
        """Description goes here."""

        # TODO: make a generic check_gen method

        t, y = self.times, self.values
        y2 = lc_gen_ou(theta, mu, sigma, self.times, scale=np.std(self.values), loc=np.mean(self.values))

        # plot the two signals
        fig, ax = plt.subplots()
        ax.plot(t, y, "b-", label="orig", lw=0.5, alpha=0.8)
        ax2 = ax.twinx()
        ax2.plot(t, y2, "r-", label="gen", lw=0.5, alpha=0.8)
        plt.show()

        # plot their histogram
        fig, ax = plt.subplots()
        bins = "auto"  # bins = np.int(y.size**0.5/1.5) #
        rang = (np.percentile(y, 0), np.percentile(y, 99))
        ax.hist(y, density=True, color="b", alpha=0.4, bins=bins, range=rang)
        ax2 = ax.twinx()
        bins = "auto"  # bins = np.int(y.size**0.5/1.5) #
        rang = (np.percentile(y2, 0), np.percentile(y2, 99))
        ax2.hist(y2, density=True, color="r", alpha=0.4, bins=bins, range=rang)
        plt.show()

        # plot their PSD
        fig, ax = plt.subplots()
        ax.psd(y, color="b", lw=1, alpha=0.5)
        ax2 = ax.twinx()
        ax2.psd(y2, color="r", lw=1, alpha=0.5)
        plt.show()

    def PSD_check_gen(self, fgen=None, ax=None):
        """Description goes here."""

        # TODO: make a generic check_gen method

        if fgen is None:
            fgen = self.fgen
        if ax is None:
            ax = plt.gca()

        if fgen == "lc_gen_psd_nft":
            s2 = lc_gen_psd_nft(self.times, self.values)
        elif fgen == "lc_gen_psd_lombscargle":
            s2 = lc_gen_psd_lombscargle(self.times, self.values)
        elif fgen == "lc_gen_psd_fft":
            s2 = lc_gen_psd_fft(self.values)
        else:
            raise Exception("No valid fgen specified")

        ax.plot(self.times, self.values, "b-", label="orig", lw=0.5, alpha=0.8)
        ax.plot(self.times, s2, "r-", label="gen", lw=0.5, alpha=0.8)
        ax.legend()

    @staticmethod
    def pdf(xx, ll, mu):
        """Fit pdf as a curve."""
        return (ll * mu) ** (1 + ll) / scipy_special.gamma(1 + ll) * np.exp(-ll * mu / xx) / xx ** (ll + 2)
