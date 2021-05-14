# Licensed under a 3-clause BSD style license - see LICENSE
"""Synthetic generation of light curves."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as scipy_optimize
import scipy.special as scipy_special
import scipy.stats as scipy_stats
from matplotlib.offsetbox import AnchoredText

from mutis.signal.methods import *

__all__ = ["Signal"]

log = logging.getLogger(__name__)


class Signal:
    """Analysis and generation of a signal.

    Description goes here.

    Parameters
    ----------
    times : numpy.ndarray
        Values of the time axis.
    signs : numpy.ndarray
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

        for n in range(N):
            if self.method == "lc_gen_samp":
                self.synth[n] = lc_gen_samp(self.signs)

            if self.method == "lc_gen_psd":
                self.synth[n] = lc_gen_psd(self.signs)

            if self.method == "lc_gen_ou":
                if self.theta is None or self.mu is None or self.sigma is None:
                    raise Exception("You need to set the parameters for the signal")
                else:
                    self.synth[n] = lc_gen_ou(self.theta, self.mu, self.sigma, self.times)

    def OU_fit(self):
        y = self.signs
        t = self.times

        bins = np.int(y.size ** 0.5 / 1.5)  # bins='auto'
        rang = (np.percentile(y, 0), np.percentile(y, 99))

        p, x = np.histogram(y, density=True, bins=bins, range=rang)  # bins='sqrt')
        x = (x + np.roll(x, -1))[:-1] / 2.0

        # plot histogram
        plt.subplots()

        plt.hist(y, density=True, alpha=0.75, bins=bins, range=rang)
        plt.plot(x, p, "r-", alpha=0.5)

        anchored_text = AnchoredText(
            f"mean    {np.mean(y):.2g} \n "
            "median  {np.median(y):.2g} \n "
            "mode    {scipy_stats.mode(y)[0][0]:.2g} \n "
            "std     {np.std(y):.2g} \n "
            "var     {np.var(y):.2g}",
            loc="upper right",
        )
        plt.gca().add_artist(anchored_text)

        # fit pdf as a curve
        pdf = lambda x, l, mu: (l * mu) ** (1 + l) / scipy_special.gamma(1 + l) * np.exp(-l * mu / x) / x ** (l + 2)

        try:
            popt, pcov = scipy_optimize.curve_fit(f=pdf, xdata=x, ydata=p)

            # print('curve_fit: (l, mu)')
            # print('popt: ')
            # print(popt)
            # print('pcov: ')
            # print(np.sqrt(np.diag(pcov)))

            x_c = np.linspace(1e-5, 1.1 * np.max(x), 1000)
            plt.plot(x_c, pdf(x_c, *popt), "k-", label="curve_fit", alpha=0.8)
        except Exception as e:
            print("Some error fitting with curve_fit:")
            print(e)

        # fit pdf with MLE
        class OU(scipy_stats.rv_continuous):
            def _pdf(self, x, l, mu):
                return (l * mu) ** (1 + l) / scipy_special.gamma(1 + l) * np.exp(-l * mu / x) / x ** (l + 2)

        try:
            fit = OU(a=1e-5, b=100 * np.percentile(y, 100)).fit(y, 1, 1, floc=0, fscale=1)

            # print('MLE fit: (l, mu)')
            # print(fit)

            x_c = np.linspace(0, 1.1 * np.max(x), 1000)
            plt.plot(x_c, pdf(x_c, fit[0], fit[1]), "k-.", label="MLE", alpha=0.8)
        except Exception as e:
            print("Some error fitting with MLW:")
            print(e)

        plt.legend(loc="lower right")

        plt.show()

        dy = np.diff(y)
        dt = np.diff(t)
        sigma_est = (np.mean(dy ** 2 / y[:-1] ** 2 / dt)) ** 0.5

        th_est1 = fit[0] * sigma_est ** 2 / 2
        th_est2 = popt[0] * sigma_est ** 2 / 2

        return {
            "curve_fit": (popt, np.sqrt(np.diag(pcov))),
            "MLE_fit": fit[:-2],
            "sigma_est": sigma_est,
            "th_est1": th_est1,
            "th_est2": th_est2,
        }

    def OU_check_gen(self, theta, mu, sigma):
        t, y = self.t, self.s
        y2 = lc_gen_ou(theta, mu, sigma, self.t, scale=np.std(self.s), loc=np.mean(self.s))

        # Plot the two signals
        fig, ax = plt.subplots()

        ax.plot(t, y, "b-", label="orig", lw=0.5, alpha=0.8)

        ax2 = ax.twinx()
        ax2.plot(t, y2, "r-", label="gen", lw=0.5, alpha=0.8)

        plt.show()

        # Plot their histogram
        fig, ax = plt.subplots()

        bins = "auto"  # bins = np.int(y.size**0.5/1.5) #
        rang = (np.percentile(y, 0), np.percentile(y, 99))
        ax.hist(y, density=True, color="b", alpha=0.4, bins=bins, range=rang)

        ax2 = ax.twinx()
        bins = "auto"  # bins = np.int(y.size**0.5/1.5) #
        rang = (np.percentile(y2, 0), np.percentile(y2, 99))
        ax2.hist(y2, density=True, color="r", alpha=0.4, bins=bins, range=rang)

        plt.show()

        # Plot their PSD
        fig, ax = plt.subplots()

        ax.psd(y, color="b", lw=1, alpha=0.5)

        ax2 = ax.twinx()
        ax2.psd(y2, color="r", lw=1, alpha=0.5)

        plt.show()
