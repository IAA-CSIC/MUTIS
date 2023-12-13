# Licensed under a 3-clause BSD style license - see LICENSE
"""Synthetic generation of light curves."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as scipy_optimize
import scipy.signal as scipy_signal
import scipy.special as scipy_special
import scipy.stats as scipy_stats
from matplotlib.offsetbox import AnchoredText

from mutis.lib.signal import *

__all__ = ["Signal"]

log = logging.getLogger(__name__)


class Signal:
    """Analysis and generation of a signal.

    Class for a generic signal.

    Attributes
    ----------
    times : :class:`numpy.ndarray` or :class:`pandas.Series`
        Values of the time axis.
    values : :class:`numpy.ndarray` or :class:`pandas.Series`
        Values of the signal axis.
    dvalues : :class:`numpy.ndarray` or :class:`pandas.Series`
        Uncertainties on the values of the signal axis.

    Other Attributes
    ----------------

    fgen : :class:`str`
        Method to generate the synthetic signal.

    synth : :class:`numpy.ndarray`
        Synthetic signals generated with `fgen`.

    OU_theta : :class:`float`
        (For OU method) Parameter theta of the OU process.
    OU_mu : :class:`float`
        (For OU method) Parameter mu of the OU process.
    OU_sigma : :class:`float`
        (For OU method) Parameter sigma of the OU process.
    """

    def __init__(self, times, values, dvalues=None, fgen=None):
        self.times = np.array(times)
        self.values = np.array(values)
        self.fgen = fgen

        self.dvalues = np.array(dvalues) if dvalues is not None else None
        self.synth = None

        # TODO make attributes below specific of OU method / not the entire class
        self.OU_theta = None
        self.OU_mu = None
        self.OU_sigma = None

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        return ax.plot(self.times, self.values, ".-", lw=1, alpha=0.7)

    def gen_synth(self, samples):
        """Generate synthethic light curves for this signal.
        
        Generated curves are saved in `self.synth`. The times used are the same
        of this signal.
        
        Parameters
        ----------
        samples: :int: number of samples to generate.
        
        """

        self.synth = np.empty((samples, self.times.size))
        for n in range(samples):
            self.synth[n] = fgen_wrapper(fgen=self.fgen, t=self.times, y=self.values,
                         fgen_params={'theta':self.OU_theta, 'mu':self.OU_mu, 'sigma':self.OU_sigma})


    def OU_fit(self, bins=None, rang=None, a=1e-5, b=100):
        """Fit the signal to an OU stochastic process, using several statistical approaches.

        This function tries to fit the signal to an OU stochastic
        process using both basic curve fitting and the Maximum
        Likelihood Estimation method, and returns some plots of
        the signals and its properties, and the estimated parameters.
        """

        # TODO: make a generic fit method
        res = dict()

        y = self.values
        t = self.times

        dy = np.diff(y)
        dt = np.diff(t)

        # estimate sigma
        try:
            sigma_est = (np.nanmean(dy ** 2 / y[:-1] ** 2 / dt)) ** 0.5
            res["sigma_est"] = sigma_est
        except Exception as e:
            log.error(f"Could not estimate sigma: {e}")

        # plot histogram
        if bins is None:
            bins = int(y.size ** 0.5 / 1.5)  # bins='auto'
        if rang is None:
            rang = (np.percentile(y, 0), np.percentile(y, 99))

        p, x = np.histogram(y, density=True, bins=bins, range=rang)  # bins='sqrt')
        x = (x + np.roll(x, -1))[:-1] / 2.0

        plt.subplots()

        plt.hist(y, density=True, alpha=0.75, bins=bins, range=rang)
        plt.plot(x, p, "r-", alpha=0.5)

        anchored_text = AnchoredText(
            f"mean    {np.mean(y):.2g} \n "
            f"median  {np.median(y):.2g} \n "
            f"mode    {scipy_stats.mode(y)[0]:.2g} \n "
            f"std     {np.std(y):.2g} \n "
            f"var     {np.var(y):.2g}",
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
            log.error(f"Some error fitting with curve_fit {e}")

        # fit pdf with MLE
        # TODO: place outside as a helper class
        #       mu here is different than self.mu
        class OU(scipy_stats.rv_continuous):
            def _pdf(self, x, l, mu):
                return (
                    (l * mu) ** (1 + l)
                    / scipy_special.gamma(1 + l)
                    * np.exp(-l * mu / x)
                    / x ** (l + 2)
                )

        try:
            fit = OU(a=a, b=np.percentile(y, b)).fit(y, 1, 1, floc=0, fscale=1)
            # print('MLE fit: (l, mu)')
            # print(fit)
            x_c = np.linspace(0, 1.1 * np.max(x), 1000)
            plt.plot(x_c, self.pdf(x_c, fit[0], fit[1]), "k-.", label="MLE", alpha=0.8)
            res["MLE_fit"] = fit[:-2]
        except Exception as e:
            log.error(f"Some error fitting with MLE {e}")

        plt.legend(loc="lower right")
        # plt.show()

        # estimate theta (from curve_fit)
        try:
            res["th_est1"] = fit[0] * sigma_est ** 2 / 2
        except NameError as e:
            res["th_est1"] = None

        # estimate theta (from MLE)
        try:
            res["th_est2"] = popt[0] * sigma_est ** 2 / 2
        except NameError as e:
            res["th_est2"] = None

        return res
        
        
    def check_gen(self, fgen=None, fgen_params=None, fpsd="lombscargle", **axes):
        """Check the generation of synthetic signals.

        This function checks the generation of a synthetic light curve.
        
        Parameters
        ----------
        fgen: :str:
            A valid fgen method name.
        fgen_params: :dict:
            Parameters for fgen (e.g. for fgen='lc_gen_ou', a dict containing
            values for theta, mu and sigma).

        Returns
        -------
        axes: :tuple:
            Tuple of three axes, on which:
            - The first plot show both the original signal and the synthetic one.
            - The second plot shows the histogram of the values taken by both signals.
            - The third plot shows their PSD.
        """
        
        t, y = self.times, self.values

        y2 = fgen_wrapper(fgen=fgen, t=t, y=y, fgen_params=fgen_params)
        
        print(f"Original vs Synthethic:",
              f"mean: {np.mean(y)} / {np.mean(y2)}",
              f"std: {np.std(y)} / {np.std(y2)}", sep='\n')
        
        if ("ax1" not in axes) or ("ax2" not in axes) or ("ax3" not in axes):
            fig, (axes["ax1"], axes["ax2"], axes["ax3"]) = plt.subplots(
                nrows=1, ncols=3, figsize=(20, 4)
            )

        # plot the two signals
        axes["ax1"].plot(t, y, "b-", label="orig", lw=0.5, alpha=0.8)

        # ax1p = ax1.twinx()
        axes["ax1"].plot(t, y2, "r-", label="gen", lw=0.5, alpha=0.8)
        axes["ax1"].set_title("light curves")

        # plot their histogram
        bins = "auto"  # bins = int(y.size**0.5/1.5) #
        rang = (np.percentile(y, 0), np.percentile(y, 99))
        axes["ax2"].hist(y, density=True, color="b", alpha=0.4, bins=bins, range=rang)

        # ax2p = ax2.twinx()
        bins = "auto"  # bins = int(y.size**0.5/1.5) #
        rang = (np.percentile(y2, 0), np.percentile(y2, 99))
        axes["ax2"].hist(y2, density=True, color="r", alpha=0.4, bins=bins, range=rang)

        axes["ax2"].set_title("pdf")

        # plot their PSD
        if fpsd == "lombscargle":
            k = np.linspace(1e-3, self.values.size / 2, self.values.size // 2)
            freqs = k / 2 / np.pi

            pxx = scipy_signal.lombscargle(t, y, freqs, normalize=True)
            axes["ax3"].plot(freqs, pxx, "b-", lw=1, alpha=0.5)

            pxx2 = scipy_signal.lombscargle(t, y2, freqs, normalize=True)
            axes["ax3"].plot(freqs, pxx2, "r-", lw=1, alpha=0.5)

            axes["ax3"].set_xscale("log")
            # ax3.set_yscale('log')
        else:
            axes["ax3"].psd(y, color="b", lw=1, alpha=0.5)

            ax3p = axes["ax3"].twinx()
            ax3p.psd(y2, color="r", lw=1, alpha=0.5)

        axes["ax3"].set_title("PSD")

        return axes
    
    
        
    @staticmethod
    def pdf(xx, ll, mu):
        """Helper func to fit pdf as a curve."""
        return (
            (ll * mu) ** (1 + ll)
            / scipy_special.gamma(1 + ll)
            * np.exp(-ll * mu / xx)
            / xx ** (ll + 2)
        )
