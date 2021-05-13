# Licensed under a 3-clause BSD style license - see LICENSE
"""Methods for synthetic generation of light curves."""

import logging

import numpy as np
import scipy.signal as scipy_signal

__all__ = ["lc_gen_samp", "lc_gen_psd", "lc_gen_ou"]

log = logging.getLogger(__name__)


def lc_gen_samp(signs):
    """Generation by sampling np.random.choice with same mean and std"""

    return np.random.choice(signs, signs.size)


def lc_gen_psd(signs):
    """Generation using input PSD for light curves with similar PSD, mean and std."""

    f, pxx = scipy_signal.welch(signs)
    # fft2 = np.sqrt(2*pxx*pxx.size)*np.exp(1j*2*np.pi*np.random.randn(pxx.size))
    fft2 = np.sqrt(2 * pxx * pxx.size) * np.exp(1j * 2 * np.pi * np.random.random(pxx.size))
    s2 = np.fft.irfft(fft2, n=signs.size)
    a = signs.std() / s2.std()
    b = signs.mean() - a * s2.mean()
    s2 = a * s2 + b
    return s2


def lc_gen_psd_c(ts, signs, times):
    """Generation using interpolated PSD for light curves with similar PSD, mean and std."""

    f, p = scipy_signal.welch(signs, nperseg=ts.size / 2)
    fp = np.linspace(min(f), max(f), times.size // 2 + 1)
    pp = np.interp(fp, f, p)
    fft = np.sqrt(2 * pp * pp.size) * np.exp(1j * 2 * np.pi * np.random.random(pp.size))
    s2 = np.fft.irfft(fft, n=signs.size)
    a = signs.std() / s2.std()
    b = signs.mean() - a * s2.mean()
    s2 = a * s2 + b
    return s2


def lc_gen_ou(theta, mu, sigma, times, scale=None, loc=None):
    """Generation from an OU process integrating the stochastic differential equation."""

    width = 100 * times.size
    dt = (max(times) - min(times)) / width
    yv = np.empty(times.size)
    yv[0] = mu  # should get it from OU.rvs()!!!!
    for i in range(1, times.size):
        ti = times[i - 1]
        y = yv[i - 1]
        while ti < times[i]:
            y = y + dt * (theta * (mu - y) + sigma * y * np.random.randn() / np.sqrt(dt))
            ti = ti + dt
        yv[i] = y
    if scale is not None:
        yv = scale * yv / np.std(yv)
    if loc is not None:
        yv = yv - np.mean(yv) + loc
    return yv
