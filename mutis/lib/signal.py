# Licensed under a 3-clause BSD style license - see LICENSE
"""Methods for synthetic generation of light curves."""

import logging

import nfft
import numpy as np
import scipy.signal as scipy_signal

__all__ = [
    "lc_gen_samp",
    "lc_gen_psd_nft",
    "lc_gen_ou",
    "lc_gen_psd_lombscargle",
    "lc_gen_psd_fft",
    "lc_gen_psd_c",
]

log = logging.getLogger(__name__)


def lc_gen_samp(signs):
    """Generation by sampling np.random.choice with same mean and std"""

    return np.random.choice(signs, signs.size)


def lc_gen_ou(theta, mu, sigma, times, scale=None, loc=None):
    """Generation from an OU process integrating the stochastic differential equation."""

    width = 100 * times.size
    dt = (max(times) - min(times)) / width
    s2 = np.empty(times.size)
    s2[0] = mu  # should get it from OU.rvs()!!!!
    for i in range(1, times.size):
        ti = times[i - 1]
        y = s2[i - 1]
        while ti < times[i]:
            y = y + dt * (theta * (mu - y) + sigma * y * np.random.randn() / np.sqrt(dt))
            ti = ti + dt
        s2[i] = y
    if scale is not None:
        s2 = scale * s2 / np.std(s2)
    if loc is not None:
        s2 = s2 - np.mean(s2) + loc
    return s2


def lc_gen_psd_c(ts, values, times):
    """Generation using interpolated PSD for light curves with similar PSD, mean and std."""

    f, p = scipy_signal.welch(values, nperseg=ts.size / 2)
    fp = np.linspace(min(f), max(f), times.size // 2 + 1)
    pp = np.interp(fp, f, p)
    fft = np.sqrt(2 * pp * pp.size) * np.exp(1j * 2 * np.pi * np.random.random(pp.size))
    s2 = np.fft.irfft(fft, n=values.size)
    a = values.std() / s2.std()
    b = values.mean() - a * s2.mean()
    s2 = a * s2 + b
    return s2


def lc_gen_psd_fft(values):
    """Generation using Welch algorithm and the FFT, of synthetic signals with similar PSD, mean and std.

    Generates synthetic light curves using Lomb-Scargle algorithm
    to compute the power spectral density and the non-uniform fft
    to generate the signal."""

    # this is not valid for non-uniform times (see PSD tests for a comparison)
    f, pxx = scipy_signal.welch(values)
    # fft2 = np.sqrt(2*Pxx*Pxx.size)*np.exp(1j*2*pi*np.random.randn(Pxx.size))
    fft2 = np.sqrt(2 * pxx * pxx.size) * np.exp(1j * 2 * np.pi * np.random.random(pxx.size))
    s2 = np.fft.irfft(fft2, n=values.size)
    a = values.std() / s2.std()
    b = values.mean() - a * s2.mean()
    s2 = a * s2 + b
    return s2


def lc_gen_psd_lombscargle(times, values):
    """Generation using Lomb-Scargle algorithm and the non-uniform FFT, of synthetic signals with similar PSD, mean and std.

    Generates synthetic light curves using Lomb-Scargle algorithm
    to compute the power spectral density and the non-uniform fft
    to reconstruct the randomised signal."""

    if values.size % 2 != 0:
        sigp = values[:-1]
        tp = times[:-1]
    else:
        sigp = values
        tp = times

    offset = 1e-6
    n = sigp.size
    # k = np.arange(-n/2, n/2) no bc scipy_signal.lombscargle does not support freq zero
    k = np.linspace(-n / 2, n / 2 - 1 + offset, n)
    freqs = k / 2 / np.pi

    pxx = scipy_signal.lombscargle(tp, sigp, freqs)

    # build random phase to get real signal
    phase = np.random.random(pxx.size // 2)
    phase = np.concatenate((-np.flip(phase), [0], phase[:-1]))
    fft2 = np.sqrt(2 * pxx * pxx.size) * np.exp(1j * 2 * np.pi * phase)
    s2 = (
        nfft.nfft((times - (times.max() + times.min()) / 2) / np.ptp(times), fft2, n, use_fft=True)
        / n
    )

    # fix small deviations
    a = values.std() / s2.std()
    b = values.mean() - a * s2.mean()
    s2 = a * s2 + b

    return s2


def lc_gen_psd_nft(times, values):
    """Generation using the non-uniform FFT of synthetic signals with similar PSD, mean and std.

    Generates synthetic light curves using the non-uniform FFT to
    compute the power spectral density and to reconstruct the
    randomised signal."""

    k = np.arange(-times.size // 2, times.size / 2)
    n = k.size

    nft = nfft.nfft_adjoint(
        (times - (times.max() + times.min()) / 2) / np.ptp(times), values, n, use_fft=True
    )

    # build random phase to get real signal
    phase = np.random.random(n // 2)
    phase = np.concatenate((-np.flip(phase), [0], phase[:-1]))

    fft2 = np.abs(nft) * np.exp(1j * 2 * np.pi * phase)
    s2 = (
        nfft.nfft((times - (times.max() + times.min()) / 2) / np.ptp(times), fft2, use_fft=True) / n
    )
    s2 = np.real(s2)  # np.real to fix small imaginary part from numerical error

    # fix small mean, std difference from numerical error
    a = values.std() / s2.std()
    b = values.mean() - a * s2.mean()
    s2 = a * s2 + b

    return s2
