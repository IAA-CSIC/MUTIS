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


#### LCs with similar PSD, mean and std,
### uses the input psd's directly.
def lc_gen_psd_fft(pl1, sig, pl2):
    """
    This old version uses welch and fft, which is not valid for non-uniform times (see PSD tests for a comparison)
    """
    f, Pxx = sp.signal.welch(sig)
    #fft2 = np.sqrt(2*Pxx*Pxx.size)*np.exp(1j*2*pi*np.random.randn(Pxx.size))
    fft2 = np.sqrt(2*Pxx*Pxx.size)*np.exp(1j*2*pi*np.random.random(Pxx.size))
    sig2 = np.fft.irfft(fft2, n=sig.size)
    a = (sig.std()/sig2.std())
    b = sig.mean()-a*sig2.mean()
    sig2 = a*sig2+b
    return sig2


def lc_gen_psd_lombscargle(t, sig, pl2, N=None):
    sigp = sig
    tp = t

    if sig.size % 2 != 0:
        #print('Odd number')
        sigp = sig[:-1]
        tp = t[:-1]
    else:
        sigp = sig
        tp = t

    N = sigp.size
    #k = np.arange(-N/2,N/2) no bc sp.signal.lombscargle does not support freq zero:
    k = np.linspace(-N/2,N/2-1+1e-6,N)
    freqs = k/2/pi

    Pxx = sp.signal.lombscargle(tp, sigp, freqs)

    # construct random phase to get real signal:
    phase = np.random.random(Pxx.size//2)
    phase = np.concatenate((-np.flip(phase),[0], phase[:-1]))

    fft2 = np.sqrt(2*Pxx*Pxx.size)*np.exp(1j*2*pi*phase)

    sig2 = nfft.nfft((t-(t.max()+t.min())/2)/np.ptp(t), fft2, N, use_fft=True)/N

    #return sig2
    #fix small deviations
    a = (sig.std()/sig2.std())
    b = sig.mean()-a*sig2.mean()
    sig2 = a*sig2+b

    return sig2


def lc_gen_psd_nft(t, sig, pl2, N=None):
    k = np.arange(-t.size//2, t.size/2)
    N = k.size
    freqs = k/2/pi

    nft = nfft.nfft_adjoint((t-(t.max()+t.min())/2)/np.ptp(t), sig, N, use_fft=True)

    # construct random phase to get real signal:
    phase = np.random.random(N//2)
    phase = np.concatenate((-np.flip(phase),[0],phase[:-1]))

    fft2 = np.abs(nft)*np.exp(1j*2*pi*phase)
    sig2 = nfft.nfft((t-(t.max()+t.min())/2)/np.ptp(t), fft2, use_fft=True)/N

    #return sig2
    sig2 = np.real(sig2)     # np.real to fix small imaginary part from numerical error

    # fix small mean, std difference from numerical error
    a = (sig.std()/sig2.std())
    b = sig.mean()-a*sig2.mean()
    sig2 = a*sig2+b

    return sig2


### Set it to the best version
lc_gen_psd = lc_gen_psd_nft


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
