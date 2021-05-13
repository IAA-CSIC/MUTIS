# Licensed under a 3-clause BSD style license - see LICENSE
"""Methods for correlation of light curves."""

import logging

import numpy as np

from mutis.utils.utils import get_grid

__all__ = ["kroedel_ab", "welsh_ab", "nindcf"]

log = logging.getLogger(__name__)


#  Krolik & Edelson with adaptative bining
def kroedel_ab_p(t1, d1, t2, d2, t, dt):
    t1m, t2m = get_grid(t1, t2)
    d1m, d2m = get_grid(d1, d2)

    mask = ~(((t - dt / 2) < (t2m - t1m)) & ((t2m - t1m) < (t + dt / 2)))

    udcf = (d1m - np.mean(d1)) * (d2m - np.mean(d2)) / np.std(d1) / np.std(d2)
    udcf = np.ma.masked_where(mask, udcf)

    return np.ma.mean(udcf)


def kroedel_ab(t1, d1, t2, d2, t, dt):
    if dt.size != t.size:
        print("Error, t and dt not the same size")
        return -1

    res = np.array([])
    for i in range(t.size):
        res = np.append(res, kroedel_ab_p(t1, d1, t2, d2, t[i], dt[i]))
    return res


# Welsh with adaptative bining
def welsh_ab_p(t1, d1, t2, d2, t, dt):
    t1m, t2m = get_grid(t1, t2)
    d1m, d2m = np.meshgrid(d1, d2)

    msk = ((t - dt / 2) < (t2m - t1m)) & ((t2m - t1m) < (t + dt / 2))

    udcf = (d1m - np.mean(d1m[msk])) * (d2m - np.mean(d2m[msk])) / np.std(d1m[msk]) / np.std(d2m[msk])

    return np.mean(udcf[msk])


def welsh_ab(t1, d1, t2, d2, t, dt):
    if t.size != dt.size:
        print("Error, t and dt not the same size")
        return -1
    if t1.size != d1.size:
        print("Error, t1 and d1 not the same size")
        return -1
    if t2.size != d2.size:
        print("Error, t2 and d2 not the same size")
        return -1

    # res = np.array([])
    res = np.empty(t.size)
    for i in range(t.size):
        res[i] = welsh_ab_p(t1, d1, t2, d2, t[i], dt[i])
    return res


def fkroedel(t1, d1, t2, d2, t, dt=None):
    """Krolik & Edelson 1988."""

    if dt is None:
        dt = (np.max([t1.max(), t2.max()]) - np.min([t1.min(), t2.min()])) / np.min([t1.size, t2.size])
        # print('dt is {:.3f}'.format(dt))

    t1m, t2m = get_grid(t1, t2)
    d1m, d2m = get_grid(d1, d2)

    mask = ~(((t - dt / 2) < (t2m - t1m)) & ((t2m - t1m) < (t + dt / 2)))
    udcf = (d1m - np.mean(d1)) * (d2m - np.mean(d2)) / np.std(d1) / np.std(d2)
    udcf = np.ma.masked_where(mask, udcf)

    if np.sum(mask) < 12:
        return np.nan

    return np.ma.mean(udcf)


def fwelsh(t1, d1, t2, d2, t, dt=None):
    """Welsh 1999."""

    if dt is None:
        dt = (np.max([t1.max(), t2.max()]) - np.min([t1.min(), t2.min()])) / np.min([t1.size, t2.size])
        # print('dt is {:.3f}'.format(dt))

    t1m, t2m = get_grid(t1, t2)
    d1m, d2m = get_grid(d1, d2)

    msk = ~(((t - dt / 2) < (t2m - t1m)) & ((t2m - t1m) < (t + dt / 2)))
    d1msk = np.ma.array(d1m, mask=msk)
    d2msk = np.ma.array(d2m, mask=msk)

    udcf = (d1m - d1msk.mean()) * (d2m - d2msk.mean()) / d1msk.std() / d2msk.std()
    udcf = np.ma.masked_where(msk, udcf)

    return np.mean(udcf)


# vectorize funcions
kroedel = np.vectorize(fkroedel, excluded=(0, 1, 2, 3, 5), otypes=[np.float])
welsh = np.vectorize(fwelsh, excluded=(0, 1, 2, 3, 5), otypes=[np.float])


def nindcf(t1, s1, t2, s2):
    """Implement normalization and interpolation over numpy correlate function."""
    # numpy correlate function is not designed for unevenly spaced data
    # correlation C(tau), where tau goes from
    # -( np.max([t1.max(),t2.max()]) - np.min([t1.min(),t2.min()]) )
    # to
    # +( np.min([t1.max(),t2.max()]) - np.max([t1.min(),t2.min()]) )

    s1i = np.interp(np.linspace(t1.min(), t1.max(), t1.size), t1, s1)
    s2i = np.interp(np.linspace(t2.min(), t2.max(), t2.size), t2, s2)
    x = (s1i - np.mean(s1i)) / np.std(s1i) / len(s1i)
    y = (s2i - np.mean(s2i)) / np.std(s2i)
    return np.correlate(x, y, "full")


#
# MC estimators of uncertainties
#

# def MC_corr_err(t1, s1, t1p, t2, s2, t2p, t, dt, fcorr, fgen, N=400):
#     """ MC percentiles generation."""
#     mc_corr = np.empty((N, t.size))
#     for n in range(0, N):
#         mc_corr[n] = fcorr(t1, fgen(t1, s1, t1p), t2, fgen(t2, s2, t2p), t, dt)
#
#     uppp, lowpp = np.percentile(mc_corr, [0.135, 99.865], axis=0)
#     upp, lowp = np.percentile(mc_corr, [2.28, 97.73], axis=0)
#     up, low = np.percentile(mc_corr, [15.865, 84.135], axis=0)
#
#     return (up, low), (upp, lowp), (uppp, lowpp)
#
#
# def MC_sig_err(t1, s1, ds1, t2, s2, ds2, t, dt, fcorr, fgen, N=400):
#     """ MC error generation."""
#     mc_corr2 = np.empty((N, t.size))
#     for n in range(0, N):
#         mc_corr2[n] = fcorr(t1, s1 + ds1 * np.random.randn(s1.size),
#                             t2, s2 + ds2 * np.random.randn(s2.size), t, dt)
#
#     uppp2, lowpp2 = np.percentile(mc_corr2, [0.135, 99.865], axis=0)
#     upp2, lowp2 = np.percentile(mc_corr2, [2.28, 97.73], axis=0)
#     up2, low2 = np.percentile(mc_corr2, [15.865, 84.135], axis=0)
#
#     return (up2, low2), (upp2, lowp2), (uppp2, lowpp2)
