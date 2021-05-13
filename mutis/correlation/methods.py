# Licensed under a 3-clause BSD style license - see LICENSE
"""Methods for correlation of light curves."""

import logging
import numpy as np

from mutis.utils.utils import get_grid

__all__ = ["get_times"]

log = logging.getLogger(__name__)




def get_times(t1, t2, dt0=None, ndtmax=0.9, nbinsmin=121):
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

    # tmin = -(np.min([t1.max(),t2.max()]) - np.max([t1.min(),t2.min()]))
    tmax = +(np.max([t1.max(), t2.max()]) - np.min([t1.min(), t2.min()]))
    tmin = -tmax

    if dt0 is None:
        # dt0 = 1*(tmax-tmin)/(t1.size+t2.size-1)
        # dt0 = 1.0*(tmax-tmin)/np.sqrt(t1.size*t2.size+1)
        # dt0 = 30/365; ndtmax=2.9; nbinsmin=12*12
        # dt0 = 0.25; ndtmax=10; nbinsmin=5
        dt0 = 0.25 * (tmax - tmin) / np.sqrt(t1.size * t2.size + 1)

    t = np.array([])
    dt = np.array([])
    nb = np.array([])
    t1m, t2m = np.meshgrid(t1, t2)
    ti = tmin
    tf = ti + dt0

    while tf < tmax:
        tm = (ti + tf) / 2
        dtm = (tf - ti) / 2
        nbins = np.sum((((tm - dtm / 2) < (t2m - t1m)) & ((t2m - t1m) < (tm + dtm / 2))))
        if dtm < dt0 * ndtmax:
            if nbins > nbinsmin:
                t = np.append(t, tm)
                dt = np.append(dt, dtm)
                nb = np.append(nb, nbins)
                ti, tf = tf, tf + dt0
            else:
                tf = tf + dt0
        else:
            ti, tf = tf, tf + dt0
    return t, dt, nb


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
