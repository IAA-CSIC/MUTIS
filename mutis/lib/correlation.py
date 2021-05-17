# Licensed under a 3-clause BSD style license - see LICENSE
"""Methods for correlation of light curves."""

import logging

import numpy as np

from mutis.lib.utils import get_grid

__all__ = ["kroedel_ab", "welsh_ab", "nindcf", "gen_times_rawab", "gen_times_uniform", "gen_times_canopy"]

log = logging.getLogger(__name__)


def kroedel_ab_p(t1, d1, t2, d2, t, dt):
    """Krolik & Edelson with adaptative binning."""
    t1m, t2m = get_grid(t1, t2)
    d1m, d2m = get_grid(d1, d2)
    mask = ~(((t - dt / 2) < (t2m - t1m)) & ((t2m - t1m) < (t + dt / 2)))
    udcf = (d1m - np.mean(d1)) * (d2m - np.mean(d2)) / np.std(d1) / np.std(d2)
    udcf = np.ma.masked_where(mask, udcf)

    return np.ma.mean(udcf)


def kroedel_ab(t1, d1, t2, d2, t, dt):
    """Krolik & Edelson with adaptative binning.

    Description goes here.

    Parameters
    ----------
    t1 : :class:`~numpy.ndarray`
        Parameter description.
    d1 : :class:`~numpy.ndarray`
        Parameter description.
    t2 : :class:`~numpy.ndarray`
        Parameter description.
    d2 : :class:`~numpy.ndarray`
        Parameter description.
    t  : :class:`~numpy.ndarray`
        Parameter description.
    dt : :class:`~numpy.ndarray`
        Parameter description.

    Returns
    -------
    udcf : :class:`~float`
        Description here.

    Examples
    --------
    Provide examples as below because they could be tested:

    >>> import numpy as np
    >>> from mutis.lib.correlation import kroedel_ab
    >>> t1 = np.linspace(1, 10, 100); s1 = np.sin(t1)
    >>> t2 = np.linspace(1, 10, 100); s2 = np.cos(t2)
    >>> t = np.linspace(1, 10, 100);  dt = np.tan(t2)
    >>> kroedel_ab_p(t1, d1, t2, d2, t, dt)
    """
    if dt.size != t.size:
        print("Error, t and dt not the same size")
        return -1
    res = np.array([])
    for i in range(t.size):
        res = np.append(res, kroedel_ab_p(t1, d1, t2, d2, t[i], dt[i]))
    return res


def welsh_ab_p(t1, d1, t2, d2, t, dt):
    """Welsh with adaptative binning."""
    t1m, t2m = get_grid(t1, t2)
    d1m, d2m = np.meshgrid(d1, d2)
    msk = ((t - dt / 2) < (t2m - t1m)) & ((t2m - t1m) < (t + dt / 2))
    udcf = (d1m - np.mean(d1m[msk])) * (d2m - np.mean(d2m[msk])) / np.std(d1m[msk]) / np.std(d2m[msk])
    return np.mean(udcf[msk])


def welsh_ab(t1, d1, t2, d2, t, dt):
    """Description goes here."""
    if t.size != dt.size:
        log.error("Error, t and dt not the same size")
        return -1
    if t1.size != d1.size:
        log.error("Error, t1 and d1 not the same size")
        return -1
    if t2.size != d2.size:
        log.error("Error, t2 and d2 not the same size")
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


def ndcf(x, y):
    """Description goes here."""
    x = (x - np.mean(x)) / np.std(x) / len(x)
    y = (y - np.mean(y)) / np.std(y)
    return np.correlate(y, x, "full")


def nindcf(t1, s1, t2, s2):
    """Description goes here."""
    dt = np.max([(t1.max() - t1.min()) / t1.size, (t2.max() - t2.min()) / t2.size])
    n1 = np.int(np.ptp(t1) / dt * 10.0)
    n2 = np.int(np.ptp(t1) / dt * 10.0)
    s1i = np.interp(np.linspace(t1.min(), t1.max(), n1), t1, s1)
    s2i = np.interp(np.linspace(t2.min(), t2.max(), n2), t2, s2)
    return ndcf(s1i, s2i)


def gen_times_rawab(t1, t2, dt0=None, ndtmax=1.0, nbinsmin=121, force=None):
    """Returns t, dt for use with adaptative binning methods."""

    # Sensible values for these parameters must be found by hand, and depend
    # on the characteristic of input data.
    #
    # dt0:
    #     minimum bin size, also used as step in a.b.
    #         default: dt0 = 0.25*(tmax-tmin)/np.sqrt(t1.size*t2.size+1)
    #     (more or less a statistically reasonable binning,
    #     to increase precision)
    # ndtmax:
    #     Maximum size of bins (in units of dt0).
    #     default: 1.0
    # nbinsmin:
    #     if the data has a lot of error, higher values are needed
    #     to soften the correlation beyond spurious variability.
    #         default: 121 (11x11)

    # tmin = -(np.min([t1.max(),t2.max()]) - np.max([t1.min(),t2.min()]))
    tmax = +(np.max([t1.max(), t2.max()]) - np.min([t1.min(), t2.min()]))
    tmin = -tmax

    if dt0 is None:
        dt0 = 0.25 * (tmax - tmin) / np.sqrt(t1.size * t2.size + 1)

    t = np.array([])
    dt = np.array([])
    nb = np.array([])
    t1m, t2m = np.meshgrid(t1, t2)

    ti = tmin
    tf = ti + dt0

    while tf < tmax:
        tm = (ti + tf) / 2
        dtm = tf - ti
        nbins = np.sum((((tm - dtm / 2) < (t2m - t1m)) & ((t2m - t1m) < (tm + dtm / 2))))
        if dtm <= dt0 * ndtmax:
            if nbins >= nbinsmin:
                t = np.append(t, tm)
                dt = np.append(dt, dtm)
                nb = np.append(nb, nbins)
                ti, tf = tf, tf + dt0
            else:
                tf = tf + 0.1 * dt0  # try small increments
        else:
            ti, tf = tf, tf + dt0

    # force zero to appear in t ##
    if force is None:
        force = [0]
    for tm in force:
        dtm = dt0 / 2
        nbins = np.sum((((tm - dtm / 2) < (t2m - t1m)) & ((t2m - t1m) < (tm + dtm / 2))))
        while dtm <= dt0 * ndtmax:
            if nbins >= nbinsmin:
                t = np.append(t, tm)
                dt = np.append(dt, dtm)
                nb = np.append(nb, nbins)
                break
            else:
                dtm = dtm + dt0

    idx = np.argsort(t)
    t = t[idx]
    dt = dt[idx]
    nb = nb[idx]

    return t, dt, nb


def gen_times_uniform(t1, t2, tmin=None, tmax=None, nbinsmin=121, N=200):
    """Description goes here."""

    if tmax is None:
        tmax = +(np.max([t1.max(), t2.max()]) - np.min([t1.min(), t2.min()]))
    if tmin is None:
        tmin = -tmax

    t = np.linspace(tmin, tmax, N)
    dtm = (tmax - tmin) / N
    dt = np.full(t.shape, dtm)
    nb = np.empty(t.shape)
    t1m, t2m = np.meshgrid(t1, t2)

    for im, tm in enumerate(t):
        nb[im] = np.sum((((tm - dtm / 2) < (t2m - t1m)) & ((t2m - t1m) < (tm + dtm / 2))))
    idx = nb < nbinsmin
    t = np.delete(t, idx)
    dt = np.delete(dt, idx)
    nb = np.delete(nb, idx)

    return t, dt, nb


def gen_times_canopy(t1, t2, dtmin=0.01, dtmax=0.5, nbinsmin=500, nf=0.5):
    """Description goes here."""

    t1m, t2m = np.meshgrid(t1, t2)
    tmax = +(np.max([t1.max(), t2.max()]) - np.min([t1.min(), t2.min()]))
    tmin = -tmax

    def _comp_nb(tis, dtis):
        nbis = np.empty(len(tis))
        for j in range(len(tis)):
            nbis[j] = np.sum((((tis[j] - dtis[j] / 2) < (t2m - t1m)) & ((t2m - t1m) < (tis[j] + dtis[j] / 2))))
        return nbis

    t = np.linspace(tmin, tmax, int((tmax - tmin) / dtmin))
    dt = np.full(t.size, np.ptp(t) / t.size)
    nb = _comp_nb(t, dt)
    for _ in range(int(np.log(dtmax / dtmin) / np.log(1 / nf))):
        idx = nb < nbinsmin
        ts, dts, nbs = t, dt, nb
        t, dt = np.copy(ts), np.copy(dts)

        n_grp = 0
        grps = (np.where(np.diff(np.concatenate(([False], idx, [False]), dtype=int)) != 0)[0]).reshape(-1, 2)
        for i_grp, grp in enumerate(grps):
            ar = grp[0]
            a = t[grp[0] - 1] if (grp[0] > 0) else t[grp[0]]
            br = grp[1] - 1
            b = t[grp[1]] if (grp[1] < t.size - 1) else t[grp[1] - 1]
            if (br - ar) < 8:
                n = br - ar + 1 if br - ar >= 1 else br - ar + 2
                tins = np.linspace(a, b, n, endpoint=False)[1:]
                ts = np.delete(ts, np.arange(ar, br + 1) - n_grp)
                dts = np.delete(dts, np.arange(ar, br + 1) - n_grp)
                ts = np.insert(ts, grp[0] - n_grp, tins)
                dts = np.insert(dts, grp[0] - n_grp, np.full(n - 1, (b - a) / (n - 1)))
                if br - ar >= 1:
                    n_grp += 1
            else:
                n = int(nf * (br - ar + 1))
                tins = np.linspace(a, b, n, endpoint=False)[1:]
                ts = np.delete(ts, np.arange(ar, br + 1) - n_grp)
                dts = np.delete(dts, np.arange(ar, br + 1) - n_grp)
                ts = np.insert(ts, grp[0] - n_grp, tins)
                dts = np.insert(dts, grp[0] - n_grp, np.full(n - 1, (b - a) / (n - 1)))
                if br - ar >= 1:
                    n_grp = n_grp + (grp[1] - grp[0] - n) + 1
        t = ts
        dt = dts
        nb = _comp_nb(t, dt)

    idx = nb < nbinsmin
    t = np.delete(t, idx)
    dt = np.delete(dt, idx)
    nb = np.delete(nb, idx)

    return t, dt, nb


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
