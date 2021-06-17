# Licensed under a 3-clause BSD style license - see LICENSE
"""Methods for correlation of light curves."""

import logging

import numpy as np

from mutis.lib.utils import get_grid

__all__ = ["kroedel_ab", "welsh_ab", "nindcf", "gen_times_rawab", "gen_times_uniform", "gen_times_canopy"]

log = logging.getLogger(__name__)



def kroedel_ab_p(t1, d1, t2, d2, t, dt):
    """Helper function for kroedel_ab()"""

    t1m, t2m = get_grid(t1, t2)
    d1m, d2m = np.meshgrid(d1, d2)

    mask = ((t - dt / 2) < (t2m - t1m)) & ((t2m - t1m) < (t + dt / 2))

    udcf = (d1m - np.mean(d1)) * (d2m - np.mean(d2)) / np.std(d1) / np.std(d2)

    return np.mean(udcf[mask])


def kroedel_ab(t1, d1, t2, d2, t, dt):
    """Krolik & Edelson (1988) correlation with adaptative binning.

    This function implements the correlation function proposed by
    Krolik & Edelson (1988), which allows for the computation of
    the correlation for -discrete- signals non-uniformly sampled
    in time.

    Parameters
    ----------
    t1 : :class:`~numpy.ndarray`
        Times corresponding to the first signal.
    d1 : :class:`~numpy.ndarray`
        Values of the first signal.
    t2 : :class:`~numpy.ndarray`
        Times corresponding to the second signal.
    d2 : :class:`~numpy.ndarray`
        Values of the second signal.
    t  : :class:`~numpy.ndarray`
        Times on which to compute the correlation (binning).
    dt : :class:`~numpy.ndarray`
        Size of the bins on which to compute the correlation.

    Returns
    -------
    res : :class:`~numpy.ndarray` (size `len(t)`)
        Values of the correlation at the times `t`.

    Examples
    --------
    An example of raw usage would be:

    >>> import numpy as np
    >>> from mutis.lib.correlation import kroedel_ab
    >>> t1 = np.linspace(1, 10, 100); s1 = np.sin(t1)
    >>> t2 = np.linspace(1, 10, 100); s2 = np.cos(t2)
    >>> t = np.linspace(1, 10, 100);  dt = np.full(t.shape, 0.1)
    >>> kroedel_ab_p(t1, d1, t2, d2, t, dt)

    However, it is recommended to be used as expalined in the
    standard MUTIS' workflow notebook.
    """

    if t.size != dt.size:
        log.error("Error, t and dt not the same size")
        return False
    if t1.size != d1.size:
        log.error("Error, t1 and d1 not the same size")
        return False
    if t2.size != d2.size:
        log.error("Error, t2 and d2 not the same size")
        return False

    res = np.empty(t.size)
    for i in range(t.size):
        res[i] = kroedel_ab_p(t1, d1, t2, d2, t[i], dt[i])
    return res



def welsh_ab_p(t1, d1, t2, d2, t, dt):
    """Helper function for welsh_ab()"""

    t1m, t2m = get_grid(t1, t2)
    d1m, d2m = np.meshgrid(d1, d2)

    msk = ((t - dt / 2) < (t2m - t1m)) & ((t2m - t1m) < (t + dt / 2))

    udcf = (d1m - np.mean(d1m[msk])) * (d2m - np.mean(d2m[msk])) / np.std(d1m[msk]) / np.std(d2m[msk])

    return np.mean(udcf[msk])


def welsh_ab(t1, d1, t2, d2, t, dt):
    """Welsh (1999) correlation with adaptative binning.

    This function implements the correlation function proposed
    by Welsh (1999), which allows for the computation of the correlation
    for -discrete- signals non-uniformly sampled in time.

    Parameters
    ----------
    t1 : :class:`~numpy.ndarray`
        Times corresponding to the first signal.
    d1 : :class:`~numpy.ndarray`
        Values of the first signal.
    t2 : :class:`~numpy.ndarray`
        Times corresponding to the second signal.
    d2 : :class:`~numpy.ndarray`
        Values of the second signal.
    t  : :class:`~numpy.ndarray`
        Times on which to compute the correlation (binning).
    dt : :class:`~numpy.ndarray`
        Size of the bins on which to compute the correlation.

    Returns
    -------
    res : :class:`~numpy.ndarray` (size `len(t)`)
        Values of the correlation at the times `t`.

    Examples
    --------
    An example of raw usage would be:

    >>> import numpy as np
    >>> from mutis.lib.correlation import welsh_ab
    >>> t1 = np.linspace(1, 10, 100); s1 = np.sin(t1)
    >>> t2 = np.linspace(1, 10, 100); s2 = np.cos(t2)
    >>> t = np.linspace(1, 10, 100);  dt = np.full(t.shape, 0.1)
    >>> welsh_ab_p(t1, d1, t2, d2, t, dt)

    However, it is recommended to be used as expalined in the
    standard MUTIS' workflow notebook.
    """

    if t.size != dt.size:
        log.error("Error, t and dt not the same size")
        return False
    if t1.size != d1.size:
        log.error("Error, t1 and d1 not the same size")
        return False
    if t2.size != d2.size:
        log.error("Error, t2 and d2 not the same size")
        return False

    # res = np.array([])
    res = np.empty(t.size)
    for i in range(t.size):
        res[i] = welsh_ab_p(t1, d1, t2, d2, t[i], dt[i])
    return res



def ndcf(x, y):
    """Computes the normalised correlation of two discrete signals (ignoring times)."""

    x = (x - np.mean(x)) / np.std(x) / len(x)
    y = (y - np.mean(y)) / np.std(y)
    return np.correlate(y, x, "full")



def nindcf(t1, s1, t2, s2):
    """Computes the normalised correlation of two discrete signals (interpolating them)."""

    dt = np.max([(t1.max() - t1.min()) / t1.size, (t2.max() - t2.min()) / t2.size])
    n1 = np.int(np.ptp(t1) / dt * 10.0)
    n2 = np.int(np.ptp(t1) / dt * 10.0)
    s1i = np.interp(np.linspace(t1.min(), t1.max(), n1), t1, s1)
    s2i = np.interp(np.linspace(t2.min(), t2.max(), n2), t2, s2)
    return ndcf(s1i, s2i)



def gen_times_rawab(t1, t2, dt0=None, ndtmax=1.0, nbinsmin=121, force=None):
    """LEGACY. Returns t, dt for use with adaptative binning methods.

    Uses a shitty algorithm to find a time binning in which each bin contains
    a minimum of points (specified by `nbinsmin`, with an starting bin size
    (`dt0`) and a maximum bin size (`ndtmax*dt0`).

    The algorithms start at the first time bin, and enlarges the bin size
    until it has enough points or it reaches the maximum length, then creates
    another starting at that point.

    If `force` is True, then it discards the created bins on which there are
    not enough points.
    """

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


def gen_times_uniform(t1, t2, tmin=None, tmax=None, nbinsmin=121, n=200):
    """Returns an uniform t, dt time binning for use with adaptative binning methods.

    The time interval on which the correlation is defined is split in
    `n` bins. Bins with a number of point less than `nbinsmin` are discarded.

    Parameters
    ----------
    t1 : :py:class:`np.ndarray`
        Times of the first signal.
    t2 : :py:class:`np.ndarray`
        Times of the second signal.
    tmin : :py:class:`~float`
        Start of the time intervals (if not specified, start of the interval on which the correlation is define).
    tmax : :py:class:`~float`
        End of the time intervals (if not specified, end of the interval on which the correlation is define).
    nbinsmin : :py:class:`~float`
        Minimum of points falling on each bin.
    n : :py:class:`~float`
        Number of bins in which to split (needs not to be the number of bins returned).

    Returns
    -------
    t : :class:`~numpy.ndarray`
        Time binning on which to compute the correlation.
    dt : :class:`~numpy.ndarray`
        Size of the bins defined by `t`
    nb : :class:`~numpy.ndarray`
        Number of points falling on each bin defined by `t` and `dt`.
    """

    if tmax is None:
        tmax = +(np.max([t1.max(), t2.max()]) - np.min([t1.min(), t2.min()]))
    if tmin is None:
        tmin = -tmax

    t = np.linspace(tmin, tmax, n)
    dtm = (tmax - tmin) / n
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
    """Returns a non-uniform t, dt time binning for use with adaptative binning methods.

    This cumbersome algorithm does more or less the following:
    1) Divides the time interval on which the correlation is defined in
    the maximum number of points (minimum bin size defined by `dtmin`).
    2) Checks the number of point falling on each bin.
    3) If there are several consecutive intervals with a number of points
    over `nbinsmin`, it groups them (reducing the number of points
    exponentially as defined by `nf`, if the number of intervals in the
    group is high, or one by one if it is low.)
    4) Repeat until APPROXIMATELY we have reached intervals of size `dtmax`.

    How the exact implementation works, I forgot! But the results are more
    or less nice...

    Parameters
    ----------
    t1 : :py:class:`np.ndarray`
        Times of the first signal.
    t2 : :py:class:`np.ndarray`
        Times of the second signal.
    dtmin : :py:class:`~float`
        Start of the time intervals (if not specified, start of the
        interval on which the correlation is define).
    dtmax : :py:class:`~float`
        End of the time intervals (if not specified, end of the interval
        on which the correlation is define).
    nbinsmin : :py:class:`~float`
        Minimum of points falling on each bin.
    nf : :py:class:`~float`
        How fast are the intervals divided.

    Returns
    -------
    t : :class:`~numpy.ndarray`
        Time binning on which to compute the correlation.
    dt : :class:`~numpy.ndarray`
        Size of the bins defined by `t`
    nb : :class:`~numpy.ndarray`
        Number of points falling on each bin defined by `t` and `dt`.
    """

    t1m, t2m = np.meshgrid(t1, t2)

    def _comp_nb(t, dt):
        nb = np.empty(len(t))
        for i in range(len(t)):
            nb[i] = np.sum((((t[i] - dt[i] / 2) < (t2m - t1m)) & ((t2m - t1m) < (t[i] + dt[i] / 2))))
        return nb

    tmax = +(np.max([t1.max(), t2.max()]) - np.min([t1.min(), t2.min()]))
    tmin = -tmax

    t = np.linspace(tmin, tmax, int((tmax - tmin) / dtmin))
    dt = np.full(t.size, np.ptp(t) / t.size)
    nb = _comp_nb(t, dt)

    k = 0
    while k < int(np.log(dtmax / dtmin) / np.log(1 / nf)):
        k = k + 1

        idx = nb < nbinsmin

        ts, dts, nbs = t, dt, nb

        t, dt = np.copy(ts), np.copy(dts)

        n_grp = 0
        grps = (np.where(np.diff(np.concatenate(([False], idx, [False]), dtype=int)) != 0)[0]).reshape(-1, 2)
        for i_grp, grp in enumerate(grps):
            if grp[0] > 0:
                ar = grp[0]
                a = t[grp[0] - 1]
            else:
                ar = grp[0]
                a = t[grp[0]]

            if grp[1] < t.size - 1:
                br = grp[1] - 1
                b = t[grp[1]]
            else:
                br = grp[1] - 1
                b = t[grp[1] - 1]

            if (br - ar) < 8:
                if br - ar >= 1:
                    n = br - ar + 1
                else:
                    n = br - ar + 2

                tins = np.linspace(a, b, n, endpoint=False)[1:]

                ts = np.delete(ts, np.arange(ar, br + 1) - n_grp)
                dts = np.delete(dts, np.arange(ar, br + 1) - n_grp)

                ts = np.insert(ts, grp[0] - n_grp, tins)
                dts = np.insert(dts, grp[0] - n_grp, np.full(n - 1, (b - a) / (n - 1)))

                if br - ar >= 1:
                    n_grp = n_grp + 1
                else:
                    pass
            else:
                n = int(nf * (br - ar + 1))

                tins = np.linspace(a, b, n, endpoint=False)[1:]

                ts = np.delete(ts, np.arange(ar, br + 1) - n_grp)
                dts = np.delete(dts, np.arange(ar, br + 1) - n_grp)

                ts = np.insert(ts, grp[0] - n_grp, tins)
                dts = np.insert(dts, grp[0] - n_grp, np.full(n - 1, (b - a) / (n - 1)))

                if br - ar >= 1:
                    n_grp = n_grp + (grp[1] - grp[0] - n) + 1
                else:
                    pass

        t = ts
        dt = dts
        nb = _comp_nb(t, dt)

    idx = nb < nbinsmin

    t = np.delete(t, idx)
    dt = np.delete(dt, idx)
    nb = np.delete(nb, idx)

    return t, dt, nb
