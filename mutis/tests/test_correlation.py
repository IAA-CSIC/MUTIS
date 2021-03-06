import numpy as np
import pytest
from numpy.testing import assert_allclose

from mutis.correlation import Correlation
from mutis.signal import Signal


@pytest.fixture
def corr():
    times1 = np.linspace(2, 6, 40)
    values1 = np.sin(times1)
    dvalues1 = 0.05 * values1
    times2 = np.linspace(8, 12, 40)
    dvalues2 = 0.05 * values1
    values2 = np.sin(times2)
    signal1 = Signal(times1, values1, fgen="lc_gen_psd_nft")
    signal2 = Signal(times2, values2, fgen="lc_gen_psd_nft")
    signal3 = Signal(times1, values1, dvalues1, fgen="lc_gen_psd_nft")
    signal4 = Signal(times2, values2, dvalues2, fgen="lc_gen_psd_nft")

    return {
        "fail": Correlation(signal1, signal2, "fail"),
        "welsh": Correlation(signal1, signal2, "welsh"),
        "kroedel": Correlation(signal1, signal2, "kroedel"),
        "welsh_old": Correlation(signal1, signal2, "welsh_old"),
        "kroedel_old": Correlation(signal1, signal2, "kroedel_old"),
        "numpy": Correlation(signal1, signal2, "numpy"),
        "kroedel_uncert": Correlation(signal3, signal4, "kroedel"),
        "welsh_uncert": Correlation(signal3, signal4, "welsh"),
    }


def test_gen_synth(corr):
    corr["welsh"].gen_synth(10)
    assert len(corr["welsh"].signal1.synth) == 10
    assert len(corr["welsh"].signal2.synth) == 10


def test_plot_signals(corr):
    corr["welsh"].plot_signals()


def test_gen_times(corr):
    with pytest.raises(Exception):
        corr["fail"].gen_corr()
    corr["welsh"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    assert_allclose(corr["welsh"].times[0], 2.160, rtol=1e-3)
    assert corr["welsh"].dts[0] == 0.1
    assert corr["welsh"].nb[0] == 3
    corr["welsh"].gen_times(ftimes="rawab", dt0=0.1, ndtmax=3, nbinsmin=3)
    assert_allclose(corr["welsh"].times[0], 2.040, rtol=1e-3)
    assert_allclose(corr["welsh"].dts[0], 0.14, rtol=1e-3)
    assert corr["welsh"].nb[0] == 3
    corr["welsh"].gen_times(ftimes="uniform", tmin=0.1, tmax=3, nbinsmin=3)
    assert_allclose(corr["welsh"].times[0], 2.198, rtol=1e-3)
    assert_allclose(corr["welsh"].dts[0], 0.0145, rtol=1e-3)
    assert corr["welsh"].nb[0] == 3
    corr["welsh"].gen_times(ftimes="numpy")
    assert corr["welsh"].times[0] == 2
    assert corr["welsh"].dts[0] == 0.01
    assert corr["welsh"].nb[0] == 3
    with pytest.raises(Exception):
        corr["fail"].gen_times(ftimes="fail")


def test_plot_times(corr):
    corr["welsh"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["welsh"].plot_times(rug=True)


def test_gen_corr(corr):
    corr["numpy"].gen_times("numpy")
    corr["numpy"].gen_synth(10)
    corr["numpy"].gen_corr()
    # TODO: currently fcorr numpy's way of computing the times
    # is arbitrary and does little sense, when it is fixed it
    # will be sensible to check the shapes.
    # assert np.shape(corr["numpy"].l1s) == (2, 77)
    # assert np.shape(corr["numpy"].l2s) == (2, 77)
    # assert np.shape(corr["numpy"].l3s) == (2, 77)

    corr["welsh"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["welsh"].gen_synth(10)
    corr["welsh"].gen_corr()
    assert np.shape(corr["welsh"].l1s) == (2, 77)
    assert np.shape(corr["welsh"].l2s) == (2, 77)
    assert np.shape(corr["welsh"].l3s) == (2, 77)

    corr["kroedel"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["kroedel"].gen_synth(10)
    corr["kroedel"].gen_corr()
    assert np.shape(corr["kroedel"].l1s) == (2, 77)
    assert np.shape(corr["kroedel"].l2s) == (2, 77)
    assert np.shape(corr["kroedel"].l3s) == (2, 77)

    corr["kroedel_uncert"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["kroedel_uncert"].gen_synth(10)
    corr["kroedel_uncert"].gen_corr()
    assert np.shape(corr["kroedel_uncert"].l1s) == (2, 77)
    assert np.shape(corr["kroedel_uncert"].l2s) == (2, 77)
    assert np.shape(corr["kroedel_uncert"].l3s) == (2, 77)

    corr["welsh_uncert"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["welsh_uncert"].gen_synth(10)
    corr["welsh_uncert"].gen_corr()
    assert np.shape(corr["welsh_uncert"].l1s) == (2, 77)
    assert np.shape(corr["welsh_uncert"].l2s) == (2, 77)
    assert np.shape(corr["welsh_uncert"].l3s) == (2, 77)

    corr["fail"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["fail"].gen_synth(10)
    with pytest.raises(Exception):
        corr["fail"].gen_corr()
        
    
    # compare old implementation with new numba implementation
    corr["welsh_old"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["welsh_old"].gen_synth(10)
    corr["welsh_old"].gen_corr()
    assert np.shape(corr["welsh_old"].l1s) == (2, 77)
    assert np.shape(corr["welsh_old"].l2s) == (2, 77)
    assert np.shape(corr["welsh_old"].l3s) == (2, 77)

    corr["kroedel_old"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["kroedel_old"].gen_synth(10)
    corr["kroedel_old"].gen_corr()
    assert np.shape(corr["kroedel_old"].l1s) == (2, 77)
    assert np.shape(corr["kroedel_old"].l2s) == (2, 77)
    assert np.shape(corr["kroedel_old"].l3s) == (2, 77)

    assert np.allclose(corr['welsh_old'].values, corr['welsh'].values, rtol=1e-5)
    assert np.allclose(corr['kroedel_old'].values, corr['kroedel'].values, rtol=1e-5)

def test_plot_corr(corr):
    corr["welsh"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["welsh"].gen_synth(10)
    corr["welsh"].gen_corr()
    corr["welsh"].plot_corr(legend=True)

    corr["kroedel_uncert"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["kroedel_uncert"].gen_synth(10)
    corr["kroedel_uncert"].gen_corr()
    corr["kroedel_uncert"].plot_corr(legend=True)
    
def test_peak_find(corr):
    corr["welsh"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["welsh"].gen_synth(10)
    corr["welsh"].gen_corr()
    corr["welsh"].plot_corr(legend=True)
    assert np.any(np.isclose(corr['welsh'].peak_find()['x'], 2*np.pi, rtol=1e-3))
    assert np.any(np.isclose(corr['welsh'].peak_find(smooth=True)['x'], 2*np.pi, rtol=1e-2))
    with pytest.raises(Exception):
        assert np.any(np.isclose(corr['welsh'].peak_find(smooth=True)['x'], 2*np.pi, rtol=1e-3))


