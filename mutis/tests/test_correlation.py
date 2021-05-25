import pytest
import numpy as np
from numpy.testing import assert_allclose
from mutis.signal import Signal
from mutis.correlation import Correlation


@pytest.fixture
def corr():
    times1 = np.linspace(2, 6, 40)
    values1 = np.sin(times1)
    times2 = np.linspace(8, 12, 40)
    values2 = np.sin(times2)
    signal1 = Signal(times1, values1, "lc_gen_psd_nft")
    signal2 = Signal(times2, values2, "lc_gen_psd_nft")
    return {
        "fail": Correlation(signal1, signal2, "test"),
        "welsh_ab": Correlation(signal1, signal2, "welsh_ab"),
        "kroedel_ab": Correlation(signal1, signal2, "kroedel_ab"),
        "numpy": Correlation(signal1, signal2, "numpy"),
        "fwelsh": Correlation(signal1, signal2, "fwelsh"),
        "fkroedel": Correlation(signal1, signal2, "fkroedel"),
    }


def test_gen_synth(corr):
    corr["welsh_ab"].gen_synth(10)
    assert len(corr["welsh_ab"].signal1.synth) == 10
    assert len(corr["welsh_ab"].signal2.synth) == 10


def test_plot_signals(corr):
    corr["welsh_ab"].plot_signals()


def test_gen_times(corr):
    corr["welsh_ab"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    assert_allclose(corr["welsh_ab"].times[0], 2.160, rtol=1e-3)
    assert corr["welsh_ab"].dts[0] == 0.1
    assert corr["welsh_ab"].nb[0] == 3
    corr["welsh_ab"].gen_times(ftimes="rawab", dt0=0.1, ndtmax=3, nbinsmin=3)
    assert_allclose(corr["welsh_ab"].times[0], 2.040, rtol=1e-3)
    assert_allclose(corr["welsh_ab"].dts[0], 0.14, rtol=1e-3)
    assert corr["welsh_ab"].nb[0] == 3
    corr["welsh_ab"].gen_times(ftimes="uniform", tmin=0.1, tmax=3, nbinsmin=3)
    assert_allclose(corr["welsh_ab"].times[0], 2.198, rtol=1e-3)
    assert_allclose(corr["welsh_ab"].dts[0], 0.0145, rtol=1e-3)
    assert corr["welsh_ab"].nb[0] == 3
    corr["welsh_ab"].gen_times(ftimes="numpy")
    assert corr["welsh_ab"].times[0] == 2
    assert corr["welsh_ab"].dts[0] == 0.01
    assert corr["welsh_ab"].nb[0] == 3


def test_plot_times(corr):
    corr["welsh_ab"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["welsh_ab"].plot_times(rug=True)


def test_gen_corr(corr):
    corr["welsh_ab"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["welsh_ab"].gen_synth(10)
    corr["welsh_ab"].gen_corr()
    assert np.shape(corr["welsh_ab"].l1s) == (2, 77)
    assert np.shape(corr["welsh_ab"].l2s) == (2, 77)
    assert np.shape(corr["welsh_ab"].l3s) == (2, 77)

    corr["kroedel_ab"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["kroedel_ab"].gen_synth(10)
    corr["kroedel_ab"].gen_corr()
    assert np.shape(corr["kroedel_ab"].l1s) == (2, 77)
    assert np.shape(corr["kroedel_ab"].l2s) == (2, 77)
    assert np.shape(corr["kroedel_ab"].l3s) == (2, 77)


def test_plot_corr(corr):
    corr["welsh_ab"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["welsh_ab"].gen_synth(10)
    corr["welsh_ab"].gen_corr()
    corr["welsh_ab"].plot_corr()
