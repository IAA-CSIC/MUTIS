import pytest
import numpy as np
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


def test_plot_signals(corr):
    corr["welsh_ab"].plot_signals()


def test_gen_times(corr):
    corr["welsh_ab"].gen_times()
    corr["welsh_ab"].gen_times(ftimes="rawab")
    corr["welsh_ab"].gen_times(ftimes="uniform")
    corr["welsh_ab"].gen_times(ftimes="numpy")
    # TODO assert values ?


def test_plot_times(corr):
    corr["welsh_ab"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["welsh_ab"].plot_times(rug=True)


def test_gen_corr(corr):
    corr["welsh_ab"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["welsh_ab"].gen_synth(10)
    corr["welsh_ab"].gen_corr()

    corr["kroedel_ab"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["kroedel_ab"].gen_synth(10)
    corr["kroedel_ab"].gen_corr()


def test_plot_corr(corr):
    corr["welsh_ab"].gen_times(dtmin=0.1, dtmax=3, nbinsmin=3)
    corr["welsh_ab"].gen_synth(10)
    corr["welsh_ab"].gen_corr()
    corr["welsh_ab"].plot_corr()
