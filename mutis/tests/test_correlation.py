import pytest
import numpy as np
from mutis.signal import Signal
from mutis.correlation import Correlation


@pytest.fixture
def corr():
    times = np.linspace(10, 20, 100)
    values = 0.5 * np.sin(1 * np.linspace(10, 20, 100)) + 0.5 * np.sin(6 * np.linspace(10, 20, 100)) + 1
    signal1 = Signal(times, values, "lc_gen_samp")
    signal2 = Signal(times, values, "lc_gen_psd_nft")
    return {
        "fail": Correlation(signal1, signal2, "test"),
        "welsh_ab": Correlation(signal1, signal2, "welsh_ab"),
        "kroedel_ab": Correlation(signal1, signal2, "kroedel_ab"),
        "numpy": Correlation(signal1, signal2, "numpy")
    }


def test_gen_synth(corr):
    corr["welsh_ab"].gen_synth(10)


def test_plot_signals(corr):
    corr["welsh_ab"].plot_signals()


def test_gen_times(corr):
    corr["welsh_ab"].gen_times()
    # TODO assert values

# def test_gen_corr(correlation):
#     correlation.gen_times()
#     correlation.gen_corr()
#
#
# def test_plot_corr(correlation):
#     correlation.gen_times()
#     correlation.gen_corr()
#     correlation.plot_corr()



