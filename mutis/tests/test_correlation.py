import pytest
import numpy as np
from mutis.signal import Signal
from mutis.correlation import Correlation


@pytest.fixture
def correlation():
    times = np.linspace(10, 20, 100)
    values = 0.5 * np.sin(1 * np.linspace(10, 20, 100)) + 0.5 * np.sin(6 * np.linspace(10, 20, 100)) + 1
    signal1 = Signal(times, values, "lc_gen_samp")
    signal2 = Signal(times, values, "lc_gen_psd_nft")
    return Correlation(signal1, signal2, "welsh_ab")


def test_gen_synth(correlation):
    correlation.gen_synth(10)


def test_plot_signals(correlation):
    correlation.plot_signals()

