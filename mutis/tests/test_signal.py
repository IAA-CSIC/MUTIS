import pytest
import numpy as np
from mutis.signal import Signal


@pytest.fixture
def profile():
    p = {"times": np.linspace(10, 20, 100)}
    p["values"] = 0.5 * np.sin(1 * p["times"]) + 0.5 * np.sin(6*p["times"]) + 1
    return p


def test_generation(profile):
    signal = Signal(profile["times"], profile["values"], "lc_gen_samp")


