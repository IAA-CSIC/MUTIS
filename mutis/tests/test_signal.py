import numpy as np
import pytest
from numpy.testing import assert_allclose

from mutis.lib.signal import lc_gen_psd_c
from mutis.signal import Signal


@pytest.fixture
def signal():
    times = np.linspace(10, 20, 100)
    values = (
        0.5 * np.sin(1 * np.linspace(10, 20, 100)) + 0.5 * np.sin(6 * np.linspace(10, 20, 100)) + 1
    )

    return {
        "fail": Signal(times, values, fgen="test"),
        "samp": Signal(times, values, fgen="lc_gen_samp"),
        "psd_c": Signal(times, values, fgen="lc_gen_psd_c"),
        "psd_nft": Signal(times, values, fgen="lc_gen_psd_nft"),
        "psd_fft": Signal(times, values, fgen="lc_gen_psd_fft"),
        "psd_lombscargle": Signal(times, values, fgen="lc_gen_psd_lombscargle"),
        "ou": Signal(times, values, fgen="lc_gen_ou"),
    }


@pytest.fixture
def ou_params():
    return {"theta": 100, "mu": 1, "sigma": 10}


def test_plot(signal):
    signal["samp"].plot()


def test_gen_synth(signal, ou_params):
    with pytest.raises(Exception):
        signal["fail"].gen_synth(10)
    signal["samp"].gen_synth(10)
    assert len(signal["samp"].synth) == 10
    signal["psd_c"].gen_synth(10)
    assert len(signal["psd_c"].synth) == 10
    signal["psd_nft"].gen_synth(10)
    assert len(signal["psd_nft"].synth) == 10
    signal["psd_fft"].gen_synth(10)
    assert len(signal["psd_fft"].synth) == 10
    signal["psd_lombscargle"].gen_synth(10)
    assert len(signal["psd_lombscargle"].synth) == 10

    with pytest.raises(Exception):
        signal["ou"].gen_synth(10)
    signal["ou"].OU_theta = ou_params["theta"]
    signal["ou"].OU_mu = ou_params["mu"]
    signal["ou"].OU_sigma = ou_params["sigma"]
    signal["ou"].gen_synth(10)
    assert len(signal["ou"].synth) == 10


def test_ou_fit(signal):
    fits = signal["ou"].OU_fit()
    assert_allclose(fits["curve_fit"][0][0], 0.24649419240104922, rtol=1e-3)


def test_check_gen(signal, ou_params):
    signal["ou"].check_gen(fgen='lc_gen_ou', fgen_params=ou_params)
    signal["ou"].check_gen(fgen='lc_gen_ou', fgen_params=ou_params)
    signal["psd_c"].check_gen(fgen='lc_gen_psd_c')
    signal["psd_nft"].check_gen(fgen='lc_gen_psd_nft')
    signal["psd_fft"].check_gen(fgen='lc_gen_psd_fft')
    signal["psd_lombscargle"].check_gen(fgen='lc_gen_psd_lombscargle')


def test_lc_gen_psd_c(signal):
    values = lc_gen_psd_c(signal["samp"].times, signal["samp"].values, signal["samp"].times)
    assert len(values) == 100
