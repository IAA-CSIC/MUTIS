import unittest
import numpy as np
from mutis.flares.bayblocks import BayesianBlocks
from mutis import Signal

def test_bayblocks():

    # test the bayblock algorithm with a signal that corresponds to 3 gaussian pulses in a row

    t = np.linspace(0, 6, 50)
    y = np.exp(-(t-1)**2/0.1) + np.exp(-(t-3)**2/0.1) + np.exp(-(t-5)**2/0.1)
    dy = 0.1 * np.abs(y) 

    signal = Signal(t, y, dy)
    bayblocks = BayesianBlocks(signal, p=1e-1)

    bayblocks.get_flares()

    # test that the signal and the bayblocks can be plotted
    bayblocks.signal.plot()
    bayblocks.plot()

    assert len(bayblocks.get_flare_list()) == 3

    ## test that the flares can be plotted too
    
    for flare in bayblocks.get_flare_list():
        flare.plot()