""" Flare analysis """

import logging

import numpy as np
import scipy as sp
import pandas as pd

import matplotlib as mplt
import matplotlib.pyplot as plt

import sys
import os
import glob

import re
from datetime import datetime

from astropy.stats import bayesian_blocks

import warnings 

from mutis import Signal
#from mutis.flares.bayblocks import BayesianBlocks
import mutis.flares.bayblocks as bayblocks

log = logging.getLogger(__name__)


    
class Flare:
    """Container class for a Flare object
    
    Attributes
    ----------
    tstart: float
            time at which the flare starts
    tstop: float
            time at which the flare ends
    """
    
    def __init__(self, tstart, tstop):
        self.tstart = tstart
        self.tstop = tstop
    
    def __repr__(self):
        return f"Flare({self.tstart},{self.tstop})"
    
    def __str__(self):
        return self.__repr__()
        
    
    def plot(self, ax=None, **kwargs):
        """Plots the flare as a colored area """"
        ax = plt.gca() if ax is None else ax
        
        ax.axvspan(self.tstart, self.tstop, facecolor='r', edgecolor=None, alpha=0.2, **kwargs)
        
        pass
    