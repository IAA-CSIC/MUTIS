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

from mutis import.Signal
from mutis.flares.bayblocks import BayesianBlocks

log = logging.getLogger(__name__)



def detect_flares(lightcurve):
    if isinstance(lightcurve, BayesianBlocks):
        pass
    else:
        return None
    
    