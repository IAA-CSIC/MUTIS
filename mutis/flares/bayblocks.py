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

import mutis.flares.flare as flare

log = logging.getLogger(__name__)

class BayesianBlocks:
    """ Return a Bayesian Block representation of the signal """

    def __init__(self, signal, p=0.1):
        edges = bayesian_blocks(signal.times, signal.values, signal.dvalues, fitness='measures', p0=p)
        
        values = list()
        for i in range( len(edges)-1 ):
            msk = (edges[i] < signal.times) & (signal.times < edges[i+1])
            if np.sum(msk) > 0:
                value = np.average( signal.values[msk], weights=1/signal.dvalues[msk] )
                values.append( value )
            else:
                values.append( 0 )
            
        self.edges = np.array(edges)
        self.values = np.array(values)
        self.signal = signal
        
        self.inflare = None
    
    def plot(self, style='color.strict', ax=None, **kwargs):
        """ Plot the bayesian block representation
        
        E.g:
        >>> signal = mutis.Signal(data['jyear'], data['Flux'], data['Flux_err'])
        >>> BayesianBlocks(signal).plot(color='k')

        """
        
        ax = plt.gca() if ax is None else ax
        
        #self.signal.plot()
        
        x = self.edges
        y = self.values[np.r_[0:len(self.values),-1]]

            
        if self.inflare is None or style == 'none':
            ax.step(x, y, 'k', where='post', **kwargs)
        else:
            if style == 'area':
                ax.step(x, y, 'k', where='post', **kwargs)
                for i in range(len(self.inflare)):
                    if self.inflare[i]:
                        ax.axvspan(x[i], x[i+1], facecolor='r', edgecolor=None,alpha=0.2)
            elif style == 'color.simple':
                for i in range(len(self.edges)-2):
                    ax.step(self.edges[[i,i+1]], self.values[[i,i+1]], 'r' if self.inflare[i] else 'k', where='post')
                ax.step(self.edges[[-2,-1]], self.values[[-1,-1]], 'r' if self.inflare[-1] else 'k', where='post')
            elif style == 'color.loose':
                for i in range(len(self.edges)-2):
                    ax.step(self.edges[[i,i+1]], self.values[[i,i]], 'r' if self.inflare[i] else 'k', where='post')
                    ax.step(self.edges[[i+1,i+1]], self.values[[i,i+1]], 'r' if self.inflare[i] or self.inflare[i+1] else 'k', where='post')
                ax.step(self.edges[[-2,-1]], self.values[[-1,-1]], 'r' if self.inflare[-1] else 'k', where='post')
            elif style == 'color.strict':
                for i in range(len(self.edges)-2):
                    ax.step(self.edges[[i,i+1]], self.values[[i,i]], 'r' if self.inflare[i] else 'k', where='post')
                    ax.step(self.edges[[i+1,i+1]], self.values[[i,i+1]], 'r' if self.inflare[i] and self.inflare[i+1] else 'k', where='post')
                ax.step(self.edges[[-2,-1]], self.values[[-1,-1]], 'r' if self.inflare[-1] else 'k', where='post')
            else:
                raise Exception("Unknown style. Available styles are 'none', 'area', 'color.simple', 'color.strict', 'color.loose'.")

                
        
    def get_flares(self, thresh=1):
        """ Get a list of flares following the algorithm proposed in 
[Meyer, Scargle, Blandford (2019)]
(https://iopscience.iop.org/article/10.3847/1538-4357/ab1651/pdf):
        
```There is no generally accepted consensus on the best way to
determine which data points belong to a flaring state and which
characterize the quiescent level. Nalewajko (2013)suggested
the simple definition that a flare is a continuous time interval
associated with a flux peak in which the flux is larger than half
the peak flux value. This definition is intuitive, however, and it
is unclear how to treat overlapping flares and identify flux
peaks in an objective way. Here we use a simple two-step
procedure tailored to the block representation: (1)identify a
block that is higher than both the previous and subsequent
blocks as a peak, and (2)proceed downward from the peak in
both directions as long as the blocks are successively lower.```
        """
        di_max = 5
        
        # get list of local maxima
        # bad beheaviour at 0 and -1:
        #imaxL = sp.signal.argrelextrema(self.values, np.greater)[0]
        imaxL = list()
        for i in range(0,len(self.values)):
            if i == 0 and thresh*self.values[0] > self.values[1]:
                imaxL.append(i)
            elif i == len(self.values)-1 and self.values[i-1] < thresh*self.values[i]:
                imaxL.append(i)
            elif self.values[i-1] < thresh*self.values[i] > self.values[i+1]:
                imaxL.append(i)
        imaxL = np.array(imaxL)
        
        
        inflareL = np.full(self.values.shape, False)
        
        # get list of local maxima over a threshold (flare peaks)
        for imax in imaxL:
            inflare = True
            
            if imax == 0:
                pass
            elif not (self.values[imax-1] < thresh*self.values[imax]):
                inflare = False
            
            
            if imax == len(self.values)-1:
                pass
            elif  not (thresh*self.values[imax] > self.values[imax+1]):
                inflare = False
                
            inflareL[imax] = inflare
           
        # extend the flare to adyacent blocks
        for imax in np.argwhere(inflareL):
            
            if imax == 0:
                pass
            else:
                di = 1
                stop = False
                while not stop and di < di_max:
                    if imax-di-1 < 0:
                        break
                    
                    if self.values[imax-di-1] < thresh*self.values[imax-di]:
                        inflareL[imax-di] = True
                    else:
                        stop = True
                    
                    
                    di = di + 1
                   
            if imax == len(self.values)-1:
                pass
            else:
                di = 1
                stop = False
                while not stop and di < di_max:
                    if imax+di+1 > len(self.values)-1:
                        break
                    
                    if thresh*self.values[imax+di] > self.values[imax+di+1]:
                        inflareL[imax+di] = True
                    else:
                        stop = True
                    
                    
                    di = di + 1
         
        
        self.inflare = np.array(inflareL)
        

    def get_flare_list(self):
        """ Join all flares into a list of mutis.flares.flare """

        groups = np.split(np.arange(len(self.inflare)), 
                          np.where((np.abs(np.diff(np.asarray(self.inflare, dtype=int))) == 1))[0]+1)
        
        # (groups contains the groups of indices of self.inflare/self.values 
        # that corresond to flares or no flares)
        
        #print(self.inflare)
        #print([list(grp) for grp in groups])
        
        flareL = list()
        for i, grp in enumerate(groups):
            if np.all(self.inflare[grp]):
                #print(grp)
                tstart = self.edges[grp[0]]
                tstop = self.edges[grp[-1]+1] # flare ends when no flare begins
                
                flareL.append(flare.Flare(tstart,tstop))
        print(flareL)
        return flareL