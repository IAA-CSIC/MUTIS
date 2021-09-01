# Licensed under a 3-clause BSD style license - see LICENSE
"""Utils specific to the field of astrophysics"""

import logging

import numpy as np

__all__ = ["Astro"]

log = logging.getLogger(__name__)

def pol_angle_reshape(s):
    """
        Reshape a signal as a polarization angle: shifting by 180 degrees in a way it varies as smoothly as possible.
    """
    
    s = np.array(s)
    s = np.mod(s,180) # s % 180
    print(np.amax(s))
    sn = np.empty(s.shape)
    for i in range(1,len(s)):
        #if t[i]-t[i-1] < 35:
        d = 181
        n = 0
        for m in range(0,100):
            m2 = (-1)**(m+1)*np.floor((m+1)/2)
            if np.abs(s[i]+180*m2-sn[i-1]) < d:
                d = np.abs(s[i]+180*m2-sn[i-1])
                n = m2
        sn[i] = s[i]+180*n
    return sn 