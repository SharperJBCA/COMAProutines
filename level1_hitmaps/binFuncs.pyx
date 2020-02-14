import numpy as np
cimport numpy as np
from cpython cimport array
import array


def binValues(double[:] image, long[:] pixels, double[:] weights=None, long[:] mask=None):
    """
    A simple binning routine for map-making. Sum is done in place.
    
    Arguments
    image  - 1D array of nypix * nxpix dimensions
    pixels - Indices for 1D image array
    
    Kwargs
    weights - 1D array the same length of pixels.
    mask    - Bool array, skip certain TOD values, 0 = skip, 1 = include
    """

    cdef int i,j,k  
    cdef int nsamples = pixels.size
    cdef int maxbin   = image.size
    for i in range(nsamples):
        if not isinstance(mask, type(None)):
            if mask[i] == 0:
                continue

        if (pixels[i] >= 0) & (pixels[i] < maxbin):
            if isinstance(weights, type(None)):
                image[pixels[i]] += 1.0
            else:
                image[pixels[i]] += weights[i]

