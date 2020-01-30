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
            else:#
                image[pixels[i]] += weights[i]


def binValues2Map(double[:] image, long[:] pixels, double[:] weights, long[:] offsetpixels):

    cdef int i
    cdef int nsamples = pixels.size
    cdef int maxbin   = image.size
    cdef int noffsets = weights.size
    for i in range(nsamples):

        if (pixels[i] >= 0) & (pixels[i] < maxbin) & (offsetpixels[i] > 0) & (offsetpixels[i] < noffsets):
            image[pixels[i]] += weights[offsetpixels[i]]




def EstimateResidual(double[:] residual, 
                     double[:] counts,
                     double[:] offsetval,
                     double[:] offsetwei,
                     double[:] skyval,
                     long[:] offseti, 
                     long[:] pixel):

    cdef int i
    cdef int nsamples = pixel.size
    cdef int maxbin1  = skyval.size
    cdef int noffsets  = residual.size
    
    for i in range(nsamples):

        if ((pixel[i] >= 0) & (pixel[i] < maxbin1)) & ((offseti[i] >= 0) & (offseti[i] < noffsets)) & (offsetwei[offseti[i]] != 0):
            #print(residual[offseti[i]], offsetwei[offseti[i]], counts[offseti[i]])
            residual[offseti[i]] += (offsetval[offseti[i]]-skyval[pixel[i]])*offsetwei[offseti[i]]
            counts[offseti[i]] += 1

    for i in range(noffsets):
        if counts[i] != 0:
            residual[i] /=  counts[i]

