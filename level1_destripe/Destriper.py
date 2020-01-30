import numpy as np
from matplotlib import pyplot

from Types import Offsets, Map
import binFuncs

def Destriper(parameters, data):
    """
    Destriping routines
    """

    niter = int(parameters['Destriper']['niter'])

    # NB : Need to change offsets to ensure that each
    # is temporally continuous in the future, for now ignore this.
    offsetLen = parameters['Destriper']['offset']
    Noffsets  = data.Nsamples//offsetLen

    # Offsets for storing the outputs
    offsets   = Offsets(offsetLen, Noffsets,  data.Nsamples)

    # For storing the offsets on the sky
    offsetMap = Map(data.naive.nxpix,
                    data.naive.nypix,
                    data.naive.wcs)


    CGM(data, offsets, offsetMap, niter=niter)

    return offsetMap, offsets

def CGM(data, offsets, offsetMap, niter=200):
    """
    Conj. Gradient Inversion
    """

    # -- We are performing inversion of Ax = b    
    # Solving for x, Ax = b
    Ax = Offsets(offsets.offset, offsets.Noffsets, offsets.Nsamples)
    b  = data.residual
    counts = offsets.offsets*0.


    binFuncs.EstimateResidual(Ax.offsets, # Holds the weighted residuals
                              counts,
                              offsets.offsets, # holds the target offsets
                              b.wei, # The weights calculated from the data
                              offsetMap.output, # Map to store the offsets in (initially all zero)
                              offsets.offsetpixels, # Maps offsets to TOD position
                              data.pixels) # Maps pixels to TOD position
    #Ax.offsets /= counts
    print('Diag counts:',np.min(counts))
    #r = (offsets.offsets[offsets.offsetpixels]-offsetMap.output[data.pixels])#*data.weights
    #binFuncs.binValues(Ax.offsets, 
    #                   offsets.offsetpixels, 
    #                   weights=r)
    

    #Ax.offsets = Ax.offsets/counts #* offsets.offset
    # -- Calculate the initial residual and direction vectors
    #b.sigwei *= offsets.offset
    print('Diags b.sigwei, Ax.offsets:', np.sum(b.sigwei), np.sum(Ax.offsets))
    residual = b.sigwei - Ax.offsets
    r2 = b.sigwei - Ax.offsets
    direction= b.sigwei - Ax.offsets
    # -- Initial threshhold
    thresh0 = np.sum(residual**2)
    dnew = np.sum(residual**2)
    alpha = 0

    print('Diags thresh0:', thresh0)

    #offsets.offsets = data.residual.offsets 
    lastoffset = 0
    for i in range(niter):
        # -- Calculate conjugate search vector Ad
        lastoffset = Ax.offsets*1.
        Ax.offsets *= 0
        counts *= 0
        binFuncs.EstimateResidual(Ax.offsets,
                                  counts,
                                  direction,
                                  data.residual.wei,
                                  offsetMap.output,
                                  offsets.offsetpixels,
                                  data.pixels)
        # Ax.offsets /= counts
        # Axcopy = Ax.offsets*1.
        # Ax.offsets *= 0

        # r = (direction[offsets.offsetpixels]-offsetMap.output[data.pixels])#*weights
        # binFuncs.binValues(Ax.offsets, 
        #                   offsets.offsetpixels, 
        #                   weights=r)

        # pyplot.subplot(211)
        # pyplot.plot(Axcopy)
        # pyplot.subplot(212)
        # pyplot.plot(Ax.offsets)
        # pyplot.show()

        print('Diags (Ax.offsets (1)):',np.sum(Ax.offsets))

        
        #Ax.offsets = Ax.offsets/counts # * offsets.offset
        #pyplot.subplot(211)
        #pyplot.plot(Ax.offsets)
        #pyplot.plot(lastoffset)
        #pyplot.subplot(212)
        #pyplot.plot(counts)
        #pyplot.show()
        # Calculate the search vector
        dTq = np.sum(direction*Ax.offsets)

        # 
        alpha = dnew/dTq

        # -- Update offsets

        olfast = offsets.offsets*1.
        offsets.offsets += alpha*direction

        # -- Calculate new residual
        if np.mod(i,50) == 0:
            offsetMap.clearmaps()
            offsetMap.binOffsets(offsets.offsets,
                                 data.residual.wei,
                                 offsets.offsetpixels,
                                 data.pixels)
            offsetMap.average()
            Ax.offsets *= 0
            counts = offsets.offsets*0.

            binFuncs.EstimateResidual(Ax.offsets, # Holds the weighted residuals
                                      counts,
                                      offsets.offsets, # holds the target offsets
                                      b.wei, # The weights calculated from the data
                                      offsetMap.output, # Map to store the offsets in (initially all zero)
                                      offsets.offsetpixels, # Maps offsets to TOD position
                                      data.pixels) # Maps pixels to TOD position
            #Ax.offsets /= counts


            # r = (offsets.offsets[offsets.offsetpixels]-offsetMap.output[data.pixels])#*weights
            # binFuncs.binValues(Ax.offsets, 
            #                    offsets.offsetpixels, 
            #                    weights=r)
            residual = b.sigwei - Ax.offsets
        else:
            residual = residual -  alpha*Ax.offsets 

        print('Diag residual:' , np.sum(residual))

        dold = dnew*1.0
        dnew = np.sum(residual**2)

        # --
        beta = dnew/dold

        # -- Update direction
        direction = residual + beta*direction

        offsetMap.clearmaps()
        offsetMap.binOffsets(direction,
                             data.residual.wei,
                             offsets.offsetpixels,
                             data.pixels)
        offsetMap.average()
                   
        print((-np.log10(dnew/thresh0))/6 )
        if dnew/thresh0 < 1e-6:
            break

    offsetMap.clearmaps()
    offsetMap.binOffsets(offsets.offsets,
                         data.residual.wei,
                         offsets.offsetpixels,
                         data.pixels)
    offsetMap.average()
