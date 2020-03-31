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

def CGM(data, offsets, offsetMap, niter=400):
    """
    Conj. Gradient Inversion
    """

    # -- We are performing inversion of Ax = b    
    # Solving for x, Ax = b
    Ax = Offsets(offsets.offset, offsets.Noffsets, offsets.Nsamples)
    b  = data.residual
    counts = offsets.offsets*0.

    b.average()
    Ax.average()

    # Estimate initial residual
    binFuncs.EstimateResidual(Ax.offsets, # Holds the weighted residuals
                              counts,
                              offsets.offsets, # holds the target offsets
                              #b.wei, # The weights calculated from the data
                              data.allweights,#residual.wei,
                              offsetMap.output, # Map to store the offsets in (initially all zero)
                              offsets.offsetpixels, # Maps offsets to TOD position
                              data.pixels) # Maps pixels to TOD position


    print('Diag counts:',np.min(counts))


    #Ax.offsets = Ax.offsets/counts #* offsets.offset
    # -- Calculate the initial residual and direction vectors
    #b.sigwei *= offsets.offset
    print('Diags b.sigwei, Ax.offsets:', np.sum(b.sigwei), np.sum(Ax.offsets))

    residual = b.sigwei - Ax.offsets
    direction= b.sigwei - Ax.offsets

    r2 = b.sigwei - Ax.offsets

    # -- Initial threshhold
    thresh0 = np.sum(residual**2)
    dnew    = np.sum(residual**2)
    alpha   = 0

    print('Diags thresh0:', thresh0)
    #offsets.offsets = data.residual.offsets 
    lastoffset = 0
    newVals = np.zeros(niter)
    alphas  = np.zeros(niter)
    betas   = np.zeros(niter)
    if np.isnan(np.sum(b.sigwei)):
        return

    for i in range(niter):
        # -- Calculate conjugate search vector Ad
        lastoffset = Ax.offsets*1.
        Ax.offsets *= 0
        counts *= 0

        offsetMap.clearmaps()
        offsetMap.binOffsets(direction,
                             data.residual.wei,
                             offsets.offsetpixels,
                             data.pixels)
        offsetMap.average()

        binFuncs.EstimateResidual(Ax.offsets,
                                  counts,
                                  direction,
                                  data.allweights,#residual.wei,
                                  offsetMap.output,
                                  offsets.offsetpixels,
                                  data.pixels)

                         
        

        # Calculate the search vector
        dTq = np.sum(direction*Ax.offsets)

        # 
        alpha = dnew/dTq
        alphas[i]=alpha
        # -- Update offsets

        olfast = offsets.offsets*1.
        offsets.offsets += alpha*direction
        #offsets.offsets[0] = offsets.offsets[1]

        # -- Calculate new residual
        if np.mod(i,5) == 0:
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
                                      #b.wei, # The weights calculated from the data
                                      data.allweights,#residual.wei,
                                      offsetMap.output, # Map to store the offsets in (initially all zero)
                                      offsets.offsetpixels, # Maps offsets to TOD position
                                      data.pixels) # Maps pixels to TOD position

            residual = b.sigwei - Ax.offsets
        else:
            residual = residual -  alpha*Ax.offsets 
        #print('Diag residual:' , np.sum(residual))

        dold = dnew*1.0
        dnew = np.sum(residual**2)
        newVals[i] = dnew
        # --
        beta = dnew/dold
        betas[i] = beta

        # -- Update direction
        direction = residual + beta*direction

        offsetMap.clearmaps()
        offsetMap.binOffsets(direction,
                             data.residual.wei,
                             offsets.offsetpixels,
                             data.pixels)
        offsetMap.average()
                   
        

        #print((-np.log10(dnew/thresh0))/8 )
        if dnew/thresh0 < 1e-8:
            break
    if False:
        pyplot.subplot(221)
        pyplot.plot(newVals)
        pyplot.yscale('log')
        pyplot.xscale('log')
        pyplot.grid()
        pyplot.subplot(222)
        pyplot.plot(alphas)
        pyplot.yscale('log')
        pyplot.xscale('log')
        pyplot.grid()
        pyplot.subplot(223)
        pyplot.plot(betas)
        pyplot.yscale('log')
        pyplot.xscale('log')
        pyplot.grid()
        pyplot.subplot(224)
        pyplot.plot(offsets())
        pyplot.grid()
        pyplot.show()
    print('Achieved {} in {} steps'.format(dnew/thresh0, i))

    offsetMap.clearmaps()
    offsetMap.binOffsets(offsets.offsets,
                         data.residual.wei,
                         offsets.offsetpixels,
                         data.pixels)
    offsetMap.average()
