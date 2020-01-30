import numpy as np
from astropy import wcs
from matplotlib import pyplot
import h5py
import binFuncs
from scipy.interpolate import interp1d
import os

from matplotlib.patches import Ellipse
           
from tqdm import tqdm 
import click
import ast

from comancpipeline.Tools import ParserClass

from Types import *

from Destriper import Destriper


@click.command()
@click.argument('filename')#, help='Level 1 hdf5 file')
def call_level1_destripe(filename):
    level1_destripe(filename)

def level1_destripe(filename):
    
    """Plot hit maps for feeds

    Arguments:

    filename: the name of the COMAP Level-1 file

    """
    # Get the inputs:
    parameters = ParserClass.Parser(filename)

    # Read in all the data
    data = Data(parameters)
    data.naive.average()

    #print(np.sum(data.todall))
    #pyplot.plot(data.todall,',')
    #pyplot.show()

    offsetMap, offsets = Destriper(parameters, data)

    #m = data.naive()
    #m[m == 0] = np.nan
    #pyplot.plot(data.todall)
    #pyplot.plot(np.repeat(offsets.offsets,offsets.offset))
    #pyplot.figure()
    pyplot.subplot(221)
    pyplot.imshow(data.naive())
    pyplot.subplot(222)
    pyplot.imshow(data.naive()-offsetMap(),vmin=-2e3,vmax=2e3)
    pyplot.colorbar()
    pyplot.subplot(223)
    pyplot.imshow(offsetMap())

    pyplot.figure()
    pyplot.imshow(data.naive()-offsetMap(),vmin=0,vmax=1.5e3)
    pyplot.show()

    from astropy.io import fits
    hdu = fits.PrimaryHDU(data.naive()-offsetMap(),header=data.naive.wcs.to_header())
    hits = fits.ImageHDU(data.hits(returnsum=True), header=data.hits.wcs.to_header())
    naive   = fits.ImageHDU(data.naive(), header=data.hits.wcs.to_header())
    offsets = fits.ImageHDU(offsetMap(), header=data.hits.wcs.to_header())

    hdu1 = fits.HDUList([hdu, hits,naive,offsets])
    hdu1.writeto('fg6_all.fits',overwrite=True)

if __name__ == "__main__":
    call_level1_destripe()
