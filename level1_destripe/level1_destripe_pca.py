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

import ast
class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


@click.command()
@click.argument('filename')#, help='Level 1 hdf5 file')
@click.option('--options', cls=PythonLiteralOption, default="{}")
def call_level1_destripe(filename, options):
    level1_destripe(filename, options)

def level1_destripe(filename,options):
    
    """Plot hit maps for feeds

    Arguments:

    filename: the name of the COMAP Level-1 file

    """
    # Get the inputs:
    parameters = ParserClass.Parser(filename)

    title = parameters['Inputs']['title'] 

    for k1,v1 in options.items():
        if len(options.keys()) == 0:
            break
        for k2, v2 in v1.items():
            parameters[k1][k2] = v2

    # Read in all the data
    if not isinstance(parameters['Inputs']['feeds'], list):
        parameters['Inputs']['feeds'] = [parameters['Inputs']['feeds']]
    if not isinstance(parameters['Inputs']['frequencies'], list):
        parameters['Inputs']['frequencies'] = [parameters['Inputs']['frequencies']]
    if not isinstance(parameters['Inputs']['bands'], list):
        parameters['Inputs']['bands'] = [parameters['Inputs']['bands']]


    # loop over band and frequency
    for band in np.array(parameters['Inputs']['bands']).astype(int):
        for frequency in np.array(parameters['Inputs']['frequencies']).astype(int):

            # Data parsing object
            data = DataLevel2PCA(parameters,band=band,frequency=frequency,keeptod=True)
            data.naive.average()


            offsetMap, offsets = Destriper(parameters, data)

            offsets.average()

            # Write offsets back out to the level2 files
            toffs = offsets()
            nFeeds = len(parameters['Inputs']['feeds'])
            filelist = np.loadtxt(parameters['Inputs']['filelist'],dtype=str,ndmin=1)

            from astropy.io import fits
            hdu = fits.PrimaryHDU(data.naive()-offsetMap(),header=data.naive.wcs.to_header())
            weights = fits.ImageHDU(data.naive.weights(), header=data.hits.wcs.to_header())
            hits = fits.ImageHDU(data.hits(returnsum=True), header=data.hits.wcs.to_header())
            naive   = fits.ImageHDU(data.naive(), header=data.hits.wcs.to_header())
            offsets = fits.ImageHDU(offsetMap(), header=data.hits.wcs.to_header())
            
            hdu1 = fits.HDUList([hdu, weights,hits,naive,offsets])
            feedstrs = [str(v) for v in parameters['Inputs']['feeds']]
            hdu1.writeto('fitsfiles/gfield-allfeeds/{}_feeds{}_offset{}_band{}_freq{}.fits'.format(title,'-'.join(feedstrs),
                parameters['Destriper']['offset'],
                band,frequency),overwrite=True)

if __name__ == "__main__":
    call_level1_destripe()

    #m = data.naive()
    #m[m == 0] = np.nan
    #pyplot.plot(data.todall)
    #pyplot.plot(np.repeat(offsets.offsets,offsets.offset))
    #pyplot.figure()
    # pyplot.subplot(221)
    # pyplot.imshow(data.naive())
    # pyplot.subplot(222)
    # pyplot.imshow(data.naive()-offsetMap(),vmin=-2e3,vmax=2e3)
    # pyplot.colorbar()
    # pyplot.subplot(223)
    # pyplot.imshow(offsetMap())

    # pyplot.figure()
    # pyplot.imshow(data.naive()-offsetMap(),vmin=0,vmax=1.5e3)
    # pyplot.show()
