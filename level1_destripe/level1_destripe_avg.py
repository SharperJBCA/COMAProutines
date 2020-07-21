import numpy as np
from astropy import wcs
from matplotlib import pyplot
import h5py
import binFuncs
from scipy.interpolate import interp1d
import os
from astropy.io import fits

from matplotlib.patches import Ellipse
           
from tqdm import tqdm 
import click
import ast

from comancpipeline.Tools import ParserClass

from Types import *

from Destriper import Destriper, DestriperHPX

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

    upperFrequency = parameters['Inputs']['upperFrequency']
    lowerFrequency = parameters['Inputs']['lowerFrequency']
    title = parameters['Inputs']['title']

    # Read in all the data
    if not isinstance(parameters['Inputs']['feeds'], list):
        parameters['Inputs']['feeds'] = [parameters['Inputs']['feeds']]
    filelist = np.loadtxt(parameters['Inputs']['filelist'],dtype=str,ndmin=1)

    nside = int(parameters['Inputs']['nside'])
    data = DataLevel2AverageHPX_test(filelist,parameters,nside=nside,keeptod=False,subtract_sky=False)
    
    offsetMap, offsets = DestriperHPX(parameters, data)

    naive = data.naive()
    offmap= offsetMap()
    hits = data.hits.return_hpx_hits()
    des = naive-offmap
    des[des == 0] = hp.UNSEEN
    naive[naive == 0] = hp.UNSEEN
    offmap[offmap==0] = hp.UNSEEN
    hits[hits == 0] = hp.UNSEEN
    hp.write_map('{}_{}-{}.fits'.format(title,upperFrequency,lowerFrequency), [naive-offmap, naive, offmap,hits],overwrite=True,partial=True)

    feedstrs = [str(v) for v in parameters['Inputs']['feeds']]


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
