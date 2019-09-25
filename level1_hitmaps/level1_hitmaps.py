import numpy as np
from astropy import wcs
from matplotlib import pyplot
import h5py
import binFuncs
from scipy.interpolate import interp1d
import os

from matplotlib.patches import Ellipse
from Mapping import Mapper
from MappingAzEl import MapperAzEl

# Class for storing source locations
class Source:
    def __init__(self,x,y, hmsMode=True):
        """
        If hmsMode = True (default) then x and y are 3 element lists containing (HH,MM,SS.S) and (DD,MM,SS.S) respectively.
        else just pass the degrees of each coordinates
        """
        
        if hmsMode:
            self.x = self.HMS2Degree(x)
            self.y = self.DMS2Degree(y)
        else:
            self.x = x
            self.y = y

    def __call__(self):
        return self.x, self.y
    
    def DMS2Degree(self,d):
        """
        Convert DD:MM:SS.S format to degrees
        """
        return d[0] + d[1]/60. + d[2]/60.**2
    
    def HMS2Degree(self,d):
        return self.DMS2Degree(d)*15

sources = {'TauA':Source([5,34,31.94],[22,0,52.2]),
           'CasA':Source([23,23,24.0],[58,48,54.0]),
           'CygA':Source([19,59,28.36],[40,44,2.10])}
            
from tqdm import tqdm 
import click
import ast
class PythonLiteralOption(click.Option):
    def type_cast_value(self,ctx,value):
        if isinstance(value,str):
            try:
                return ast.literal_eval(value)
            except:
                raise click.BadParameter(value)
        else:
            return value    
@click.command()
@click.argument('filename')#, help='Level 1 hdf5 file')
@click.option('--image_directory', default=None, help='Output image header directory')
@click.option('--band_average', default=True,type=bool, help='Average channels into single map')
@click.option('--feed_average', default=False,type=bool, help='Average all feeds into single map')
@click.option('--feeds', default=[0], cls=PythonLiteralOption, help='List of feeds to use (index from 0)')
@click.option('--make_hits', default=True,type=bool, help='Make hit maps')
@click.option('--make_sky', default=True,type=bool, help='Make sky maps')
@click.option('--cdelt', default=[1.,1.],cls=PythonLiteralOption, help='WCS cdelt parameter of form [x_pix, y_pix] in arcmin')
@click.option('--field_width', default=[3.,3.], cls=PythonLiteralOption, help='Field width list of form [ra_width, dec_width]')
@click.option('--ctype', default=['RA---TAN','DEC--TAN'], cls=PythonLiteralOption, help='Field WCS ctype list of form [RATYPE, DECTYPE]')
@click.option('--crval', default=None, cls=PythonLiteralOption, help='Field centre list of form [RA_cen, Dec_cen], (Default: None, take ra/dec from average of scans)')
@click.option('--source', default=None, help='Source name for field centre, if source unknown ignore (Default: None, take ra/dec centre from average ra/dec)')
@click.option('--plot_circle',default=False,type=bool, help='Overplot a circle of radius plot_circle_radius (Default: False)')
@click.option('--plot_circle_radius', default=1,type=float, help='Radius of over plotted circle')
@click.option('--az_el_mode',default=False,type=bool, help='Plot in az/el coordinates not Ra/Dec (Default: False)')
def call_level1_hitmaps(filename,
                        image_directory,
                        band_average,
                        feed_average,
                        feeds,
                        make_hits,
                        make_sky,
                        field_width,
                        cdelt,
                        ctype,
                        crval,
                        source,
                        plot_circle,
                        plot_circle_radius,
                        az_el_mode):

    level1_hitmaps(filename,
                   image_directory,
                   band_average,
                   feed_average,
                   feeds,
                   make_hits,
                   make_sky,
                   field_width,
                   cdelt,
                   ctype,
                   crval,
                   source,
                   plot_circle,
                   plot_circle_radius,
                   az_el_mode)
    

def level1_hitmaps(filename,
                   image_directory,
                   band_average=True,
                   feed_average=False,
                   feeds=[0],
                   make_hits=True,
                   make_sky=True,
                   field_width=[3.,3.],
                   cdelt=[1./60.,1./60.],
                   ctype=['RA---TAN','DEC--TAN'],
                   crval=None,
                   source='None',
                   plot_circle=False,
                   plot_circle_radius=1,
                   AzElMode=False):
    
    """Plot hit maps for feeds

    Arguments:

    filename: the name of the COMAP Level-1 file

    Keywords:

    feeds: Feeds (indexing starting from 0), can be list, tuple, range
    makeHitMap: Make the hit map
    makeAvgMap: Make the band average map and hit map
    cdelt: pixel size in degrees
    fieldWidth: image width in degrees
    """


    try:
        fd = h5py.File(filename,'r')
    except OSError:
        print('Unable to open file {}'.format(filename))
        return

    # cdelt given in arcmin
    xpixelWidth = int(field_width[0]/cdelt[0]*60)
    ypixelWidth = int(field_width[1]/cdelt[1]*60)

    if AzElMode:
        mapper = MapperAzEl(makeHitMap=make_hits,
                            makeAvgMap=make_sky,
                            crval=crval,
                            cdelt=cdelt,
                            crpix=[xpixelWidth//2, ypixelWidth//2],
                            ctype=ctype)
    else:
        mapper = Mapper(makeHitMap=make_hits,
                        makeAvgMap=make_sky,
                        crval=crval,
                        cdelt=cdelt,
                        crpix=[xpixelWidth//2, ypixelWidth//2],
                        ctype=ctype)
        

    if isinstance(image_directory, type(None)):
        image_directory = filename.split('/')[-1].split('.')[0]
        if not os.path.exists(image_directory):
            os.makedirs(image_directory)
            
    mapper.setLevel1(fd, source)
    if feed_average:
        
        maps = mapper(feeds, usetqdm=True)
        mapper.plotImages('{}/Hitmap_FeedAvg.png'.format(image_directory),
                          '{}/BandAverage_FeedAvg.png'.format(image_directory),
                          feeds,
                          plot_circle,
                          plot_circle_radius)
        mapper.SaveMaps('{}/BandAverage_FeedAvg.fits'.format(image_directory))
        return
                   
    for feed in tqdm(feeds):
        if not isinstance(mapper.map_bavg,type(None)):
            mapper.map_bavg *= 0.
            mapper.hits = None

        maps = mapper(feed)

        mapper.plotImages('{}/Hitmap_Feed{:02d}.png'.format(image_directory,feed),
                          '{}/BandAverage_Feed{:02d}.png'.format(image_directory,feed),
                          [feed],
                          plot_circle,
                          plot_circle_radius)
        mapper.SaveMaps('{}/BandAverage_Feed{:02d}.fits'.format(image_directory,feed))


if __name__ == "__main__":
    call_level1_hitmaps()
