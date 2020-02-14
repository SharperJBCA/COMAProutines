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
from MappingSun import MapperSun

from tqdm import tqdm 
import click
import ast
class PythonLiteralOption(click.Option):
    def type_cast_value(self,ctx,value):
        print(value)
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
@click.option('--feeds', default=[1], cls=PythonLiteralOption, help='List of feeds to use (index from 0)')
@click.option('--make_hits', default=True,type=bool, help='Make hit maps')
@click.option('--make_sky', default=True,type=bool, help='Make sky maps')
@click.option('--cdelt', default=[1.,1.],cls=PythonLiteralOption, help='WCS cdelt parameter of form [x_pix, y_pix] in arcmin')
@click.option('--field_width', default=None, cls=PythonLiteralOption, help='Field width list of form [ra_width, dec_width]')
@click.option('--ctype', default=['RA---TAN','DEC--TAN'], cls=PythonLiteralOption, help='Field WCS ctype list of form [RATYPE, DECTYPE]')
@click.option('--crval', default=None, cls=PythonLiteralOption, help='Field centre list of form [RA_cen, Dec_cen], (Default: None, take ra/dec from average of scans)')
@click.option('--source', default=None, help='Source name for field centre, if source unknown ignore (Default: None, take ra/dec centre from average ra/dec)')
@click.option('--plot_circle',default=False,type=bool, help='Overplot a circle of radius plot_circle_radius (Default: False)')
@click.option('--plot_circle_radius', default=1,type=float, help='Radius of over plotted circle')
@click.option('--az_el_mode',default=False,type=bool, help='Plot in az/el coordinates(Default: False)')
@click.option('--sun_mode',default=False,type=bool, help='Plot in Sun centric coordinates  (Default: False)')
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
                        az_el_mode,
                        sun_mode):

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
                   az_el_mode,
                   sun_mode)
    

def level1_hitmaps(filename,
                   image_directory,
                   band_average=True,
                   feed_average=False,
                   feeds=[1],
                   make_hits=True,
                   make_sky=True,
                   field_width=None,
                   cdelt=[1./60.,1./60.],
                   ctype=['RA---TAN','DEC--TAN'],
                   crval=None,
                   source='None',
                   plot_circle=False,
                   plot_circle_radius=1,
                   AzElMode=False,
                   SunMode=False):
    
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
    if not isinstance(field_width, type(None)):
        xpixelWidth = int(field_width[0]/cdelt[0]*60)
        ypixelWidth = int(field_width[1]/cdelt[1]*60)
        image_width = [xpixelWidth, ypixelWidth]
    else:
        image_width = None

    if isinstance(image_directory, type(None)):
        image_directory = filename.split('/')[-1].split('.')[0]
        if not os.path.exists(image_directory):
            os.makedirs(image_directory)


    if AzElMode:
        mapper = MapperAzEl(makeHitMap=make_hits,
                            makeAvgMap=make_sky,
                            crval=crval,
                            cdelt=cdelt,
                            npix=image_width,
                            image_directory=image_directory,
                            ctype=ctype)
    elif SunMode:
        mapper = MapperSun(makeHitMap=make_hits,
                           makeAvgMap=make_sky,
                           crval=crval,
                           cdelt=cdelt,
                           npix=image_width,
                           image_directory=image_directory,
                           ctype=ctype)
        
    else:
        mapper = Mapper(makeHitMap=make_hits,
                        makeAvgMap=make_sky,
                        image_directory=image_directory,
                        crval=crval,
                        cdelt=cdelt,
                        npix=image_width,
                        ctype=ctype)
        
            
    mapper.setLevel1(fd, source)
    if 'all' in feeds:
        feeds = [feed for feed in fd['spectrometer/feeds'][:] if feed != 20]
    if feed_average:
        
        maps = mapper(feeds, usetqdm=True)
        fstr = '-'.join(['{:02d}'.format(feed) for feed in feeds if feed in mapper.feed_ids])
        outdir = '{}/Feeds-{}'.format(image_directory,fstr)

        mapper.plotImages(feeds,
                          '{}/Hitmap_FeedAvg.png'.format(outdir),
                          '{}/BandAverage_FeedAvg.png'.format(outdir),
                          plot_circle,
                          plot_circle_radius)
       # mapper.SaveMaps('{}/BandAverage_FeedAvg.fits'.format(image_directory))
        
                   
    for feed in tqdm(feeds):
        if not isinstance(mapper.map_bavg,type(None)):
            mapper.map_bavg *= 0.
            mapper.hits = None

        maps = mapper(feed)

        fstr = '-'.join(['{:02d}'.format(feed)])
        outdir = '{}/Feeds-{}'.format(image_directory,fstr)

        mapper.plotImages([feed],
                          '{}/Hitmap_Feed{:02d}.png'.format(outdir,feed),
                          '{}/BandAverage_Feed{:02d}.png'.format(outdir,feed),
                          plot_circle,
                          plot_circle_radius)
        #mapper.SaveMaps('{}/BandAverage_Feed{:02d}.fits'.format(image_directory,feed))


if __name__ == "__main__":
    call_level1_hitmaps()
