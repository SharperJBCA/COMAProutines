import numpy as np
from astropy import wcs
from matplotlib import pyplot
import h5py
import binFuncs
from scipy.interpolate import interp1d
import os

from matplotlib.patches import Ellipse

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
    
class Mapper:

    def __init__(self, 
                 makeHitMap=True,
                 makeAvgMap=False,
                 crval=None, 
                 cdelt=[1./60.,1./60.], 
                 crpix=[128,128],
                 ctype=['RA---TAN','DEC--TAN']):
        self.crval = crval
        self.cdelt = cdelt
        self.crpix = crpix
        self.ctype = ctype
        self.makeHitMap = makeHitMap
        self.makeAvgMap = makeAvgMap

        self.nxpix = int(crpix[0]*2)
        self.nypix = int(crpix[1]*2)

        self.ra = None
        self.dec= None
        self.tod_bavg = None
        self.hits = None
        self.map_bavg = None

    def __call__(self, items, usetqdm=False):
        """
        """
        if not isinstance(items, (range,tuple,list)):
            items = [items]
 
        feedlist = self.datafile['spectrometer/feeds'][...].astype(int)
        getfeeds = lambda f: np.argmin((f-feedlist)**2)
        
        self.feeds = map(getfeeds,items)
        self.usetqdm=usetqdm

        if isinstance(self.ra, type(None)):
            print('No level 1 data loaded')
            return
        else:
            if self.makeAvgMap:
                self.map_bavg, self.hits = self.avgMap(self.feeds, self.ra, self.dec, self.tod_bavg)
                return self.map_bavg, self.hits
            elif self.makeHitMap:
                self.hits = self.hitMap(self.feeds, self.ra, self.dec)
                return self.hits
            else:
                return
        
                
    def getFlatPixels(self, x, y):
        """
        """
        if isinstance(self.wcs, type(None)):
            raise TypeError( 'No WCS object declared')
            return
        else:
            pixels = self.wcs.wcs_world2pix(x+self.wcs.wcs.cdelt[0]/2.,
                                            y+self.wcs.wcs.cdelt[1]/2.,0)
            pflat = (pixels[0].astype(int) + self.nxpix*pixels[1].astype(int)).astype(int)
            

            # Catch any wrap around pixels
            pflat[(pixels[0] < 0) | (pixels[0] > self.nxpix)] = -1
            pflat[(pixels[1] < 0) | (pixels[1] > self.nypix)] = -1

            return pflat

    def setWCS(self, crval, cdelt, crpix, ctype):
        """
        Declare world coordinate system for plots
        """
        self.wcs = wcs.WCS(naxis=2)
        self.wcs.wcs.crval = crval
        self.wcs.wcs.cdelt = cdelt
        self.wcs.wcs.crpix = crpix
        self.wcs.wcs.ctype = ctype

    def hitMap(self,items, x, y):
        """
        """
        self.hits = np.zeros((self.nxpix*self.nypix))
        for item in items:
            pixels = self.getFlatPixels(self.ra[item,:], self.dec[item,:])
            binFuncs.binValues(self.hits, pixels, mask=self.mask)
            
        return np.reshape(self.hits, (self.nypix, self.nxpix)) * self.tsamp

    def avgMap(self, items, x, y, d):
        """
        """

        if isinstance(self.hits, type(None)):
            self.hits = np.zeros((self.nxpix*self.nypix))
            doHits = True
        else:
            doHits = False

        dataShape = d.shape
        nSidebands = dataShape[1]
        self.map_bavg = np.zeros((nSidebands, self.nxpix*self.nypix))
        if self.usetqdm:
            looper = tqdm(items)
        else:
            looper = items
        for item in looper:
            pixels = self.getFlatPixels(self.ra[item,:], self.dec[item,:])
            if doHits:
                binFuncs.binValues(self.hits, pixels, mask=self.mask)
            for sideband in range(nSidebands):
                tod = d[item,sideband,:]

                rms = np.nanstd(tod[1:tod.size//2*2:2] - tod[0:tod.size//2*2:2])
                tod = (tod - np.nanmedian(tod))/rms # normalise
                gd = (np.isnan(tod) == False) & (self.mask == 1)
                try:
                    #pmdl = np.poly1d(np.polyfit(time[gd], tod[gd], 5))
                    #tod -= pmdl(time)
                    A = 1./np.sin(self.el[item,:]*np.pi/180.)
                    pmdl = np.poly1d(np.polyfit(A[gd],tod[gd],1))
                    tod -= pmdl(A)
                    binSize = 12./60.
                    nbins = int((np.nanmax(self.el[item,:])-np.nanmin(self.el[item,:]) )/binSize)
                    elEdges= np.linspace(np.nanmin(self.el[item,:]),np.nanmax(self.el[item,:]),nbins+1)
                    elMids = (elEdges[:-1] + elEdges[1:])/2.
                    s = np.histogram(self.el[item,gd], elEdges, weights=tod[gd])[0]
                    w = np.histogram(self.el[item,gd], elEdges)[0]
                    pmdl = interp1d(elMids, s/w, bounds_error=False, fill_value=0)
                    tod -= pmdl(self.el[item,:])
                    tod[self.el[item,:] < elMids[0]] -= s[0]/w[0]
                    tod[self.el[item,:] > elMids[-1]] -= s[-1]/w[-1]
                except TypeError:
                    continue

                mask = self.mask*1
                mask[np.isnan(tod)] = 0
                binFuncs.binValues(self.map_bavg[sideband,:], pixels, weights=tod, mask=mask)

        self.map_bavg = np.reshape(self.map_bavg, (nSidebands, self.nypix, self.nxpix))
        self.hits = np.reshape(self.hits, (self.nypix, self.nxpix))
        return self.map_bavg/self.hits, self.hits  * self.tsamp

    def featureBits(self,features, target):
        
        output = np.zeros(features.size).astype(bool)
        power = np.floor(np.log(features)/np.log(2) )
        mask  = (features - 2**power) == 0
        output[power == target] = True
        for i in range(24):
            features[mask] -= 2**power[mask]
            power[mask] = np.floor(np.log(features[mask])/np.log(2) )
            mask  = (features - 2**power) == 0
            output[power == target] = True
        return output

    def setSource(self,source):
        self.source = source
        
    def setLevel1(self, datafile, source =''):
        """
        """
        self.setSource(source)
        
        self.datafile = datafile

        self.attributes = self.datafile['comap'].attrs

        self.tsamp = float(self.attributes['tsamp'].decode())
        self.obsid = self.attributes['obsid'].decode()
        self.source = self.attributes['source'].decode()
        
        self.tRa, self.tDec, _ = self.datafile['hk/antenna0/deTracker/equat_geoc'][:].T
        self.tRa = self.tRa[self.tRa.size//2]/(1000.*60.*60.)
        self.tDec = self.tDec[self.tRa.size//2]/(1000.*60.*60.)

        # load but do not read yet.
        self.ra = self.datafile['spectrometer/pixel_pointing/pixel_ra']
        self.el = self.datafile['spectrometer/pixel_pointing/pixel_el']
        self.dec= self.datafile['spectrometer/pixel_pointing/pixel_dec']
        
        self.tod_bavg = self.datafile['spectrometer/band_average']
        features = self.datafile['spectrometer/features'][:]
        self.mask = np.ones(features.size).astype(int)
        self.mask[self.featureBits(features.astype(float), 13)] = 0
        self.mask[features == 0] = 0
        self.mask = self.mask.astype(int)

        
        #self.mask[:] = 1
        # If we don't spe
        if isinstance(self.crval, type(None)):
            if self.source in sources:
                sRa,sDec = sources[self.source]()
                self.crval = [sRa,sDec]
            else:
                self.crval = [np.median(self.ra[0,:]),
                              np.median(self.dec[0,:])]

        self.setWCS(self.crval, self.cdelt, self.crpix, self.ctype)

    def plotImages(self, hit_filename='hitmap.png', bavg_filename='bavgmap.png',
                   feeds = [0],
                   plot_circle=False,plot_circle_radius=1):
        """
        Write out band average and hit map distributions
        """

        flist = [feed+1 for feed in feeds]
        if len(flist) > 1:
            fstr = '{:.0f}...{:.0f}'.format(np.min(flist), np.max(flist))
        else:
            fstr = '{:.0f}'.format(flist[0])

        # Plot the Hit map
        fig = pyplot.figure(figsize=(12,10))
        ax = pyplot.subplot(111,projection=self.wcs)

        pyplot.imshow(np.log10(self.hits), aspect='auto')
        low = int(np.min(np.log10(self.hits[self.hits > 0])))
        high= int(np.max(np.log10(self.hits[self.hits > 0]))) + 1
        cbar = pyplot.colorbar(label=r'Seconds', ticks=np.arange(low,high))
        cbar.ax.set_yticklabels([r'10$^{%s}$' % v for v in np.arange(low,high)])
        pyplot.xlabel('{}'.format(self.wcs.wcs.ctype[0].split('-')[0]))
        pyplot.ylabel('{}'.format(self.wcs.wcs.ctype[1].split('-')[0]))
        pyplot.title('Integration time: obsid {}, source {}, feeds {}'.format(self.obsid, self.source, fstr),size=16)
        pyplot.grid()

        lon = pyplot.gca().coords[0]
        lat = pyplot.gca().coords[1]

        if self.crpix[0]*self.cdelt[0]*2 > 1.5:
            lon.set_major_formatter('d')
            lon.set_minor_frequency(10)
            lon.display_minor_ticks(True)
        else:
            lon.set_major_formatter('d.d')

        if self.crpix[1]*self.cdelt[1]*2 > 1.5:
            lat.set_major_formatter('d')
            lat.set_minor_frequency(10)
            lat.display_minor_ticks(True)
        else:
            lat.set_major_formatter('d.d')
        
        pyplot.savefig(hit_filename,bbox_inches='tight')
        pyplot.clf()

        # Plot BandAverageMaps
        if self.makeAvgMap:
            fig.suptitle('Band Avg: obsid {}, source {}, feeds {}'.format(self.obsid, self.source, fstr),size=14)

            for band in range(self.map_bavg.shape[0]):
                ax = pyplot.subplot(2,2,1+band,projection=self.wcs)
                pyplot.imshow((self.map_bavg[band,...]), aspect='auto',vmin=-2,vmax=2)
                cbar = pyplot.colorbar(label=r'Normalised Units')
                pyplot.xlabel('{}'.format(self.wcs.wcs.ctype[0].split('-')[0]))
                pyplot.ylabel('{}'.format(self.wcs.wcs.ctype[1].split('-')[0]))

                if self.source in sources:
                    sRa,sDec = sources[self.source]()
                    pyplot.plot(sRa,sDec, 'xr', transform=pyplot.gca().get_transform('world'))
        
                pyplot.title('band {}'.format(band),size=16)
                pyplot.grid()

                if plot_circle:
                    pyplot.gca().add_patch(
                        Ellipse((self.crval[0],self.crval[1]),
                                plot_circle_radius/np.cos(self.crval[1]*np.pi/180.),
                                plot_circle_radius,
                                edgecolor='red',facecolor='none',
                                transform=pyplot.gca().get_transform('fk5'))
                    )

                lon = pyplot.gca().coords[0]
                lat = pyplot.gca().coords[1]
                                 
                                        
                if self.crpix[0]*self.cdelt[0]*2 > 1.5:
                    lon.set_major_formatter('d')
                    lon.set_minor_frequency(10)
                    lon.display_minor_ticks(True)
                else:
                    lon.set_major_formatter('d.d')
                    
                if self.crpix[1]*self.cdelt[1]*2 > 1.5:
                    lat.set_major_formatter('d')
                    lat.set_minor_frequency(10)
                    lat.display_minor_ticks(True)
                else:
                    lat.set_major_formatter('d.d')
                    
            pyplot.tight_layout(h_pad=4., w_pad=6., pad=4)
            pyplot.savefig(bavg_filename,bbox_inches='tight')

    def SaveMaps(self,filename):
        d = h5py.File(filename)

        dset = d.create_dataset('maps/bandavg', data=self.map_bavg)
        hset = d.create_dataset('maps/hitmap',data=self.hits)

        d.close()
        
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
@click.option('--cdelt', default=[1./60.,1./60.],cls=PythonLiteralOption, help='WCS cdelt parameter of form [x_pix, y_pix]')
@click.option('--field_width', default=[3.,3.], cls=PythonLiteralOption, help='Field width list of form [ra_width, dec_width]')
@click.option('--ctype', default=['RA---TAN','DEC--TAN'], cls=PythonLiteralOption, help='Field WCS ctype list of form [RATYPE, DECTYPE]')
@click.option('--crval', default=None, cls=PythonLiteralOption, help='Field centre list of form [RA_cen, Dec_cen], (Default: None, take ra/dec from average of scans)')
@click.option('--source', default=None, help='Source name for field centre, if source unknown ignore (Default: None, take ra/dec centre from average ra/dec)')
@click.option('--plot_circle',default=False,type=bool, help='Overplot a circle of radius plot_circle_radius (Default: False)')
@click.option('--plot_circle_radius', default=1,type=float, help='Radius of over plotted circle')
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
                        plot_circle_radius):

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
                   plot_circle_radius)
    

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
                   plot_circle_radius=1):
    
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

    xpixelWidth = int(field_width[0]/cdelt[0])
    ypixelWidth = int(field_width[1]/cdelt[1])

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
        #mapper.SaveMaps('{}/OutputMap.hd5'.format(image_directory))
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
        

if __name__ == "__main__":
    call_level1_hitmaps()
