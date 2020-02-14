import numpy as np
from astropy import wcs
from matplotlib import pyplot
import h5py
import binFuncs
from scipy.interpolate import interp1d
import os

from tqdm import tqdm 
from matplotlib.patches import Ellipse
from matplotlib import gridspec
from Utilities import Source, sources

class NormaliseFilter:
    def __init__(self,**kwargs):
        pass

    def __call__(self,DataClass, tod, **kwargs):
        rms = np.nanstd(tod[1:tod.size//2*2:2] - tod[0:tod.size//2*2:2])
        tod = (tod - np.nanmedian(tod))/rms # normalise
        return tod

class AtmosphereFilter:
    def __init__(self,**kwargs):
        pass

    def __call__(self,DataClass, tod, **kwargs):
        feed = kwargs['FEED']
        el   = DataClass.el[feed,:]
        mask = DataClass.atmmask

        gd = (np.isnan(tod) == False) & (mask == 1)
        try:
            # Calculate slab
            A = 1./np.sin(el*np.pi/180.)
            # Build atmospheric model
            pmdl = np.poly1d(np.polyfit(A[gd],tod[gd],1))
            # Subtract atmospheric slab
            tod -= pmdl(A)

            # Bin by elevation, and remove with interpolation (redundant?) 
            binSize = 12./60.
            nbins = int((np.nanmax(el)-np.nanmin(el) )/binSize)
            elEdges= np.linspace(np.nanmin(el),np.nanmax(el),nbins+1)
            elMids = (elEdges[:-1] + elEdges[1:])/2.
            s = np.histogram(el[gd], elEdges, weights=tod[gd])[0]
            w = np.histogram(el[gd], elEdges)[0]
            pmdl = interp1d(elMids, s/w, bounds_error=False, fill_value=0)
            tod -= pmdl(el)
            tod[el < elMids[0]] -= s[0]/w[0]
            tod[el > elMids[-1]] -= s[-1]/w[-1]
        except TypeError:
            return tod 


        return tod

def quick_cal(hkmjd,_status,_thot,mjd,tod):
    """
    A routine for quickly calculating the gain calibration
    """

    status = np.interp(mjd,hkmjd,_status)
    thot   = np.interp(mjd,hkmjd,_thot)

    vane_in = np.where((status == 1))[0]
    dvane = vane_in[1:] - vane_in[:-1]
    nvanes = len(np.where((dvane > 2))[0]) + 1
    if nvanes > 1:
        vane_edges = [[vane_in[0],vane_in[np.where((dvane > 2))[0][0]]],
                      [vane_in[np.where((dvane > 2))[0][0]+1],vane_in[-1]]]
    else:
        vane_edges = [[vane_in[0],vane_in[-1]]]

    vane_edges = np.array(vane_edges).astype(int)
    offset = int(20*50) # 20 second offset
    tcold  = 2.73
    gains = np.zeros(nvanes)
    for i, (low, high) in enumerate(vane_edges):
        phot = np.mean(tod[low:high])
        offset = int(offset*(-1)**i)
        pcold= np.mean(tod[low+offset:high+offset])
        gains[i] = (phot - pcold)/(np.mean(thot[low:high]) - tcold)
    return np.nanmean(gains)

class Mapper:

    def __init__(self, 
                 makeHitMap=True,
                 makeAvgMap=False,
                 crval=None, 
                 cdelt=[1.,1.], 
                 npix=None,#128,128],
                 ctype=['RA---TAN','DEC--TAN'],
                 image_directory='',
                 plot_circle=False,
                 plot_circle_radius=1):

        self.plot_circle=plot_circle
        self.plot_circle_radius=plot_circle_radius
        self.image_directory = image_directory
        # Cdelt given in arcmin
        self.crval = crval
        self.cdelt = [cd/60. for cd in cdelt]
        #self.crpix = crpix
        self.ctype = ctype
        self.makeHitMap = makeHitMap
        self.makeAvgMap = makeAvgMap

        self.npix = npix # int(crpix[0]*2)
        #self.nypix = int(crpix[1]*2)

        # Data containers
        self.x = None
        self.y = None
        self.tod_bavg = None
        self.hits = None
        self.map_bavg = None

        # TOD filters:
        self.filters = [AtmosphereFilter()]

    def __call__(self, items, usetqdm=False):
        """
        """

        if not isinstance(items, (range,tuple,list)):
            items = [items]

        # Store list of feed ids in array indexing
        self.feedlist = self.datafile['spectrometer/feeds'][...].astype(int)

        # Create a dictionary to map feedid to feed array index
        self.feedmap  = {feedid:feed_index for feed_index,feedid in enumerate(self.feedlist)}
        
        # Store a list of the feed array indices
        
        self.feed_indexs = [self.feedmap[feedid] for feedid in items if feedid in self.feedmap]
        if len(self.feed_indexs) == 0:
            raise ValueError('None of the chosen feeds are available')
        self.feed_ids    = [feedid for feedid in items if feedid in self.feedlist]
        self.usetqdm=usetqdm

        if isinstance(self.x, type(None)):
            print('No level 1 data loaded')
            return
        else:
            if self.makeAvgMap:
                self.map_bavg, self.hits = self.avgMap(self.feed_indexs, self.x, self.y, self.tod_bavg, self.mask)

                fstr = '-'.join(['{:02d}'.format(feed) for feed in self.feed_ids])

                outdir = '{}/Feeds-{}'.format(self.image_directory,fstr)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                # self.plotImages(self.map_bavg, self.hits,
                #                 '{}/Hitmap_Feeds-{}.png'.format(outdir,fstr),
                #                 '{}/BandAverage_Feeds-{}.png'.format(outdir,fstr),
                #                 self.plot_circle,
                #                 self.plot_circle_radius)
                # self.SaveMaps(self.map_bavg,
                #               '{}/BandAverage_Feeds-{}.fits'.format(outdir,fstr))

                return self.map_bavg, self.hits
            elif self.makeHitMap:
                self.hits = self.hitMap(self.feeds, self.x, self.y)
                return self.hits
            else:
                return
        
                
    def getFlatPixels(self, x, y):
        """
        Convert sky angles to pixel space
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

    def hitMap(self,feeds, x, y, mask):
        """
        Generate hit count maps
        """
        self.hits = np.zeros((self.nxpix*self.nypix))
        for feed in feeds:
            pixels = self.getFlatPixels(x[feed,:], y[feed,:])
            binFuncs.binValues(self.hits, pixels, mask=mask)
            
        return np.reshape(self.hits, (self.nypix, self.nxpix)) * self.tsamp

    def avgMap(self, feeds, x, y, d, mask):
        """
        Generate sky maps
        """

        doHits = True

        dataShape  = d.shape
        nSidebands = dataShape[1]
        map_bavg   = np.zeros((nSidebands, self.nxpix*self.nypix))
        hits       = np.zeros((self.nxpix*self.nypix))
        if self.usetqdm:
            tfeeds = tqdm(feeds)
        else:
            tfeeds= feeds

        # Loop over COMAP feeds/pixels
        mjd   = self.datafile['spectrometer/MJD'][...]
        hkmjd = self.datafile['hk/antenna0/vane/utc'][:]
        status= self.datafile['hk/antenna0/vane/status'][:]
        if np.sum(status) == 0:
            status = (self.datafile['hk/antenna0/vane/angle'][:] < 7000).astype(int)
        thot  = self.datafile['hk/antenna0/vane/Tvane'][:]/100. + 273.15
        
        for feed_index in tfeeds:

            # Get pixels from sky coordinates
            pixels = self.getFlatPixels(x[feed_index,:], y[feed_index,:])
            if doHits:
                binFuncs.binValues(hits, pixels, mask=mask)

            # Loop over sidebands
            for sideband in range(nSidebands):
                tod = d[feed_index,sideband,:]
                gain = quick_cal(hkmjd,status,thot,mjd,tod)
                tod /= gain
                for filtermode in self.filters:
                    tod = filtermode(self,tod,**{'FEED':feed_index,'SIDEBAND':sideband})

                _mask = mask*1
                _mask[np.isnan(tod)] = 0
                binFuncs.binValues(map_bavg[sideband,:], pixels, weights=tod.astype(float), mask=_mask)
        map_bavg = np.reshape(map_bavg, (nSidebands, self.nypix, self.nxpix))
        hits = np.reshape(hits, (self.nypix, self.nxpix))

        outputmap = map_bavg*1.
        outputmap[:,hits != 0] /= hits[hits !=0]
        return outputmap, hits  * self.tsamp

    def featureBits(self,features, target):
        """
        Return list of features encoded into feature bit
        """
        
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

    def setCrval(self):
        if isinstance(self.crval, type(None)):
            if self.source in sources:
                sRa,sDec = sources[self.source]()
                self.crval = [sRa,sDec]
            else:
                self.crval = [np.median(self.x[0,:]),
                              np.median(self.y[0,:])]

        if isinstance(self.npix, type(None)):
            xmax, xmin = np.max(self.x[...]), np.min(self.x[...])
            ymax, ymin = np.max(self.y[...]), np.min(self.y[...])
            self.nxpix = int((xmax-xmin)/self.cdelt[0])
            self.nypix = int((ymax-ymin)/self.cdelt[1])
        else:
            self.nxpix = int(self.npix[0])
            self.nypix = int(self.npix[1])
        # force equal aspect ratio
        largest_pix_axis = int(np.max([self.nxpix, self.nypix]))
        self.nxpix = largest_pix_axis
        self.nypix = largest_pix_axis

        self.crpix = [int((self.crval[0]-xmin)/self.cdelt[0]), int((self.crval[1]-ymin)/self.cdelt[1])]

    def setLevel1(self, datafile, source =''):
        """
        Store HDF5 file unit, read in any data that is needed
        """
        self.setSource(source)
        
        self.datafile = datafile

        self.attributes = self.datafile['comap'].attrs

        self.tsamp = float(self.attributes['tsamp'].decode())
        self.obsid = self.attributes['obsid'].decode()
        self.source = self.attributes['source'].decode()
        self.bandnames =self.datafile['spectrometer/bands'][:]

        # load but do not read yet.
        self.x = self.datafile['spectrometer/pixel_pointing/pixel_ra']
        self.y = self.datafile['spectrometer/pixel_pointing/pixel_dec']
        self.xCoordinateName = 'RA'
        self.yCoordinateName = 'Dec'

        self.el = self.datafile['spectrometer/pixel_pointing/pixel_el']

        
        self.tod_bavg = self.datafile['spectrometer/band_average']
        self.features = self.datafile['spectrometer/features'][:]
        self.mask = np.ones(self.features.size).astype(int)
        self.mask[self.featureBits(self.features.astype(float), 13)] = 0
        self.mask[self.features == 0] = 0
        self.mask = self.mask.astype(int)
        self.atmmask = self.mask
        
        # If we don't specify crval or npix then guess some reasonable values
        self.setCrval()

        self.setWCS(self.crval, self.cdelt, self.crpix, self.ctype)

    def plotImages(self, feeds=None,
                   hit_filename='hitmap.png', bavg_filename='bavgmap.png',
                   plot_circle=False,plot_circle_radius=1):
        """
        Write out band average and hit map distributions
        """

        if isinstance(feeds, type(None)):
            flist = [feed for feed in self.feed_ids]
        else:
            flist = [feed for feed in feeds if feed in self.feed_ids]

        if len(flist) > 1:
            fstr = '{:.0f}...{:.0f}'.format(np.min(flist), np.max(flist))
        else:
            fstr = '{:.0f}'.format(flist[0])

        # Plot the Hit map
        fig = pyplot.figure(figsize=(12,10))
        ax = pyplot.subplot(111,projection=self.wcs)
        loghits = self.hits*1
        loghits[loghits == 0] = np.nan
        pyplot.imshow(np.log10(loghits), aspect='auto')
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
        
        pyplot.gca().invert_xaxis()
        pyplot.savefig(hit_filename,bbox_inches='tight')
        pyplot.clf()

        # Plot BandAverageMaps
        if self.makeAvgMap:
            # Explicitly define matplotlib axes
            widths  = [2,0.2,2,0.2]
            heights = [2,2]
            
            spec = fig.add_gridspec(ncols=4,nrows=2,width_ratios=widths,
                                    height_ratios=heights)

            fig.suptitle('Band Avg: obsid {}, source {}, feeds {}'.format(self.obsid, self.source, fstr),size=14)
            positions = [[0,0],[2,0],[0,1],[2,1]]
            for band, (xp,yp) in enumerate(positions):
#range(self.map_bavg.shape[0]):
                #ax = pyplot.subplot(2,2,1+band,projection=self.wcs)
                ax = fig.add_subplot(spec[yp,xp], projection=self.wcs)
                ax_pos = ax.get_position()
                pad = 0.05
                if xp > 1:
                    pad *= -1
                else:
                    pad *= 1
                ypad = 0.05
                if yp == 0:
                    ypad *= -1
                else:
                    ypad *= 1
                #ax.set_position([ax_pos.x0+pad, ax_pos.y0+pad, ax_pos.width-pad*2, ax_pos.height-pad*2])
                ax.set_position([ax_pos.x0-pad, ax_pos.y0-ypad, ax_pos.width, ax_pos.height])

                pyplot.imshow((self.map_bavg[band,...]),aspect='equal') 
                #aspect='equal')
                cbar_ax = fig.add_subplot(spec[yp,xp+1])
                cax_pos = cbar_ax.get_position()
                ax_pos = ax.get_position()
                cbar_ax.set_position([cax_pos.x0-pad, ax_pos.y0, cax_pos.width, ax_pos.height])
                
                cbar = pyplot.colorbar(cax=cbar_ax)
                cbar.ax.tick_params(labelsize=12)
                cbar.set_label('K',size=14)
                pyplot.sca(ax)
                pyplot.xlabel('{}'.format(self.xCoordinateName))
                pyplot.ylabel('{}'.format(self.yCoordinateName))

                #if self.source in sources:
                #    sRa,sDec = sources[self.source]()
                #    pyplot.plot(sRa,sDec, 'xr', transform=pyplot.gca().get_transform('world'))
        
                pyplot.title('{}'.format(self.bandnames[band].decode('utf-8')),size=16)
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
                pyplot.gca().invert_xaxis()

            # pyplot.tight_layout(h_pad=0.0, w_pad=0.5)#, pad=4)
            pyplot.savefig(bavg_filename,bbox_inches='tight')

    def SaveMaps(self,map_bavg, filename):

        from astropy.io import fits
        header = self.wcs.to_header()
        hdu = fits.PrimaryHDU(map_bavg, header=header)
        hdu1 = fits.HDUList([hdu])
        hdu1.writeto(filename,overwrite=True)
        #d = h5py.File(filename)

        #dset = d.create_dataset('maps/bandavg', data=self.map_bavg)
        #hset = d.create_dataset('maps/hitmap',data=self.hits)

        #d.close()
