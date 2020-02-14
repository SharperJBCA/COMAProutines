import numpy as np
from astropy import wcs
from matplotlib import pyplot
import h5py
import binFuncs
from scipy.interpolate import interp1d
import os
import pandas as pd

from matplotlib.patches import Ellipse

import subscans
from Utilities import Source, sources

from Mapping import Mapper,NormaliseFilter,AtmosphereFilter
from Utilities import Source
try:
    from comancpipeline.Tools import Coordinates
except:
    Coordinates = None


class MapperSubScans(Mapper):        

    def __call__(self, items, usetqdm=False):
        """
        """

        self.usetqdm = usetqdm
        if not isinstance(items, (range,tuple,list)):
            items = [items]

        # --- Select the feeds
        feedlist = self.datafile['spectrometer/feeds'][...].astype(int)

        getfeeds = lambda f: np.argmin((f-feedlist)**2)            
        self.feeds = [feed for feed in map(getfeeds,items)]
        self.feednames = feedlist[self.feeds]

        # --- declare dicts to store output maps
        self.map_bavg = {}
        self.hits = {}

        if isinstance(self.x, type(None)):
            print('No level 1 data loaded')
            return
        else:
            # --- in this instance we are splitting the mapping into subscans
            self.tsys = {}
            for scan in range(self.Nscans):
                if scan in self.map_bavg:
                    self.map_bavg[scan] *= 0
                    self.hits[scan] *= 0 
                self.el = self.y[scan]
                self.atmmask = self.mask[scan]


                # If we don't spe
                self.crval= None
                self.setCrval(self.x[scan][0,:],self.y[scan][0,:])
                self.setWCS(self.crval, self.cdelt, self.crpix, self.ctype)

                # atms = AtmosphereFilter()
                # feed12 = np.where((feedlist == 1))[0][0]
                # print(self.mask[scan].shape, self.x[scan].shape, self.tod_bavg[scan].shape)
                # pyplot.figure()
                # pyplot.subplot(2,1,1)
                # self.el = self.y[3]
                # self.atmmask = self.mask[3]
                # #pyplot.plot(atms(self,self.tod_bavg[3][feed12,0,:],**{'FEED':feed12})-0.25, label='Subscan 3')
                # pyplot.plot(self.tod_bavg[3][feed12,0,:], label='Subscan 3')
                # self.el = self.y[4]
                # self.atmmask = self.mask[4]
                # #pyplot.plot(atms(self,self.tod_bavg[4][feed12,0,:],**{'FEED':feed12})+0.25, label='Subscan 4')
                # pyplot.plot(self.tod_bavg[4][feed12,0,:], label='Subscan 4')
                # pyplot.grid()
                # pyplot.legend()
                # pyplot.ylabel('K')
                # pyplot.xlabel('')
                # pyplot.title('{}'.format(self.datafile.filename.split('/')[-1].split('.')[0]))
                # pyplot.xlim(0,5000)

                # pyplot.subplot(2,1,2)
                # self.el = self.y[3]
                # self.atmmask = self.mask[3]
                # pyplot.plot(atms(self,self.tod_bavg[3][feed12,0,:],**{'FEED':feed12})-0.25, label='Subscan 3')
                # self.el = self.y[4]
                # self.atmmask = self.mask[4]
                # pyplot.plot(atms(self,self.tod_bavg[4][feed12,0,:],**{'FEED':feed12})+0.25, label='Subscan 4')
                # pyplot.grid()
                # pyplot.ylabel('K')
                # pyplot.xlim(0,5000)
                # pyplot.savefig('{}_todcompare.png'.format(self.datafile.filename.split('/')[-1].split('.')[0]),
                #                bbox_inches='tight')
                # pyplot.show()

                N = self.tod_bavg[scan].shape[-1]//2*2
                diff = self.tod_bavg[scan][:,:,:N:2] - self.tod_bavg[scan][:,:,1:N:2]
                dt, beta = 1./50. , self.bw
                self.tsys[scan] = np.nanstd(diff,axis=-1)/np.sqrt(2) * np.sqrt(dt*beta)


                             

                self.map_bavg[scan], self.hits[scan] = self.avgMap(self.feeds, 
                                                                   self.x[scan], 
                                                                   self.y[scan], 
                                                                   self.tod_bavg[scan],
                                                                   self.mask[scan])    

                fstr = '-'.join(['{:02d}'.format(feed) for feed in self.feeds])

                outdir = '{}/Feeds-{}'.format(self.image_directory,fstr)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                self.plotImages(self.map_bavg[scan], self.hits[scan],
                                '{}/Hitmap_scan{:02d}_Feeds-{}.png'.format(outdir,scan,fstr),
                                '{}/BandAverage_scan{:02d}_Feeds-{}.png'.format(outdir,scan,fstr),
                                self.feeds,
                                self.plot_circle,
                                self.plot_circle_radius)
                self.SaveMaps(self.map_bavg[scan],
                              '{}/BandAverage_scan{:02d}_Feeds-{}.fits'.format(outdir,scan,fstr))

                pyplot.figure(10)
                pyplot.plot(scan, self.tsys[scan][0,0],'.')
                pyplot.xlabel('Sub Scan')
                pyplot.ylabel('Tsys')
                pyplot.title('{}'.format(self.datafile.filename.split('/')[-1].split('.')[0]))

            pyplot.figure(10)
            pyplot.savefig('{}_tsys_subscan.png'.format(self.datafile.filename.split('/')[-1].split('.')[0]),
                           bbox_inches='tight')


    def setCrval(self,x,y):
        if isinstance(self.crval, type(None)):
            self.crval = [np.median(x),
                          np.median(y)]



    def setLevel1(self, datafile, source =''):
        """
        For sub-scans we want to split self.x, self.y and self.bavg into the specific sub-scans of the observation.
        A sub-scan is defined as when the telescope pointing center changes.
        """

        # --- Set all the ancillary information needed
        self.datafile = datafile
        filename = self.datafile.filename.split('/')[-1].split('.')[0]

        ancildir = '/scratch/nas_comap1/sharper/COMAP/runcomapreduce/AncillaryData/CalVanes'
        self.calvanedir = '{}/{}_TsysGainRMS.pkl'.format(ancildir,filename)
        if os.path.isfile(self.calvanedir):
            idx = pd.IndexSlice
            self.calvane = pd.read_pickle(self.calvanedir).loc(axis=0)[idx[:,:,['Gain','RMS'],:,:]]
            index = self.calvane.index
            self.calvane.index = index.droplevel(level=[0,1])


        self.setSource(source)

        self.teleLon = self.datafile['hk/antenna0/tracker/siteActual'][0,0]/(60.**2 * 1000.)
        self.teleLat = self.datafile['hk/antenna0/tracker/siteActual'][0,1]/(60.**2 * 1000.)
        
        self.datafile = datafile
        feedlist = self.datafile['spectrometer/feeds'][...].astype(int)

        self.attributes = self.datafile['comap'].attrs

        self.tsamp = float(self.attributes['tsamp'].decode())
        self.obsid = self.attributes['obsid'].decode()
        self.source = self.attributes['source'].decode()
        
        # --- Now go through all the data to locate the subscans
        azCen = self.datafile['spectrometer/pixel_pointing/pixel_az'][0,:]
        elCen = self.datafile['spectrometer/pixel_pointing/pixel_el'][0,:]
        features = self.datafile['spectrometer/features'][:]
        scan = np.where((features == np.max(features)))[0]

        
        edges = np.zeros(azCen.size).astype(int)
        var = np.zeros(azCen.size)
        edges[scan],var[scan] = subscans.subscanedges(azCen[scan],100, 0.01)
        edgepoints = subscans.findmidpoints(edges)
        edgepoints = edgepoints[edgepoints != 0]
        self.Nscans = len(edgepoints) - 1

        # print(scan.size)
        # pyplot.plot(azCen[:])
        # pyplot.plot(var)
        # for thisscan in edgepoints:
        #     pyplot.axvline(thisscan,color='k')
        # pyplot.show()

        # print(midpoints)
        # pyplot.subplot(2,1,1)
        # pyplot.plot(azCen,'.')
        # for midpoint in midpoints:
        #     pyplot.axvline(midpoint,color='k')
        # pyplot.subplot(2,1,2)
        # pyplot.plot(edges,'.')
        # for midpoint in midpoints:
        #     pyplot.axvline(midpoint,color='k')
        # pyplot.show()
        
        if os.path.isfile('{}_bandavg.hd5'.format(filename)):
            band_average = h5py.File('{}_bandavg.hd5'.format(filename),'r')['band_average'][...]
            self.bw = h5py.File('{}_bandavg.hd5'.format(filename),'r')['band_width'][...]
        else:
            band_average = np.zeros(self.datafile['spectrometer/band_average'].shape)
            self.bw = np.zeros((self.datafile['spectrometer/band_average'].shape[0],
                                self.datafile['spectrometer/band_average'].shape[1]))
            tod = self.datafile['spectrometer/tod']#[:,:,:,edgepoints[scan]:edgepoints[scan+1]]
            from tqdm import tqdm
            for horn in tqdm(range(tod.shape[0])):
                for sb in range(tod.shape[1]):
                    todslice = tod[horn,sb,:,:]
                    weights = 1./self.calvane.loc(axis=0)[idx['RMS',feedlist[horn],sb]].values.astype(float)**2
                    gain = self.calvane.loc(axis=0)[idx['Gain',feedlist[horn],sb]].values.astype(float)
                    weights[np.isinf(weights)]= 0
                    todavg = np.nansum(todslice[...]*weights[:,np.newaxis],axis=0)/np.nansum(weights)
                    gainavg = np.nansum(gain*weights)/np.nansum(weights)*2
                    band_average[horn,sb,:] = todavg/gainavg
                    nchans = np.nansum(np.arange(1024)*weights)/np.nansum(weights)
                    self.bw[horn,sb] = 1e9/1024 * nchans  # *1e9/1024  # 1e9*np.nansum(np.ones(1024)*weights)/1024
                    print(self.bw[horn,sb],nchans)
            band_average_file =  h5py.File('{}_bandavg.hd5'.format(filename))
            band_average_file.create_dataset('band_average',data=band_average)
            band_average_file.create_dataset('band_width',data=self.bw)
            band_average_file.close()

        # Load each scan
        self.x, self.y, self.tod_bavg, self.mask = {}, {} ,{}, {}
        idx = pd.IndexSlice
        for scan in range(self.Nscans):
            self.x[scan] = self.datafile['spectrometer/pixel_pointing/pixel_az'][:,edgepoints[scan]:edgepoints[scan+1]]
            self.y[scan] = self.datafile['spectrometer/pixel_pointing/pixel_el'][:,edgepoints[scan]:edgepoints[scan+1]]
            self.tod_bavg[scan] = band_average[:,:,edgepoints[scan]:edgepoints[scan+1]]
            self.mask[scan] = np.ones(self.x[scan].shape[-1]).astype(int)
        
        self.xCoordinateName = 'Azimuth'
        self.yCoordinateName = 'Elevation'
