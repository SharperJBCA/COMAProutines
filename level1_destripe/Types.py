import numpy as np
import h5py
from astropy import wcs
from matplotlib import pyplot
from tqdm import tqdm
import pandas as pd
from scipy import linalg as la

import binFuncs


class Data:
    """
    Reads in the TOD, stores it immediate in a naive map
    """
    def __init__(self, parameters, frequency=0, band=0):
        
        self.nmodes = 5
        # -- constants
        self.ifeature = 5
        self.chunks = []
        self.datasizes = []
        self.Nsamples = 0
        self.Nhorns = 0
        self.Nbands = 4
        self.band = int(band)
        self.frequency = int(frequency)
        self.keeptod = False
        # -- read data
        filelist = np.loadtxt(parameters['Inputs']['filelist'],dtype=str,ndmin=1)

        # Setup the map coordinates
        self.crval = parameters['WCS']['crval']
        self.cdelt = [cd/60. for cd in parameters['WCS']['cdelt']]
        self.crpix = parameters['WCS']['crpix']
        self.ctype = parameters['WCS']['ctype']
        self.nxpix = int(self.crpix[0]*2)
        self.nypix = int(self.crpix[1]*2)
        self.setWCS(self.crval, self.cdelt, self.crpix, self.ctype)

        self.offsetLen = parameters['Destriper']['offset']

        self.Feeds  = parameters['Inputs']['feeds']
        try:
            self.Nfeeds = len(parameters['Inputs']['feeds'])
            self.Feeds = [int(f) for f in self.Feeds]
        except TypeError:
            self.Feeds = [int(self.Feeds)]
            self.Nfeeds = 1

        for filename in filelist:
            self.countDataSize(filename)

        Noffsets  = self.Nsamples//self.offsetLen

        self.pixels = np.zeros(self.Nsamples,dtype=int)

        if self.keeptod:
            self.todall = np.zeros(self.Nsamples)


        self.naive  = Map(self.nxpix,self.nypix, self.wcs)
        self.hits  = Map(self.nxpix,self.nypix, self.wcs)

        print('About to Read Data')
        for i, filename in enumerate(tqdm(filelist)):
            self.readData(i,filename)


        self.naive.average()
        # -- Finally create the noise offset residual
        self.residual = Offsets(self.offsetLen, Noffsets, self.Nsamples)
        for i, filename in enumerate(tqdm(filelist)):
            self.offsetResidual(i,filename)
        self.residual.average()
        

    def GetFeeds(self, feedlist, feeds):
        """
        Return feed index position
        """
        output = []
        for feed in feeds:
            pixel = np.where((feedlist == feed))[0]
            if (len(pixel) == 1):
                output += [pixel]

        return output

    def setWCS(self, crval, cdelt, crpix, ctype):
        """
        Declare world coordinate system for plots
        """
        self.wcs = wcs.WCS(naxis=2)
        self.wcs.wcs.crval = crval
        self.wcs.wcs.cdelt = cdelt
        self.wcs.wcs.crpix = crpix
        self.wcs.wcs.ctype = ctype

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



    def featureBits(self,features, target):
        """
        Return list of features encoded into feature bit
        """
        
        #output = np.zeros(features.size).astype(bool)
        #power = np.floor(np.log(features)/np.log(2) )
        #mask  = (features - 2**power) == 0
        #output[power == target] = True
        #for i in range(24):
        #    features[mask] -= 2**power[mask]
        #    power[mask] = np.floor(np.log(features[mask])/np.log(2) )
        #    mask  = (features - 2**power) == 0
        #    output[power == target] = True
        features[features == 0] = 0.1
        p2 = np.floor(np.log(features)/np.log(2))
        
        select = (p2 != 13) & (p2 != -1)
        a = np.where(select)[0]
        select[a[:1000]] = False
        return select
        

    def countDataSize(self,filename):
        """
        Get size of data for this file
        """
        
        d = h5py.File(filename,'r')
        features = d['spectrometer/features'][:]
        selectFeature = self.featureBits(features.astype(float), self.ifeature)
        N = len(features[selectFeature])
        d.close()

        N = (N//self.offsetLen) * self.offsetLen

        N = N*self.Nfeeds

        self.chunks += [[int(self.Nsamples), int(self.Nsamples+N)]]
        self.datasizes += [int(N/self.Nfeeds)]
        self.Nsamples += int(N)

    def processChunks(self,tod, step=10000):
        
        from scipy.linalg import inv
        from scipy.signal import medfilt
        nSteps = tod.shape[1]//step
        templates = np.ones((step,2))

        #nu = np.fft.fftfreq(tod.shape[1], d=1/50)
        #ps1 = np.abs(np.fft.fft(tod[0,:]))**2
        for i in range(nSteps):
            lo = i*step
            if i < (nSteps - 1):
                hi = (i+1)*step
            else:
                hi = tod.shape[1]
                templates = np.ones((hi - lo,2))

            #templates[:,0] = medfilt(np.median(tod[:,lo:hi],axis=0),151)
            # 
            
            #print(mdl.shape, tod[:,lo:hi].shape)
            
            medfilt_tod = np.zeros((tod.shape[0],hi-lo))
            binDown = 5
            binCount = int((hi-lo)/binDown +0.5)
            if binCount*binDown < tod.shape[1]:
                binCount += 1
            binEdges = np.linspace(0,hi-lo, binCount+1)
            positions = np.arange(hi-lo)
            w = np.histogram(positions, binEdges)[0]
            for feed in range(tod.shape[0]):
               s = np.histogram(positions, binEdges, weights=tod[feed,lo:hi])[0]
               #print(s.size*binDown, hi-lo)
               medfilt_tod[feed,:] = np.repeat(s/w, binDown)[:hi-lo]
                ##medfilt(tod,501)

            #C = medfilt_tod.dot(medfilt_tod.T)
            #C = tod[:,lo:hi].dot(tod[:,lo:hi].T)
            #U, s, Vh = la.svd(medfilt_tod, full_matrices=False)
            #print(U.shape, templates.shape, s.shape)
            #val, vec = linalg.eigh(C)
            #templates = Vh[0:1,:].T
            #C = templates.T.dot(templates)
            #Cinv = inv(C)
            #a = Cinv.dot(templates.T.dot(tod[:,lo:hi].T))
            #a = a.T
            #mdl = np.sum(templates[:,np.newaxis,:]*a[np.newaxis,:,:],axis=-1).T
            #icut = 5
            #mdl = U[:,:icut].dot(np.diag(s)[:icut,:].dot(Vh[:,:]))

            #for feed in range(Vh.shape[0]):
            #    pyplot.plot(tod[feed,lo:hi])
            #    pyplot.plot(mdl[feed,:])
            #    pyplot.show()
            #pyplot.subplot(211)
            #pyplot.imshow(C)
            #pyplot.subplot(212)
            #pyplot.imshow(C2)
            #pyplot.show()
            #pyplot.plot(tod[0,:])
            #pyplot.plot(mdl[0,:])
            #pyplot.plot(Vh[0,:])
            #pyplot.show()
            #pyplot.plot(tod[0,lo:hi])
            #pyplot.plot(medfilt_tod[0,:])
            #pyplot.show()
            tod[:,lo:hi] -= medfilt_tod #mdl
        #ps2 = np.abs(np.fft.fft(tod[0,:]))**2
        #pyplot.plot(nu[1:nu.size//2], ps1[1:nu.size//2])
        #pyplot.plot(nu[1:nu.size//2], ps2[1:nu.size//2])
        #pyplot.yscale('log')
        #pyplot.xscale('log')
        #pyplot.grid()
        #pyplot.show()
        return tod
       # pyplot.plot(tod[0,:])
        #pyplot.show()

    def readData(self, i, filename):
        """
        Reads data
        """

        gdir = '/scratch/nas_comap1/sharper/COMAP/runcomapreduce/AncillaryData/CalVanes/'
        gfile = filename.split('/')[-1].split('.')[0]+'_TsysGainRMS.pkl'
        gainDF = pd.read_pickle(gdir+gfile)
        idx = pd.IndexSlice
        gains = gainDF.loc(axis=0)[idx[:,:,'Gain',self.Feeds,self.band]].values.astype('float')
        gains = np.nanmedian(gains,axis=1)

        d = h5py.File(filename,'r')

        # -- Only want to look at the observation data
        features = d['spectrometer/features'][:]
        #print(np.unique(np.log(features)/np.log(2)))
        selectFeature = self.featureBits(features.astype(float), self.ifeature)
        features = features[selectFeature]
        

        # --- Feed position indices can change
        Feeds = self.GetFeeds(d['spectrometer/feeds'][...], self.Feeds)

        # We store all the pointing information
        x  = (d['spectrometer/pixel_pointing/pixel_ra'][...])[Feeds,selectFeature]
        x  = x[...,0:self.datasizes[i]].flatten()
        y  = (d['spectrometer/pixel_pointing/pixel_dec'][...])[Feeds,selectFeature]
        y  = y[...,0:self.datasizes[i]].flatten()

        el  = (d['spectrometer/pixel_pointing/pixel_el'][...])[Feeds,selectFeature]
        el  = el[...,0:self.datasizes[i]]


        pixels = self.getFlatPixels(x,y)
        pixels[pixels < 0] = -1
        pixels[pixels > self.naive.npix] = -1
        self.pixels[self.chunks[i][0]:self.chunks[i][1]] = pixels
        
        # Now accumulate the TOD into the naive map
        tod = ((d['spectrometer/band_average'][Feeds,:,:])[:,self.band,:])[:,selectFeature]
        tod = tod[...,0:self.datasizes[i]]

        #tod = np.zeros((todin.shape[0], todin.shape[1], np.sum(selectFeature)))
        #print(tod.shape)
        # print(features.shape)
        t = np.arange(tod.shape[-1])
        weights = np.ones(tod.shape)
        for j in range(tod.shape[0]):
            bad = np.isnan(tod[j,:])
            tod[j,bad] = np.interp(t[bad], t[~bad], tod[j,~bad])
            pmdl = np.poly1d(np.polyfit(1./np.sin(el[j,:]*np.pi/180.), tod[j,:],1))
            tod[j,:] -= pmdl(1./np.sin(el[j,:]*np.pi/180.))
            tod[j,:] -= np.nanmedian(tod[j,:])
            tod[j,:] /= gains[j]

            N = tod.shape[0]//2 * 2
            rms = np.nanstd(tod[j,1:N:2] - tod[j,0:N:2])
            weights[j,:] *= 1./rms**2
            print('Horn', j, rms)

        weights = weights.flatten()
        tod = tod.flatten()
        bad = (np.isnan(tod)) | (self.pixels[self.chunks[i][0]:self.chunks[i][1]] == -1)
        tod[bad] = 0
        weights[bad] = 0

        #pyplot.scatter(np.arange(tod[0,0,:].size),tod[0,0,:], c=np.log(features)/np.log(2))
        #pyplot.show()
        if self.keeptod:
            self.todall[self.chunks[i][0]:self.chunks[i][1]] = tod*1.


        self.naive.accumulate(tod,weights,pixels)
        self.hits.accumulatehits(pixels)

    def offsetResidual(self, i, filename):
        """
        Reads data
        """
        gdir = '/scratch/nas_comap1/sharper/COMAP/runcomapreduce/AncillaryData/CalVanes/'
        gfile = filename.split('/')[-1].split('.')[0]+'_TsysGainRMS.pkl'
        gainDF = pd.read_pickle(gdir+gfile)
        idx = pd.IndexSlice
        gains = gainDF.loc(axis=0)[idx[:,:,'Gain',self.Feeds,self.band]].values.astype('float')
        gains = np.nanmedian(gains,axis=1)


        d = h5py.File(filename,'r')

        # -- Only want to look at the observation data
        features = d['spectrometer/features'][:]
        selectFeature = self.featureBits(features.astype(float), self.ifeature)
        features = features[selectFeature]

        # --- Feed position indices can change
        Feeds = self.GetFeeds(d['spectrometer/feeds'][...], self.Feeds)


       
        # Now accumulate the TOD into the naive map
        el  = (d['spectrometer/pixel_pointing/pixel_el'][...])[Feeds,selectFeature]
        el = el[...,0:self.datasizes[i]]
        tod = (d['spectrometer/band_average'][Feeds,:,:])[:,self.band,selectFeature]
        tod = tod[...,0:self.datasizes[i]]
        t = np.arange(tod.shape[-1])
        weights = np.ones(tod.shape)
        for j in range(tod.shape[0]):
            bad = np.isnan(tod[j,:])
            tod[j,bad] = np.interp(t[bad], t[~bad], tod[j,~bad])
            pmdl = np.poly1d(np.polyfit(1./np.sin(el[j,:]*np.pi/180.), tod[j,:],1))
            tod[j,:] -= pmdl(1./np.sin(el[j,:]*np.pi/180.))
            tod[j,:] -= np.nanmedian(tod[j,:])
            tod[j,:] /= gains[j]

            N = tod.shape[0]//2 * 2
            rms = np.nanstd(tod[j,1:N:2] - tod[j,0:N:2])
            weights[j,:] *= 1./rms**2

        weights = weights.flatten()
        tod = tod.flatten()
        bad = (np.isnan(tod)) | (self.pixels[self.chunks[i][0]:self.chunks[i][1]] == -1)
        tod[bad] = 0
        weights[bad] = 0

        self.residual.accumulate(tod,weights,self.naive.output,self.pixels,self.chunks[i])

    def skyPixels(self,i, d,Feeds, selectFeature):
        """
        Returns the pixel coordinates in the WCS frame
        """

        # We store all the pointing information
        x  = (d['level1/spectrometer/pixel_pointing/pixel_ra'][...])[Feeds,selectFeature]
        x  = x[...,0:self.datasizes[i]].flatten()
        y  = (d['level1/spectrometer/pixel_pointing/pixel_dec'][...])[Feeds,selectFeature]
        y  = y[...,0:self.datasizes[i]].flatten()


        el  = (d['level1/spectrometer/pixel_pointing/pixel_el'][...])[Feeds,selectFeature]
        el  = el[...,0:self.datasizes[i]]


        pixels = self.getFlatPixels(x,y)
        pixels[pixels < 0] = -1
        pixels[pixels > self.naive.npix] = -1

        return pixels

class DataLevel2(Data):
    def __init__(self, parameters, band=0, frequency=0,keeptod=False):
        
        self.nmodes = 5
        # -- constants -- a lot of these are COMAP specific
        self.ifeature = 5
        self.chunks = []
        self.datasizes = []
        self.Nsamples = 0
        self.Nhorns = 0
        self.Nbands = 4

        self.band = int(band)
        self.frequency = int(frequency)

        self.keeptod = keeptod

        self.offsetLen = parameters['Destriper']['offset']

        self.Feeds  = parameters['Inputs']['feeds']
        try:
            self.Nfeeds = len(parameters['Inputs']['feeds'])
            self.Feeds = [int(f) for f in self.Feeds]
        except TypeError:
            self.Feeds = [int(self.Feeds)]
            self.Nfeeds = 1


        # Setup the map coordinates -- Needed for generating pixel coordinates
        self.crval = parameters['WCS']['crval']
        self.cdelt = [cd/60. for cd in parameters['WCS']['cdelt']]
        self.crpix = parameters['WCS']['crpix']
        self.ctype = parameters['WCS']['ctype']
        self.nxpix = int(self.crpix[0]*2)
        self.nypix = int(self.crpix[1]*2)
        self.setWCS(self.crval, self.cdelt, self.crpix, self.ctype)

        self.naive  = Map(self.nxpix,self.nypix, self.wcs)
        self.hits   = Map(self.nxpix,self.nypix, self.wcs)

        # -- read data
        filelist = np.loadtxt(parameters['Inputs']['filelist'],dtype=str,ndmin=1)

        # Will define Nsamples, datasizes[], and chunks[[]]
        for filename in filelist:
            self.countDataSize(filename)

        self.pixels = np.zeros(self.Nsamples,dtype=int)

        # If we want to keep all the TOD samples for plotting purposes...
        if self.keeptod:
            self.todall = np.zeros(self.Nsamples)
        self.allweights = np.zeros(self.Nsamples)



        # First read in all the data
        # Remember we want to solve Ax = b,
        # "b" contains all the data, so we construct that now:
        # 1a) Create a naive binned map
        # 1b) Sum all the data into offsets
        # 2) Subtract the naive weighted map from the offsets
        # "b" residual vector is saved in residual Offset object
        Noffsets  = self.Nsamples//self.offsetLen
        self.residual = Offsets(self.offsetLen, Noffsets, self.Nsamples)

        for i, filename in enumerate(tqdm(filelist)):
            self.readData(i,filename)
        self.naive.average()
        self.residual.accumulate(-self.naive.output[self.pixels],self.allweights,[0,self.pixels.size])

        self.residual.average()



        # -- Finally create the noise offset residual
        #for i, filename in enumerate(tqdm(filelist)):
        #    self.offsetResidual(i,filename)
        #self.residual.average()
        


    def countDataSize(self,filename):
        """
        Opens each datafile and determines the number of samples

        Uses the features to select the correct chunk of data
        """
        
        d = h5py.File(filename,'r')
        features = d['level1/spectrometer/features'][:]
        selectFeature = self.featureBits(features.astype(float), self.ifeature)
        N = len(features[selectFeature])
        d.close()

        N = (N//self.offsetLen) * self.offsetLen

        N = N*self.Nfeeds

        self.chunks += [[int(self.Nsamples), int(self.Nsamples+N)]]
        self.datasizes += [int(N/self.Nfeeds)]
        self.Nsamples += int(N)


    def readData(self, i, filename):
        """
        Reads data
        """

        gdir = '/scratch/nas_comap1/sharper/COMAP/runcomapreduce/AncillaryData/SourceFitsAllFeeds/'
        calfile = 'CalFactors_Nov2019_Jupiter.dat'
        caldata = np.loadtxt('{}/{}'.format(gdir,calfile))
    

        d = h5py.File(filename,'r')
        # -- Only want to look at the observation data
        features = d['level1/spectrometer/features'][:]
        selectFeature = self.featureBits(features.astype(float), self.ifeature)
        features = features[selectFeature]

        # --- Feed position indices can change
        Feeds = self.GetFeeds(d['level1/spectrometer/feeds'][...], self.Feeds)

        self.pixels[self.chunks[i][0]:self.chunks[i][1]] = self.skyPixels(i, d,Feeds, selectFeature)
        el  = (d['level1/spectrometer/pixel_pointing/pixel_el'][...])[Feeds,selectFeature]
        el  = el[...,0:self.datasizes[i]]

        # Now accumulate the TOD into the naive map
        tod = d['level2/averaged_tod'][Feeds,...]
        nFeeds, nBands, nChannels, nSamples = tod.shape

        #todbig = np.reshape(tod,(nFeeds*nBands*nChannels, nSamples))

        

        print('check badn/freq', self.band, self.frequency)
        tod = tod[:,self.band,self.frequency,selectFeature]
        tod = tod[...,0:self.datasizes[i]]

        # Replace with simulated data
        if False:
            x  = (d['level1/spectrometer/pixel_pointing/pixel_ra'][...])[Feeds,selectFeature]
            x  = x[...,0:self.datasizes[i]].flatten()
            y  = (d['level1/spectrometer/pixel_pointing/pixel_dec'][...])[Feeds,selectFeature]
            y  = y[...,0:self.datasizes[i]].flatten()
            tod[...] = np.random.normal(size=tod.shape)
            gauss2d = lambda P,x,y: P[0] * np.exp(-0.5*((x-P[1])**2 + (y-P[2])**2)/P[3]**2)
            xmid, ymid = np.mean(x),np.mean(y)
            todshape = tod.shape 
            tod = tod.flatten()
            tod += gauss2d([50,xmid,ymid,5/60.], x,y)
            tod = np.reshape(tod, todshape)


        weights = np.ones(tod.shape)
        t = np.arange(tod.shape[-1])
        for j, feed in enumerate(Feeds):

            bad = np.isnan(tod[j,:])
            if all(bad):
                continue
            tod[j,bad] = np.interp(t[bad], t[~bad], tod[j,~bad])
            pmdl = np.poly1d(np.polyfit(1./np.sin(el[j,:]*np.pi/180.), tod[j,:],1))
            tod[j,:] -= pmdl(1./np.sin(el[j,:]*np.pi/180.))
            tod[j,:] -= np.nanmedian(tod[j,:])

            ifeed = np.where((caldata[:,0] == self.Feeds[j]))[0]

            N = tod.shape[1]//2 * 2
            diffTOD = tod[j,:N:2]-tod[j,1:N:2]
            rms = np.sqrt(np.nanmedian(diffTOD**2)*1.4826)
            weights[j,:] *= 1./rms**2

            # Remove spikes
            select = np.where((np.abs(diffTOD) > rms*5))[0]*2
            weights[j,select] *= 1e-10
            print('Horn', self.Feeds[j], rms)

        #tod = self.processChunks(tod, step=10000)
        rms = np.sqrt(np.nanmedian(tod**2,axis=1))*1.4826
        for j, feed in enumerate(Feeds):
            select = np.abs(tod[j,:]) > rms[j]*10
            weights[j,select] *= 1e-10

        weights = weights.flatten()
        tod = tod.flatten()
        bad = (np.isnan(tod)) | (self.pixels[self.chunks[i][0]:self.chunks[i][1]] == -1)
        tod[bad] = 0
        weights[bad] = 0

        if self.keeptod:
            self.todall[self.chunks[i][0]:self.chunks[i][1]] = tod*1.

        self.allweights[self.chunks[i][0]:self.chunks[i][1]] = weights
        
        self.naive.accumulate(tod,weights,self.pixels[self.chunks[i][0]:self.chunks[i][1]])
        self.hits.accumulatehits(self.pixels[self.chunks[i][0]:self.chunks[i][1]])
        self.residual.accumulate(tod,weights,self.chunks[i])


class DataSim(Data):
    def __init__(self, parameters):
        
        self.nmodes = 5
        # -- constants
        self.ifeature = 5
        self.chunks = []
        self.datasizes = []
        self.Nsamples = 0
        self.Nhorns = 0
        self.Nbands = 4

        self.band = 0
        self.frequency = 0

        self.keeptod = False
        # -- read data
        filelist = np.loadtxt(parameters['Inputs']['filelist'],dtype=str,ndmin=1)

        # Setup the map coordinates
        self.crval = parameters['WCS']['crval']
        self.cdelt = [cd/60. for cd in parameters['WCS']['cdelt']]
        self.crpix = parameters['WCS']['crpix']
        self.ctype = parameters['WCS']['ctype']
        self.nxpix = int(self.crpix[0]*2)
        self.nypix = int(self.crpix[1]*2)
        self.setWCS(self.crval, self.cdelt, self.crpix, self.ctype)

        self.offsetLen = parameters['Destriper']['offset']

        self.Feeds  = parameters['Inputs']['feeds']
        try:
            self.Nfeeds = len(parameters['Inputs']['feeds'])
            self.Feeds = [int(f) for f in self.Feeds]
        except TypeError:
            self.Feeds = [int(self.Feeds)]
            self.Nfeeds = 1

        for filename in filelist:
            self.countDataSize(filename)

        Noffsets  = self.Nsamples//self.offsetLen

        self.pixels = np.zeros(self.Nsamples,dtype=int)

        if self.keeptod:
            self.todall = np.zeros(self.Nsamples)


        self.naive  = Map(self.nxpix,self.nypix, self.wcs)
        for i, filename in enumerate(tqdm(filelist)):
            self.readData(i,filename)


        self.naive.average()
        # -- Finally create the noise offset residual
        self.residual = Offsets(self.offsetLen, Noffsets, self.Nsamples)
        for i, filename in enumerate(tqdm(filelist)):
            self.offsetResidual(i,filename)
        self.residual.average()
        


    def countDataSize(self,filename):
        """
        Get size of data for this file
        """
        
        d = h5py.File(filename,'r')
        features = d['spectrometer/features'][:]
        selectFeature = self.featureBits(features.astype(float), self.ifeature)
        N = len(features[selectFeature])
        d.close()

        N = (N//self.offsetLen) * self.offsetLen

        N = N*self.Nfeeds

        self.chunks += [[int(self.Nsamples), int(self.Nsamples+N)]]
        self.datasizes += [int(N/self.Nfeeds)]
        self.Nsamples += int(N)


    def readData(self, i, filename):
        """
        Reads data
        """
    

        d = h5py.File(filename,'r')
        # -- Only want to look at the observation data
        features = d['spectrometer/features'][:]
        selectFeature = self.featureBits(features.astype(float), self.ifeature)
        features = features[selectFeature]

        # --- Feed position indices can change
        Feeds = self.GetFeeds(d['spectrometer/feeds'][...], self.Feeds)

        # We store all the pointing information
        x  = (d['spectrometer/pixel_pointing/pixel_ra'][...])[Feeds,selectFeature]
        x  = x[...,0:self.datasizes[i]].flatten()
        y  = (d['spectrometer/pixel_pointing/pixel_dec'][...])[Feeds,selectFeature]
        y  = y[...,0:self.datasizes[i]].flatten()

        el  = (d['spectrometer/pixel_pointing/pixel_el'][...])[Feeds,selectFeature]
        el  = el[...,0:self.datasizes[i]]


        pixels = self.getFlatPixels(x,y)
        pixels[pixels < 0] = -1
        pixels[pixels > self.naive.npix] = -1
        self.pixels[self.chunks[i][0]:self.chunks[i][1]] = pixels
        
        # Now accumulate the TOD into the naive map
        tod = d['spectrometer/band_average'][Feeds,...]
        nFeeds, nBands,  nSamples = tod.shape
        
        tod = tod[:,self.band,selectFeature]
        tod  = tod[...,0:self.datasizes[i]]


        t = np.arange(tod.shape[-1])
        weights = np.ones(tod.shape)
        print(tod.shape, el.shape, nSamples,d['spectrometer/pixel_pointing/pixel_el'].shape)
        for j, feed in enumerate(Feeds):
            bad = np.isnan(tod[j,:])
            tod[j,bad] = np.interp(t[bad], t[~bad], tod[j,~bad])
            pmdl = np.poly1d(np.polyfit(1./np.sin(el[j,:]*np.pi/180.), tod[j,:],1))
            tod[j,:] -= pmdl(1./np.sin(el[j,:]*np.pi/180.))
            tod[j,:] -= np.nanmedian(tod[j,:])

            #pyplot.plot(tod[j,:])
            #pyplot.show()
            N = tod.shape[0]//2 * 2
            rms = np.nanstd(tod[j,1:N:2] - tod[j,0:N:2])
            weights[j,:] *= 1./rms**2
            print('Horn', j, rms)

        from scipy.signal import medfilt
        bkgd = medfilt(np.nanmedian(tod,axis=0),kernel_size=401)
        tod -= bkgd[np.newaxis,:]
        # pyplot.subplot(2,2,1)
        # pyplot.plot(tod[0,:])
        # pyplot.plot(bkgd)
        # pyplot.subplot(2,2,2)
        # pyplot.plot(tod[-1,:])
        # pyplot.plot(bkgd)
        # pyplot.subplot(2,2,3)
        # pyplot.plot(tod[-5,:])
        # pyplot.plot(bkgd)
        # pyplot.subplot(2,2,4)
        # pyplot.plot(tod[3,:])
        # pyplot.plot(bkgd)

        # pyplot.show()

        weights = weights.flatten()
        tod = tod.flatten()
        bad = (np.isnan(tod)) | (self.pixels[self.chunks[i][0]:self.chunks[i][1]] == -1)
        tod[bad] = 0
        weights[bad] = 0

        if self.keeptod:
            self.todall[self.chunks[i][0]:self.chunks[i][1]] = tod*1.


        self.naive.accumulate(tod,weights,pixels)
        #self.residual.accumulate(tod,weights,self..output,self.pixels,self.chunks[i])

    def offsetResidual(self, i, filename):
        """
        Reads data
        """

        d = h5py.File(filename,'r')

        # -- Only want to look at the observation data
        features = d['spectrometer/features'][:]
        selectFeature = self.featureBits(features.astype(float), self.ifeature)
        features = features[selectFeature]

        # --- Feed position indices can change
        Feeds = self.GetFeeds(d['spectrometer/feeds'][...], self.Feeds)


        el  = (d['spectrometer/pixel_pointing/pixel_el'][...])[Feeds,selectFeature]
        el  = el[...,0:self.datasizes[i]]

        # Now accumulate the TOD into the naive map
        tod = d['spectrometer/band_average'][Feeds,...]
        nFeeds, nBands, nSamples = tod.shape
        tod = tod[:,self.band,selectFeature]
        tod  = tod[...,0:self.datasizes[i]]

        t = np.arange(tod.shape[-1])
        weights = np.ones(tod.shape)
        for j, feed in enumerate(Feeds):
            bad = np.isnan(tod[j,:])
            tod[j,bad] = np.interp(t[bad], t[~bad], tod[j,~bad])
            pmdl = np.poly1d(np.polyfit(1./np.sin(el[j,:]*np.pi/180.), tod[j,:],1))
            tod[j,:] -= pmdl(1./np.sin(el[j,:]*np.pi/180.))
            tod[j,:] -= np.nanmedian(tod[j,:])


            N = tod.shape[0]//2 * 2
            rms = np.nanstd(tod[j,1:N:2] - tod[j,0:N:2])
            weights[j,:] *= 1./rms**2
            print('Horn', j, rms)
        from scipy.signal import medfilt
        bkgd = medfilt(np.nanmedian(tod,axis=0),kernel_size=401)
        tod -= bkgd[np.newaxis,:]


        weights = weights.flatten()
        tod = tod.flatten()
        bad = (np.isnan(tod)) | (self.pixels[self.chunks[i][0]:self.chunks[i][1]] == -1)
        tod[bad] = 0
        weights[bad] = 0

        self.residual.accumulate(tod,weights,self.naive.output,self.pixels,self.chunks[i])


class Map:
    """
    Stores pixel information
    """
    def __init__(self,nxpix, nypix,wcs,storehits=False):

        self.storehits = storehits
        # --- Need to create several arrays:
        # 1) Main output map array
        # 2) Signal*Weights array
        # 3) Weights array
        # 4) Hits

        self.wcs = wcs
        self.npix = nypix*nxpix
        self.nypix = nypix
        self.nxpix = nxpix
        self.output = np.zeros(self.npix)
        self.sigwei = np.zeros(self.npix)
        self.wei    = np.zeros(self.npix)
        if self.storehits:
            self.hits = np.zeros(self.npix)

    def clearmaps(self):
        self.output *= 0
        self.sigwei *= 0
        self.wei *= 0
        if self.storehits:
            self.hits *= 0

    def accumulate(self,tod,weights,pixels):
        """
        Add more data to the naive map
        """
        binFuncs.binValues(self.sigwei, pixels, weights=tod*weights)
        binFuncs.binValues(self.wei   , pixels, weights=weights    )
        if self.storehits:
            binFuncs.binValues(self.hits, pixels,mask=weights)

    def accumulatehits(self,pixels):
        binFuncs.binValues(self.sigwei,pixels)

    def binOffsets(self,offsets,weights,offsetpixels,pixels):
        """
        Add more data to the naive map
        """
        binFuncs.binValues2Map(self.sigwei, pixels, offsets*weights, offsetpixels)
        binFuncs.binValues2Map(self.wei   , pixels, weights        , offsetpixels)



    def __call__(self, average=False, returnsum=False):
        if average:
            self.average()
        
        if returnsum:
            return np.reshape(self.sigwei, (self.nypix, self.nxpix))

        return np.reshape(self.output, (self.nypix, self.nxpix))


    def __getitem__(self,pixels, average=False):
        if average:
            self.average()
        return self.output[pixels]

    def average(self):
        self.goodpix = np.where((self.wei != 0 ))[0]
        self.output[self.goodpix] = self.sigwei[self.goodpix]/self.wei[self.goodpix]

class Offsets:
    """
    Stores offset information
    """
    def __init__(self,offset, Noffsets, Nsamples):
        """
        """
        
        self.Noffsets= int(Noffsets)
        self.offset = int(offset)
        self.Nsamples = int(Nsamples )

        self.offsets = np.zeros(self.Noffsets)

        self.sigwei = np.zeros(self.Noffsets)
        self.wei    = np.zeros(self.Noffsets)

        self.offsetpixels = np.arange(self.Nsamples)//self.offset

    def __getitem__(self,i):
        """
        """
        
        return self.offsets[i//self.offset]

    def __call__(self):
        return np.repeat(self.offsets, self.offset)[:self.Nsamples]


    def accumulate(self,tod,weights,chunk):
        """
        Add more data to residual offset
        """
        binFuncs.binValues(self.sigwei, self.offsetpixels[chunk[0]:chunk[1]], weights=tod*weights )
        binFuncs.binValues(self.wei   , self.offsetpixels[chunk[0]:chunk[1]], weights=weights    )


    def average(self):
        self.goodpix = np.where((self.wei != 0 ))[0]
        self.offsets[self.goodpix] = self.sigwei[self.goodpix]/self.wei[self.goodpix]
