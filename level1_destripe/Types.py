import numpy as np
import h5py
from astropy import wcs
from matplotlib import pyplot
from tqdm import tqdm
import pandas as pd
from scipy import linalg as la
import healpy as hp

import binFuncs
from scipy import signal
def butter_highpass(cutoff, fs, order=5,btype='highpass'):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b,a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    return b,a

def butt_bandpass(data, cutoff, fs, order=3):
    b,a = butter_highpass(cutoff,fs,order=order,btype='bandpass')
    y = signal.filtfilt(b,a,data)
    return data-y

def butt_highpass(data, cutoff, fs, order=3):
    b,a = butter_highpass(cutoff,fs,order=order)
    y = signal.filtfilt(b,a,data)
    return y
def butt_lowpass(data, cutoff, fs, order=3):
    b,a = butter_highpass(cutoff,fs,order=order,btype='lowpass')
    y = signal.filtfilt(b,a,data)
    return y



def removeplane(img, slce=0.4):
    """
    Remove a quadratic 2D plane from an image
    """
    img[img == 0] = np.nan

    xr, yr = np.arange(slce*img.shape[0],(1-slce)*img.shape[0],dtype=int),\
             np.arange(slce*img.shape[1],(1-slce)*img.shape[1],dtype=int)
    x, y = np.meshgrid(xr,yr)

    
    subimg = img[xr[0]:xr[-1]+1,yr[0]:yr[-1]+1]
    imgf = subimg[np.isfinite(subimg)].flatten()

    vecs = np.ones((5,imgf.size))
    vecs[0,:] = x[np.isfinite(subimg)].flatten()
    vecs[1,:] = y[np.isfinite(subimg)].flatten()
    vecs[2,:] = x[np.isfinite(subimg)].flatten()**2
    vecs[3,:] = y[np.isfinite(subimg)].flatten()**2

    C = vecs.dot(vecs.T)
    xv = la.inv(C).dot(vecs.dot(imgf[:,np.newaxis]))
    x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))

    img -= (xv[0]*x    + xv[1]*y    + \
            xv[2]*x**2 + xv[3]*y**2 + \
            xv[4])
    return img

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

        output = np.array(output).flatten().astype(int)
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
            #print('Horn', j, rms)

        weights = weights.flatten()
        tod = tod.flatten()
        bad = (np.isnan(tod)) | (self.pixels[self.chunks[i][0]:self.chunks[i][1]] == -1)
        tod[bad] = 0
        weights[bad] = 0

        #pyplot.scatter(np.arange(tod[0,0,:].size),tod[0,0,:], c=np.log(features)/np.log(2))
        #pyplot.show()
        if self.keeptod:
            self.todall[self.chunks[i][0]:self.chunks[i][1]] = tod*1.

        
        self.naive[(band,frequency)].accumulate(tod,weights,pixels)
        self.hits[(band,frequency)].accumulatehits(pixels)

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
        x  = (d['level1/spectrometer/pixel_pointing/pixel_ra'][...])[Feeds[:,None],selectFeature]
        x  = x[...,0:self.datasizes[i]].flatten()
        y  = (d['level1/spectrometer/pixel_pointing/pixel_dec'][...])[Feeds[:,None],selectFeature]
        y  = y[...,0:self.datasizes[i]].flatten()


        el  = (d['level1/spectrometer/pixel_pointing/pixel_el'][...])[Feeds[:,None],selectFeature]
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

        #gdir = '/scratch/nas_comap1/sharper/COMAP/runcomapreduce/AncillaryData/SourceFitsAllFeeds/'
        #calfile = 'CalFactors_Nov2019_Jupiter.dat'
        #caldata = np.loadtxt('{}/{}'.format(gdir,calfile))
    

        d = h5py.File(filename,'r')
        # -- Only want to look at the observation data
        features = d['level1/spectrometer/features'][:]
        selectFeature = self.featureBits(features.astype(float), self.ifeature)
        features = features[selectFeature]

        # --- Feed position indices can change
        self.FeedIndex = self.GetFeeds(d['level1/spectrometer/feeds'][...], self.Feeds)

        self.pixels[self.chunks[i][0]:self.chunks[i][1]] = self.skyPixels(i, d,self.FeedIndex, selectFeature)
        el  = (d['level1/spectrometer/pixel_pointing/pixel_el'][...])[self.FeedIndex[:,None],selectFeature]
        el  = el[...,0:self.datasizes[i]]

        # Now accumulate the TOD into the naive map
        tod = d['level2/averaged_tod'][self.FeedIndex,...]
        nFeeds, nBands, nChannels, nSamples = tod.shape


        #todbig = np.reshape(tod,(nFeeds*nBands*nChannels, nSamples))

        

        print('check band/freq', self.band, self.frequency)
        tod = tod[:,self.band,self.frequency,selectFeature]
        tod = tod[...,0:self.datasizes[i]]

        # Replace with simulated data
        if False:
            x  = (d['level1/spectrometer/pixel_pointing/pixel_ra'][...])[self.FeedIndex,selectFeature]
            x  = x[...,0:self.datasizes[i]].flatten()
            y  = (d['level1/spectrometer/pixel_pointing/pixel_dec'][...])[self.FeedIndex,selectFeature]
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
        for j, feed in enumerate(self.FeedIndex):

            bad = np.isnan(tod[j,:])
            if all(bad):
                continue
            tod[j,bad] = np.interp(t[bad], t[~bad], tod[j,~bad])
            pmdl = np.poly1d(np.polyfit(1./np.sin(el[j,:]*np.pi/180.), tod[j,:],1))
            tod[j,:] -= pmdl(1./np.sin(el[j,:]*np.pi/180.))
            tod[j,:] -= np.nanmedian(tod[j,:])


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
        for j, feed in enumerate(self.FeedIndex):
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

class DataWithOffsets(DataLevel2):
    def __init__(self, parameters,feeds=[1], bands=[0], frequencies=[0],keeptod=False):
        
        self.nmodes = 5
        # -- constants -- a lot of these are COMAP specific
        self.ifeature = 5
        self.chunks = []
        self.datasizes = []
        self.Nsamples = 0
        self.Nhorns = 0
        self.Nbands = 4

        #self.band = int(band)
        #self.frequency = int(frequency)

        self.keeptod = keeptod

        self.offsetLen = parameters['Destriper']['offset']

        self.Feeds  = feeds
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

        self.naive    = {(feed,band,frequency):Map(self.nxpix,self.nypix, self.wcs) \
                         for band in bands for frequency in frequencies for feed in self.Feeds}
        self.hits     = {(feed,band,frequency):Map(self.nxpix,self.nypix, self.wcs) \
                         for band in bands for frequency in frequencies for feed in self.Feeds}
        self.residual = {(feed,band,frequency):Offsets(self.offsetLen, Noffsets, self.Nsamples) \
                         for band in bands for frequency in frequencies for feed in self.Feeds}

        # output = h5py.File('maps/DestripedMaps.hd5','a')
        # grp = output.create_group('50samples')
        # key = (self.Feeds[0], frequencies[0], bands[0])
        # dset = grp.create_dataset('maps',(len(filelist), 19, 4, 2, self.nxpix, self.nypix))
        # wei  = grp.create_dataset('weights',(len(filelist), 19, 4, 2, self.nxpix, self.nypix))
        # dset.attrs['CDELT'] = self.naive[key].wcs.wcs.cdelt
        # dset.attrs['CRVAL'] = self.naive[key].wcs.wcs.crval
        # dset.attrs['CRPIX'] = self.naive[key].wcs.wcs.crpix
        # dset.attrs['CTYPE'] = [v.encode('utf-8') for v in self.naive[key].wcs.wcs.ctype]
        for i, filename in enumerate(tqdm(filelist)):
            d = h5py.File(filename,'r')
            # --- Feed position indices can change
            self.Feeds = [1]
            self.FeedIndex = self.GetFeeds(d['level1/spectrometer/feeds'][...], self.Feeds)

            for (feedindex,feed) in zip(self.FeedIndex, self.Feeds):
                for band in [0]:#bands:
                    for frequency in tqdm([0]):#frequencies):
                        self.naive[(feed,band,frequency)].clearmaps()
                        self.residual[(feed,band,frequency)].clear()
                        self.hits[(feed,band,frequency)].clearmaps()

                        self.readData(i,d,feedindex,feed,band, frequency)
                        self.naive[(feed,band,frequency)].average()
                        self.residual[(feed,band,frequency)].accumulate(-self.naive[(feed,band,frequency)].output[self.pixels],
                                                                        self.allweights,
                                                                        [0,self.pixels.size])
                        self.residual[(feed,band,frequency)].average()
                        #img = self.naive[(feed,band,frequency)]()
                        #img = removeplane(img)
                        #weights = self.naive[(feed,band,frequency)].wei
                        #dset[i,int(feed-1),band,frequency,:,:] = img
                        #wei[i,int(feed-1),band,frequency,:,:]  = np.reshape(weights,img.shape)

            d.close()
        #output.close()


    def readData(self, i, d, feedindex, feed,band, frequency):
        """
        Reads data
        """

        #gdir = '/scratch/nas_comap1/sharper/COMAP/runcomapreduce/AncillaryData/SourceFitsAllFeeds/'
        #calfile = 'CalFactors_Nov2019_Jupiter.dat'
        #caldata = np.loadtxt('{}/{}'.format(gdir,calfile))
    

        # -- Only want to look at the observation data
        features = d['level1/spectrometer/features'][:]
        selectFeature = self.featureBits(features.astype(float), self.ifeature)
        features = features[selectFeature]


        self.pixels[self.chunks[i][0]:self.chunks[i][1]] = self.skyPixels(i, d,
                                                                          feedindex,
                                                                          selectFeature,
                                                                          self.naive[(feed,band,frequency)])
        el  = (d['level1/spectrometer/pixel_pointing/pixel_el'][...])[feedindex,selectFeature]
        el  = el[0:self.datasizes[i]]
        az  = (d['level1/spectrometer/pixel_pointing/pixel_az'][...])[feedindex,selectFeature]
        az  = az[0:self.datasizes[i]]

        # Read in data and remove offsets
        tod = d['level2/averaged_tod'][feedindex,band,frequency,:]
        offsets = d['level2/offsets'][feedindex,band,frequency,:]

        
        #tod -= offsets # remove offsets!
        #nFeeds, nBands, nChannels, nSamples = tod.shape

        # Select data range
        tod = tod[selectFeature]
        tod = tod[0:self.datasizes[i]]
        offsets = offsets[selectFeature]
        offsets = offsets[0:self.datasizes[i]]

        
        weights = np.ones(tod.shape)
        t = np.arange(tod.size)

        bad = np.isnan(tod)
        if all(bad):
            return
        tod[bad] = np.interp(t[bad], t[~bad], tod[~bad])
        pmdl = np.poly1d(np.polyfit(1./np.sin(el*np.pi/180.), tod,1))
        tod -= pmdl(1./np.sin(el*np.pi/180.))
        tod -= np.nanmedian(tod)


        N = tod.size//2 * 2
        diffTOD = tod[:N:2]-tod[1:N:2]
        rms = np.sqrt(np.nanmedian(diffTOD**2)*1.4826)
        weights *= 1./rms**2

        # Remove spikes
        select = np.where((np.abs(diffTOD) > rms*5))[0]*2
        weights[select] *= 1e-10

        rms = np.sqrt(np.nanmedian(tod**2))*1.4826

        bad = (np.isnan(tod)) | (self.pixels[self.chunks[i][0]:self.chunks[i][1]] == -1)
        tod[bad] = 0
        weights[bad] = 0

        elc   =d['level1/hk/antenna0/driveNode/elCurrent'][:]
        elcmjd=d['level1/hk/antenna0/driveNode/utc'][:]
        mjd   =d['level1/spectrometer/MJD'][:]
        elcf  =np.interp(mjd, elcmjd, elc)

        select = np.where((elcf > 200))[0]
        dsel = select[1::2]-select[::2]
        select = select[np.where((dsel > 100))[0]]
        
        resid = tod[select[0]:select[1]] - offsets[select[0]:select[1]]
        ps = np.abs(np.fft.fft(resid))**2
        pso = np.abs(np.fft.fft(tod[select[0]:select[1]]))**2

        fnu= np.fft.fftfreq(ps.size, d=1./50.)

        pyplot.subplot(211)
        pyplot.plot(fnu[1:fnu.size//2], pso[1:ps.size//2]/ps.size*1e6, label='Original')
        pyplot.plot(fnu[1:fnu.size//2], ps[1:ps.size//2]/ps.size*1e6 , label='Destriped')

        
        pyplot.xlabel('Sample Frequency (Hz)')
        pyplot.ylabel(r' Power (mK$^2$)')
        pyplot.axvline(0.025,color='r',linestyle='--')
        pyplot.axvline(0.04 ,color='r',linestyle='--')
        pyplot.yscale('log')
        pyplot.xscale('log')
        pyplot.grid()
        pyplot.title('{} {} {}'.format(d['level1/comap'].attrs['obsid'].decode('utf-8'),  
                                       d['level1/spectrometer/feeds'][feed], 
                                       d['level1/spectrometer/bands'][band].decode('utf-8')))
        pyplot.legend()
        pyplot.subplot(212)

        nbins = 6
        azEdges = np.linspace(np.min(az[select[0]:select[1]]),np.max(az[select[0]:select[1]]),nbins+1)
        azMids  = (azEdges[1:]+azEdges[:-1])/2.

        print(az.shape, resid.shape)
        sw = np.histogram(az[select[0]:select[1]],azEdges,weights=resid)[0]
        w  = np.histogram(az[select[0]:select[1]],azEdges)[0]
        m  = sw/w
        #pyplot.plot(azMids, w)
        tmin = t[select[0]]
        pyplot.plot(t[select[0]:select[1]]/50.-tmin/50.,resid)
        #pyplot.plot(azMids, m)
        pyplot.plot(t[select[0]:select[1]]/50.-tmin/50.,np.interp(az[select[0]:select[1]],azMids[np.isfinite(m)],m[np.isfinite(m)]))
        pyplot.grid()
        pyplot.xlim(0,120)
        pyplot.xlabel('Time (seconds)')
        pyplot.ylabel(r'$T_a$ (K)')
        pyplot.savefig('maps/plots/powerspecs/{}_{}_{}.png'.format(d['level1/comap'].attrs['obsid'].decode('utf-8'),
                                                                   d['level1/spectrometer/feeds'][feed], 
                                                                   d['level1/spectrometer/bands'][band].decode('utf-8')))

        pyplot.show()


        if self.keeptod:
            self.todall[self.chunks[i][0]:self.chunks[i][1]] = tod*1.

        
        # Now accumulate the data!
        self.naive[(feed,band,frequency)].accumulate(tod,weights,self.pixels[self.chunks[i][0]:self.chunks[i][1]])
        self.hits[(feed,band,frequency)].accumulatehits(self.pixels[self.chunks[i][0]:self.chunks[i][1]])
        self.residual[(feed,band,frequency)].accumulate(tod,weights,self.chunks[i])

    def skyPixels(self,i, d,feedindex, selectFeature, naive):
        """
        Returns the pixel coordinates in the WCS frame
        """

        # We store all the pointing information
        x  = d['level1/spectrometer/pixel_pointing/pixel_ra'][feedindex,selectFeature]
        x  = x[0:self.datasizes[i]].flatten()
        y  = d['level1/spectrometer/pixel_pointing/pixel_dec'][feedindex,selectFeature]
        y  = y[0:self.datasizes[i]].flatten()

        pixels = self.getFlatPixels(x,y)
        pixels[pixels < 0] = -1
        pixels[pixels > naive.npix] = -1

        return pixels

    def countDataSize(self,filename):
        """
        Get size of data for this file
        """
        
        d = h5py.File(filename,'r')
        features = d['level1/spectrometer/features'][:]
        selectFeature = self.featureBits(features.astype(float), self.ifeature)
        N = len(features[selectFeature])
        d.close()

        N = (N//self.offsetLen) * self.offsetLen
        #N = N#*self.Nfeeds

        self.chunks += [[int(self.Nsamples), int(self.Nsamples+N)]]
        self.datasizes += [int(N)]
        self.Nsamples += int(N)

#############
# -----------
#############
class HealpixDataWithOffsets(DataLevel2):
    def __init__(self, parameters,feeds=[1], bands=[0], frequencies=[0],keeptod=False):
        
        self.nmodes = 5
        # -- constants -- a lot of these are COMAP specific
        self.ifeature = 5
        self.chunks = []
        self.datasizes = []
        self.Nsamples = 0
        self.Nhorns = 0
        self.Nbands = 4

        #self.band = int(band)
        #self.frequency = int(frequency)

        self.keeptod = keeptod

        self.offsetLen = parameters['Destriper']['offset']

        self.Feeds  = feeds
        try:
            self.Nfeeds = len(parameters['Inputs']['feeds'])
            self.Feeds = [int(f) for f in self.Feeds]
        except TypeError:
            self.Feeds = [int(self.Feeds)]
            self.Nfeeds = 1


        # Setup the map coordinates -- Needed for generating pixel coordinates
        self.nside = 4096

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

        self.naive    = HealpixMap(12*self.nside**2)

        print(filelist)
        for i, filename in enumerate(tqdm(filelist)):
            d = h5py.File(filename,'r')
            self.readData(i,d)
            d.close()
        #output.close()


    def readData(self, i, d):
        """
        Reads data
        """    

        # -- Only want to look at the observation data
        features = d['level1/spectrometer/features'][:]
        feeds = d['level1/spectrometer/feeds'][:]

        selectFeature = self.featureBits(features.astype(float), self.ifeature)
        features = features[selectFeature]

        rot = hp.rotator.Rotator(coord=['C','G'])
        ra = d['level1/spectrometer/pixel_pointing/pixel_ra'][...]
        dec= d['level1/spectrometer/pixel_pointing/pixel_dec'][...]
        gb, gl = rot((90-dec.flatten())*np.pi/180., ra.flatten()*np.pi/180.)
        gb, gl = np.reshape(gb,ra.shape), np.reshape(gl, dec.shape)
        gb, gl = gb[:,selectFeature], gl[:,selectFeature]
        gb, gl = gb[:,:self.datasizes[i]], gl[:,:self.datasizes[i]]
        #gb, gl = (np.pi/2.-gb)*180./np.pi, gl*180./np.pi
        ra, dec = ra[:,selectFeature], dec[:,selectFeature]
        ra, dec = ra[:,:self.datasizes[i]], dec[:,:self.datasizes[i]]

        el  = (d['level1/spectrometer/pixel_pointing/pixel_el'][...])[:,selectFeature]
        el  = el[:,0:self.datasizes[i]]
        az  = (d['level1/spectrometer/pixel_pointing/pixel_az'][...])[:,selectFeature]
        az  = az[:,0:self.datasizes[i]]

        for feedindex, feed in enumerate(tqdm(feeds)):
            if any([feed == z for z in [3,4,5,6,7,8,9,12,13,16,17,18,19,20]]):
                continue
            # if feed != 11:
            #     continue
            # if feed == 8:
            #     continue
            # if feed == 12:
            #     continue
            # if feed == 20:
            #     continue
            pixels = hp.ang2pix(self.nside,gb,gl)

            # Read in data and remove offsets
            tod = d['level2/averaged_tod'][feedindex,:,:,:]
            offsets = d['level2/offsets'][feedindex,:,:,:]
            tod = tod[:,:,selectFeature]
            offsets = offsets[:,:,selectFeature]

            tod = tod[:,:,:self.datasizes[i]]
            offsets = offsets[:,:,:self.datasizes[i]]

            t = np.arange(self.datasizes[i])
            weights = np.zeros(self.datasizes[i])
 
            for band in tqdm(range(4)):#tod.shape[0])):
                for channel in tqdm(range(1,tod.shape[1]-1)):
                    bad = np.isnan(tod[band,channel,:])
                    
                    if any(bad):
                        tod[band,channel,bad] = np.interp(t[bad], t[~bad], tod[band,channel,~bad])

                    tod[band,channel,:] -= np.nanmedian(tod[band,channel,:])
                    offsets[band,channel,:] = butt_lowpass(offsets[band,channel,:],1, 50.)

                    tod[band,channel,:] -= offsets[band,channel,:]

                    pmdl = np.poly1d(np.polyfit(1./np.sin(el[feedindex,:]*np.pi/180.), tod[band,channel,:],1))
                    tod[band,channel,:] -= pmdl(1./np.sin(el[feedindex,:]*np.pi/180.))
                    pmdl = np.poly1d(np.polyfit(az[feedindex,:], tod[band,channel,:],1))
                    tod[band,channel,:] -= pmdl(az[feedindex,:])


                    #pyplot.plot(tod[band,channel,:])
                    stepsize = 8000
                    nsteps = tod.shape[-1]//stepsize
                    tod[band,channel,:] = butt_highpass(tod[band,channel,:],1/180., 50.)
                    for k in range(nsteps):
                        lo = k*stepsize
                        hi = (k+1)*stepsize
                        if k == (nsteps-1):
                            hi = tod.shape[-1]
                        pmdl = np.poly1d(np.polyfit(dec[feedindex,lo:hi], tod[band,channel,lo:hi],1))
                        tod[band,channel,lo:hi] -= pmdl(dec[feedindex,lo:hi])
                        pmdl = np.poly1d(np.polyfit(ra[feedindex,lo:hi], tod[band,channel,lo:hi],1))
                        tod[band,channel,lo:hi] -= pmdl(ra[feedindex,lo:hi])


                    # pyplot.plot(tod[band,channel,:])
                    # pyplot.show()
                    # ps1 = np.abs(np.fft.fft(tod[band,channel,:]))**2
                    # nu  = np.fft.fftfreq(tod.shape[-1],d=1/50)
                    # #tod[band,channel,:] = butt_bandpass(tod[band,channel,:],np.array([0.1,0.12]), 50.)
                    # tod[band,channel,:] = butt_bandpass(tod[band,channel,:],np.array([0.05,0.08]), 50.)
                    # ps2 = np.abs(np.fft.fft(offsets[band,channel,:]))**2
                    # pyplot.plot(nu[1:nu.size//2], ps1[1:nu.size//2])
                    # pyplot.plot(nu[1:nu.size//2], ps2[1:nu.size//2])
                    # pyplot.yscale('log')
                    # pyplot.xscale('log')
                    # pyplot.grid()
                    # pyplot.show()
                    # pyplot.plot(tod[band,channel,:])
                    # pyplot.show()
                    N = tod.size//2 * 2
                    diffTOD = tod[:N:2]-tod[1:N:2]
                    rms = np.sqrt(np.nanmedian(diffTOD**2)*1.4826)
                    weights[:] = 1./rms**2

                    # Remove spikes
                    #select = np.where((np.abs(diffTOD) > rms*5))[0]
                    #weights[select] *= 1e-10

                    self.naive.accumulate(tod[band,channel,:].astype(float),weights,pixels[feedindex,:])



#############
# -----------
#############

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

        self.allweights[self.chunks[i][0]:self.chunks[i][1]] = weights

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
    def weights(self):
        return np.reshape(self.wei, (self.nypix, self.nxpix))

class HealpixMap(Map):
    """
    Stores pixel information
    """
    def __init__(self,npix,storehits=False):

        self.storehits = storehits
        # --- Need to create several arrays:
        # 1) Main output map array
        # 2) Signal*Weights array
        # 3) Weights array
        # 4) Hits

        self.npix = npix
        self.output = np.zeros(self.npix)
        self.sigwei = np.zeros(self.npix)
        self.wei    = np.zeros(self.npix)
        if self.storehits:
            self.hits = np.zeros(self.npix)

    def __call__(self):
        self.average()
        return self.output

    def weights(self):
        return self.wei

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


    def clear(self):
        self.offsets *= 0
        self.sigwei *= 0
        self.wei *= 0

    def accumulate(self,tod,weights,chunk):
        """
        Add more data to residual offset
        """
        binFuncs.binValues(self.sigwei, self.offsetpixels[chunk[0]:chunk[1]], weights=tod*weights )
        binFuncs.binValues(self.wei   , self.offsetpixels[chunk[0]:chunk[1]], weights=weights    )


    def average(self):
        self.goodpix = np.where((self.wei != 0 ))[0]
        self.offsets[self.goodpix] = self.sigwei[self.goodpix]/self.wei[self.goodpix]
