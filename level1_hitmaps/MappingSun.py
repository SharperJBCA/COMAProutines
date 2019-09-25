import numpy as np
from astropy import wcs
from matplotlib import pyplot
import h5py
import binFuncs
from scipy.interpolate import interp1d
import os

from matplotlib.patches import Ellipse

from Mapping import Mapper,NormaliseFilter,AtmosphereFilter
from Utilities import Source
from comancpipeline.Tools import Coordinates


class MapperSun(Mapper):        
                        
    def setLevel1(self, datafile, source =''):
        """
        """
        self.setSource(source)

        self.teleLon = self.datafile['hk/antenna0/tracker/siteActual'][0,0]/(60.**2 * 1000.)
        self.teleLat = self.datafile['hk/antenna0/tracker/siteActual'][0,1]/(60.**2 * 1000.)
        
        self.datafile = datafile

        self.attributes = self.datafile['comap'].attrs

        self.tsamp = float(self.attributes['tsamp'].decode())
        self.obsid = self.attributes['obsid'].decode()
        self.source = self.attributes['source'].decode()
        
        # load but do not read yet.
        self.x = self.datafile['spectrometer/pixel_pointing/pixel_ra']
        self.y = self.datafile['spectrometer/pixel_pointing/pixel_dec']
        self.utc = self.datafile['spectrometer/MJD']
        sunra, sundec, sundist = Coordinates.getPlanetPosition('Sun', self.teleLon, self.teleLat, self.utc[:], returnall=True)
        sunra, sundec = Coordinates.precess(sunra, sundec, self.utc[:])
        pa = Coordinates.pa(sunra,sundec, self.utc, self.teleLon, self.teleLat)
        for i in range(self.x.shape[0]):
            self.x[i,:], self.y[i,:] = Coordinates.Rotate(self.x[i,:], self.y[i,:],sunra, sundec, -pa)

        self.xCoordinateName = r'$\Delta$A'
        self.yCoordinateName = r'$\Delta$E'

        self.el = self.datafile['spectrometer/pixel_pointing/pixel_el']

        
        self.tod_bavg = self.datafile['spectrometer/band_average']
        self.features = self.datafile['spectrometer/features'][:]
        self.mask = np.ones(self.features.size).astype(int)
        self.mask[self.featureBits(self.features.astype(float), 13)] = 0
        self.mask[self.features == 0] = 0
        self.mask = self.mask.astype(int)

        
        # If we don't spe
        self.setCrval()

        self.setWCS(self.crval, self.cdelt, self.crpix, self.ctype)
