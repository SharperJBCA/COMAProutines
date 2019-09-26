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


class MapperAzEl(Mapper):        
                        
    def setLevel1(self, datafile, source =''):
        """
        """
        self.setSource(source)
        
        self.datafile = datafile

        self.attributes = self.datafile['comap'].attrs

        self.tsamp = float(self.attributes['tsamp'].decode())
        self.obsid = self.attributes['obsid'].decode()
        self.source = self.attributes['source'].decode()
        
        # load but do not read yet.
        self.x = self.datafile['spectrometer/pixel_pointing/pixel_az']
        self.y = self.datafile['spectrometer/pixel_pointing/pixel_el']
        self.xCoordinateName = 'Azimuth'
        self.yCoordinateName = 'Elevation'

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
