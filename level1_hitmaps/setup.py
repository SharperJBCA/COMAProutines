from numpy.distutils.core import setup
from numpy.distutils.core import Extension
from numpy import get_include
from Cython.Build import cythonize
import os

binFuncs = Extension(name='binFuncs',
                     include_dirs=[get_include()],
                     sources=['binFuncs.pyx'])
extensions = [binFuncs]
setup(ext_modules=cythonize(extensions))
