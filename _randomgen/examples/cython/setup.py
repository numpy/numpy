# python setup.py build_ext -i
import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

setup(
    ext_modules=cythonize([Extension("extending",
                                     sources=['extending.pyx'],
                                     include_dirs=[np.get_include()])])
)
