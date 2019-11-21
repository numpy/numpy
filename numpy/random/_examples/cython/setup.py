#!/usr/bin/env python3
"""
Build the Cython demonstrations of low-level access to NumPy random

Usage: python setup.py build_ext -i
"""

import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
from os.path import join, abspath, dirname

path = abspath(dirname(__file__))

extending = Extension("extending",
                      sources=[join(path, 'extending.pyx')],
                      include_dirs=[
                            np.get_include(),
                            join(path, '..', '..')
                        ],
                      )
distributions = Extension("extending_distributions",
                          sources=[join(path, 'extending_distributions.pyx')],
                          include_dirs=[np.get_include()])

extensions = [extending, distributions]

setup(
    ext_modules=cythonize(extensions)
)
