#!/usr/bin/env python

from distutils.core import setup

setup (name = "scipy_install",
       version = "0.1",
       description = "Changes to distutils needed for scipy -- mostly Fortran support",
       author = "SciPy",
       licence = "BSD Style",
       url = 'http://www.scipy.org',
       packages = ['scipy_distutils','scipy_distutils.command'],
       package_dir = {'scipy_distutils':'.'}
       )


