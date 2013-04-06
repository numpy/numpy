#!/usr/bin/env python
"""
WARNING: this code is deprecated and slated for removal soon.  See the
doc/cython directory for the replacement, which uses Cython (the actively
maintained version of Pyrex).


Install file for example on how to use Pyrex with Numpy.

For more details, see:
http://www.scipy.org/Cookbook/Pyrex_and_NumPy
http://www.scipy.org/Cookbook/ArrayStruct_and_Pyrex

"""
from __future__ import division, print_function

from distutils.core import setup
from distutils.extension import Extension

# Make this usable by people who don't have pyrex installed (I've committed
# the generated C sources to SVN).
try:
    from Pyrex.Distutils import build_ext
    has_pyrex = True
except ImportError:
    has_pyrex = False

import numpy

# Define a pyrex-based extension module, using the generated sources if pyrex
# is not available.
if has_pyrex:
    pyx_sources = ['numpyx.pyx']
    cmdclass    = {'build_ext': build_ext}
else:
    pyx_sources = ['numpyx.c']
    cmdclass    = {}


pyx_ext = Extension('numpyx',
                 pyx_sources,
                 include_dirs = [numpy.get_include()])

# Call the routine which does the real work
setup(name        = 'numpyx',
      description = 'Small example on using Pyrex to write a Numpy extension',
      url         = 'http://www.scipy.org/Cookbook/Pyrex_and_NumPy',
      ext_modules = [pyx_ext],
      cmdclass    = cmdclass,
      )
