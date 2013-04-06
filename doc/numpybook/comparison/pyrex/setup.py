#!/usr/bin/env python
from __future__ import division, print_function

from distutils.core import setup
from distutils.extension import Extension

import numpy

# Define a pyrex-based extension module, using the generated sources if pyrex
from Pyrex.Distutils import build_ext
pyx_sources = ['add.pyx']
cmdclass    = {'build_ext': build_ext}


pyx_ext = Extension('add',
                 pyx_sources,
                 include_dirs = [numpy.get_include()])

pyx_ext2 = Extension('blur',
                ['blur.pyx'],
                include_dirs = [numpy.get_include()])


# Call the routine which does the real work
setup(name        = 'add',
      description = 'Small example on using Pyrex to write a Numpy extension',
      url         = 'http://www.scipy.org/Cookbook/Pyrex_and_NumPy',
      ext_modules = [pyx_ext, pyx_ext2],
      cmdclass    = cmdclass,
      )
