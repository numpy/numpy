#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# _Series extension module
_Series = Extension("_Series",
                    ["Series_wrap.cxx",
                     "series.cxx"],
                    include_dirs = [numpy.get_numpy_include()],
#                    libraries = ["m"]
                    )

# Series setup
setup(name        = "Series",
      description = "Functions that work on series",
      author      = "Bill Spotz",
      py_modules  = ["Series"],
      ext_modules = [_Series]
      )
