#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# _Vector extension module
_Vector = Extension("_Vector",
                    ["Vector_wrap.cxx",
                     "vector.cxx"],
                    include_dirs = [numpy_include],
                    )

# _Matrix extension module
_Matrix = Extension("_Matrix",
                    ["Matrix_wrap.cxx",
                     "matrix.cxx"],
                    include_dirs = [numpy_include],
                    )

# NumyTypemapTests setup
setup(name        = "NumpyTypemapTests",
      description = "Functions that work on arrays",
      author      = "Bill Spotz",
      py_modules  = ["Vector", "Matrix"],
      ext_modules = [_Vector , _Matrix ]
      )
