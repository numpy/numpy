#! /usr/bin/env python
from __future__ import division, print_function

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.
numpy_include = numpy.get_include()

# Array extension module
_Array = Extension("_Array",
                   ["Array_wrap.cxx",
                    "Array1.cxx",
                    "Array2.cxx",
                    "ArrayZ.cxx"],
                   include_dirs = [numpy_include],
                   )

# Farray extension module
_Farray = Extension("_Farray",
                    ["Farray_wrap.cxx",
                     "Farray.cxx"],
                    include_dirs = [numpy_include],
                    )

# _Vector extension module
_Vector = Extension("_Vector",
                    ["Vector_wrap.cxx",
                     "Vector.cxx"],
                    include_dirs = [numpy_include],
                    )

# _Matrix extension module
_Matrix = Extension("_Matrix",
                    ["Matrix_wrap.cxx",
                     "Matrix.cxx"],
                    include_dirs = [numpy_include],
                    )

# _Tensor extension module
_Tensor = Extension("_Tensor",
                    ["Tensor_wrap.cxx",
                     "Tensor.cxx"],
                    include_dirs = [numpy_include],
                    )

_Fortran = Extension("_Fortran",
                    ["Fortran_wrap.cxx",
                     "Fortran.cxx"],
                    include_dirs = [numpy_include],
                    )

_Flat = Extension("_Flat",
                    ["Flat_wrap.cxx",
                     "Flat.cxx"],
                    include_dirs = [numpy_include],
                    )

# NumyTypemapTests setup
setup(name        = "NumpyTypemapTests",
      description = "Functions that work on arrays",
      author      = "Bill Spotz",
      py_modules  = ["Array", "Farray", "Vector", "Matrix", "Tensor",
                     "Fortran", "Flat"],
      ext_modules = [_Array, _Farray, _Vector, _Matrix, _Tensor,
                     _Fortran, _Flat]
      )
