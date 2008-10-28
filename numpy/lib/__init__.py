"""
Basic functions used by several sub-packages and
useful to have in the main name-space.

Type Handling
-------------
================ ===================
iscomplexobj     Test for complex object, scalar result
isrealobj        Test for real object, scalar result
iscomplex        Test for complex elements, array result
isreal           Test for real elements, array result
imag             Imaginary part
real             Real part
real_if_close    Turns complex number with tiny imaginary part to real
isneginf         Tests for negative infinity, array result
isposinf         Tests for positive infinity, array result
isnan            Tests for nans, array result
isinf            Tests for infinity, array result
isfinite         Tests for finite numbers, array result
isscalar         True if argument is a scalar
nan_to_num       Replaces NaN's with 0 and infinities with large numbers
cast             Dictionary of functions to force cast to each type
common_type      Determine the minimum common type code for a group
                 of arrays
mintypecode      Return minimal allowed common typecode.
================ ===================

Index Tricks
------------
================ ===================
mgrid            Method which allows easy construction of N-d
                 'mesh-grids'
``r_``           Append and construct arrays: turns slice objects into
                 ranges and concatenates them, for 2d arrays appends rows.
index_exp        Konrad Hinsen's index_expression class instance which
                 can be useful for building complicated slicing syntax.
================ ===================

Useful Functions
----------------
================ ===================
select           Extension of where to multiple conditions and choices
extract          Extract 1d array from flattened array according to mask
insert           Insert 1d array of values into Nd array according to mask
linspace         Evenly spaced samples in linear space
logspace         Evenly spaced samples in logarithmic space
fix              Round x to nearest integer towards zero
mod              Modulo mod(x,y) = x % y except keeps sign of y
amax             Array maximum along axis
amin             Array minimum along axis
ptp              Array max-min along axis
cumsum           Cumulative sum along axis
prod             Product of elements along axis
cumprod          Cumluative product along axis
diff             Discrete differences along axis
angle            Returns angle of complex argument
unwrap           Unwrap phase along given axis (1-d algorithm)
sort_complex     Sort a complex-array (based on real, then imaginary)
trim_zeros       Trim the leading and trailing zeros from 1D array.
vectorize        A class that wraps a Python function taking scalar
                 arguments into a generalized function which can handle
                 arrays of arguments using the broadcast rules of
                 numerix Python.
================ ===================

Shape Manipulation
------------------
================ ===================
squeeze          Return a with length-one dimensions removed.
atleast_1d       Force arrays to be > 1D
atleast_2d       Force arrays to be > 2D
atleast_3d       Force arrays to be > 3D
vstack           Stack arrays vertically (row on row)
hstack           Stack arrays horizontally (column on column)
column_stack     Stack 1D arrays as columns into 2D array
dstack           Stack arrays depthwise (along third dimension)
split            Divide array into a list of sub-arrays
hsplit           Split into columns
vsplit           Split into rows
dsplit           Split along third dimension
================ ===================

Matrix (2D Array) Manipulations
-------------------------------
================ ===================
fliplr           2D array with columns flipped
flipud           2D array with rows flipped
rot90            Rotate a 2D array a multiple of 90 degrees
eye              Return a 2D array with ones down a given diagonal
diag             Construct a 2D array from a vector, or return a given
                 diagonal from a 2D array.
mat              Construct a Matrix
bmat             Build a Matrix from blocks
================ ===================

Polynomials
-----------
================ ===================
poly1d           A one-dimensional polynomial class
poly             Return polynomial coefficients from roots
roots            Find roots of polynomial given coefficients
polyint          Integrate polynomial
polyder          Differentiate polynomial
polyadd          Add polynomials
polysub          Substract polynomials
polymul          Multiply polynomials
polydiv          Divide polynomials
polyval          Evaluate polynomial at given argument
================ ===================

Import Tricks
-------------
================ ===================
ppimport         Postpone module import until trying to use it
ppimport_attr    Postpone module import until trying to use its attribute
ppresolve        Import postponed module and return it.
================ ===================

Machine Arithmetics
-------------------
================ ===================
machar_single    Single precision floating point arithmetic parameters
machar_double    Double precision floating point arithmetic parameters
================ ===================

Threading Tricks
----------------
================ ===================
ParallelExec     Execute commands in parallel thread.
================ ===================

1D Array Set Operations
-----------------------
Set operations for 1D numeric arrays based on sort() function.

================ ===================
ediff1d          Array difference (auxiliary function).
unique1d         Unique elements of 1D array.
intersect1d      Intersection of 1D arrays with unique elements.
intersect1d_nu   Intersection of 1D arrays with any elements.
setxor1d         Set exclusive-or of 1D arrays with unique elements.
setmember1d      Return an array of shape of ar1 containing 1 where
                 the elements of ar1 are in ar2 and 0 otherwise.
union1d          Union of 1D arrays with unique elements.
setdiff1d        Set difference of 1D arrays with unique elements.
================ ===================

"""
from info import __doc__
from numpy.version import version as __version__

from type_check import *
from index_tricks import *
from function_base import *
from shape_base import *
from stride_tricks import *
from twodim_base import *
from ufunclike import *

import scimath as emath
from polynomial import *
from machar import *
from getlimits import *
#import convertcode
from utils import *
from arraysetops import *
from io import *
from financial import *
import math
from arrayterator import *

__all__ = ['emath','math']
__all__ += type_check.__all__
__all__ += index_tricks.__all__
__all__ += function_base.__all__
__all__ += shape_base.__all__
__all__ += stride_tricks.__all__
__all__ += twodim_base.__all__
__all__ += ufunclike.__all__
__all__ += polynomial.__all__
__all__ += machar.__all__
__all__ += getlimits.__all__
__all__ += utils.__all__
__all__ += arraysetops.__all__
__all__ += io.__all__
__all__ += financial.__all__

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
