""" Basic functions used by several sub-packages and useful to have in the
main name-space

Type handling
==============
iscomplexobj     --  Test for complex object, scalar result
isrealobj        --  Test for real object, scalar result
iscomplex        --  Test for complex elements, array result
isreal           --  Test for real elements, array result
imag             --  Imaginary part
real             --  Real part
real_if_close    --  Turns complex number with tiny imaginary part to real
isneginf         --  Tests for negative infinity ---|
isposinf         --  Tests for positive infinity    |
isnan            --  Tests for nans                 |----  array results
isinf            --  Tests for infinity             |
isfinite         --  Tests for finite numbers    ---| 
isscalar         --  True if argument is a scalar
nan_to_num       --  Replaces NaN's with 0 and infinities with large numbers
typename         --  Return english name for given typecode character
cast             --  Dictionary of functions to force cast to each type
common_type      --  Determine the 'minimum common type code' for a group
                       of arrays

Index tricks
==================
mgrid            --  Method which allows easy construction of N-d 'mesh-grids'
r_               --  Append and construct arrays -- turns slice objects into
                       ranges and concatenates them, for 2d arrays appends
                       rows.
c_               --  Append and construct arrays -- for 2d arrays appends
                       columns.

index_exp        --  Konrad Hinsen's index_expression class instance which
                     can be useful for building complicated slicing syntax.

Useful functions
==================
select           --  Extension of where to multiple conditions and choices
extract          --  Extract 1d array from flattened array according to mask
insert           --  Insert 1d array of values into Nd array according to mask
linspace         --  Evenly spaced samples in linear space
logspace         --  Evenly spaced samples in logarithmic space
fix              --  Round x to nearest integer towards zero
mod              --  Modulo mod(x,y) = x % y except keeps sign of y
amax             --  Array maximum along axis
amin             --  Array minimum along axis
ptp              --  Array max-min along axis
cumsum           --  Cumulative sum along axis
prod             --  Product of elements along axis
cumprod          --  Cumluative product along axis
diff             --  Discrete differences along axis
angle            --  Returns angle of complex argument
unwrap           --  Unwrap phase along given axis (1-d algorithm)
sort_complex     --  Sort a complex-array (based on real, then imaginary)
trim_zeros       --  trim the leading and trailing zeros from 1D array.

vectorize        -- a class that wraps a Python function taking scalar
                         arguments into a generalized function which
                         can handle arrays of arguments using the broadcast
                         rules of Numeric Python.

Shape manipulation
===================
squeeze          --  Return a with length-one dimensions removed.
atleast_1d       --  Force arrays to be > 1D
atleast_2d       --  Force arrays to be > 2D
atleast_3d       --  Force arrays to be > 3D
vstack           --  Stack arrays vertically (row on row)
hstack           --  Stack arrays horizontally (column on column)
column_stack     --  Stack 1D arrays as columns into 2D array
dstack           --  Stack arrays depthwise (along third dimension)
split            --  Divide array into a list of sub-arrays
hsplit           --  Split into columns
vsplit           --  Split into rows
dsplit           --  Split along third dimension

Matrix (2d array) manipluations
===============================
fliplr           --  2D array with columns flipped
flipud           --  2D array with rows flipped
rot90            --  Rotate a 2D array a multiple of 90 degrees
eye              --  Return a 2D array with ones down a given diagonal
diag             --  Construct a 2D array from a vector, or return a given
                       diagonal from a 2D array.                       
mat              --  Construct a Matrix

Polynomials
============
poly1d           --  A one-dimensional polynomial class

poly             --  Return polynomial coefficients from roots
roots            --  Find roots of polynomial given coefficients
polyint          --  Integrate polynomial
polyder          --  Differentiate polynomial
polyadd          --  Add polynomials
polysub          --  Substract polynomials
polymul          --  Multiply polynomials
polydiv          --  Divide polynomials
polyval          --  Evaluate polynomial at given argument

General functions
=================
vectorize -- Generalized Function class

Import tricks
=============
ppimport         --  Postpone module import until trying to use it
ppimport_attr    --  Postpone module import until trying to use its
                      attribute

Machine arithmetics
===================
machar_single    --  MachAr instance storing the parameters of system
                     single precision floating point arithmetics
machar_double    --  MachAr instance storing the parameters of system
                     double precision floating point arithmetics

Threading tricks
================
ParallelExec     --  Execute commands in parallel thread.
"""

standalone = 1
