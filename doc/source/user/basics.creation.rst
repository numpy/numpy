.. _arrays.creation:

**************
Array creation
**************

.. seealso:: :ref:`Array creation routines <routines.array-creation>`

Introduction
============

There are 6 general mechanisms for creating arrays:

1) Conversion from other Python structures (e.g., lists, tuples)
2) Intrinsic NumPy array creation objects (e.g., arange, ones, zeros,
   etc.)
3) Replicating, joining, or mutating existing arrays
4) Reading arrays from disk, either from standard or custom formats
5) Creating arrays from raw bytes through the use of strings or buffers
6) Use of special library functions (e.g., random)

You can use these methods to create object or structured arrays, but
this application is outside the scope of this document. 

Converting Python structures to NumPy Arrays
====================================================

Python structures such as lists, tuples, and dictionaries. Lists and
tuples are defined using ``[...]`` and ``(...)``, respectively. Lists
and tuples lend themselves to ND-array creation:
* a list of numbers will create a 1D array, 
* a list of lists will create a 2D array, 
* and further nested lists will create ND arrays

::
a1D = np.array([1,2,3])
a2D = np.array([ [1, 2], [3, 4] ])
a3D = np.array([ [ [1, 2], [3, 4]],
                 [ [5, 6], [7, 8]]
               ])

lists

tuples

dictionary [what is a Python object?]

When you use :ref:`array` to define a new array, its important to
consider the `dtype` of the elements in the array. 

NumPy arrays enable you to specify the ``dtype`` for elements in the
array. This feature gives you more control over the underlying data
structures and how the elements are handled in C/C++ functions. If you
are not careful with ``dtype`` assignments, you can get unwanted
overflow, as such 

::
  >>> a = np.array([127,128,129], dtype = int8)
  >>> print(a)
  [ 127 -128 -127 ]

An 8-bit signed integer can represent integers from -128 to 127.
Assigning the ``int8`` array to integers outside of this range results
in overflow. This feature can often be misunderstood if you use it to
perform calculations without matching ``dtypes``, for example

::
    >>> a = np.array([2, 3, 4], dtype = uint32)
    >>> b = np.array([5, 6, 7], dtype = uint32)
    >>> c_unsigned32 = a - b
    >>> print('unsigned c:', c_unsigned32, c_unsigned32.dtype)
    unsigned c: [4294967293 4294967293 4294967293] uint32
    >>> c_signed32 = a - b.astype(int32)
    >>> print('signed c:', c_signed32, c_signed32.dtype)
    signed c: [-3 -3 -3] int64

Notice when you perform operations with two arrays of the same
``dtype``, the resulting array maintains the ``uint32`` type. When you
perform operations with different ``dtype`` arrays, NumPy will try to
assign a new type that satisfies all of the array elements involved in
the computation, here ``unit32`` and ``int32`` can both be represented in
the ``int64`` ``dtype``

In general, numerical data arranged in an array-like structure in Python can
be converted to arrays through the use of the array() function. The most
obvious examples are lists and tuples. See the documentation for array() for
details for its use. Some objects may support the array-protocol and allow
conversion to arrays this way. A simple way to find out if the object can be
converted to a NumPy array using array() is simply to try it interactively and
see if it works! (The Python Way).

Examples: ::

 >>> x = np.array([2,3,1,0])
 >>> x = np.array([2, 3, 1, 0])
 >>> x = np.array([[1,2.0],[0,0],(1+1j,3.)]) # note mix of tuple and lists,
     and types
 >>> x = np.array([[ 1.+0.j, 2.+0.j], [ 0.+0.j, 0.+0.j], [ 1.+1.j, 3.+0.j]])

Intrinsic NumPy array creation objects
======================================

NumPy has built-in functions for creating arrays from scratch as laid
out in the :ref:`Array creation routines <routines.array-creation>`.
When creating NumPy 

zeros(shape) will create an array filled with 0 values with the specified
shape. The default dtype is float64. ::

 >>> np.zeros((2, 3))
 array([[ 0., 0., 0.], [ 0., 0., 0.]])

ones(shape) will create an array filled with 1 values. It is identical to
zeros in all other respects.

arange() will create arrays with regularly incrementing values. Check the
docstring for complete information on the various ways it can be used. A few
examples will be given here: ::

 >>> np.arange(10)
 array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
 >>> np.arange(2, 10, dtype=float)
 array([ 2., 3., 4., 5., 6., 7., 8., 9.])
 >>> np.arange(2, 3, 0.1)
 array([ 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9])

Note that there are some subtleties regarding the last usage that the user
should be aware of that are described in the arange docstring.

linspace() will create arrays with a specified number of elements, and
spaced equally between the specified beginning and end values. For
example: ::

 >>> np.linspace(1., 4., 6)
 array([ 1. ,  1.6,  2.2,  2.8,  3.4,  4. ])

The advantage of this creation function is that one can guarantee the
number of elements and the starting and end point, which arange()
generally will not do for arbitrary start, stop, and step values.

indices() will create a set of arrays (stacked as a one-higher dimensioned
array), one per dimension with each representing variation in that dimension.
An example illustrates much better than a verbal description: ::

 >>> np.indices((3,3))
 array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]])

This is particularly useful for evaluating functions of multiple dimensions on
a regular grid.

Reading Arrays From Disk
========================

This is presumably the most common case of large array creation. The details,
of course, depend greatly on the format of data on disk and so this section
can only give general pointers on how to handle various formats.

Standard Binary Formats
-----------------------

Various fields have standard formats for array data. The following lists the
ones with known python libraries to read them and return NumPy arrays (there
may be others for which it is possible to read and convert to NumPy arrays so
check the last section as well)
::

 HDF5: h5py
 FITS: Astropy

Examples of formats that cannot be read directly but for which it is not hard to
convert are those formats supported by libraries like PIL (able to read and
write many image formats such as jpg, png, etc).

Common ASCII Formats
------------------------

Comma Separated Value files (CSV) are widely used (and an export and import
option for programs like Excel). There are a number of ways of reading these
files in Python. There are CSV functions in Python and functions in pylab
(part of matplotlib).

More generic ascii files can be read using the io package in scipy.

Custom Binary Formats
---------------------

There are a variety of approaches one can use. If the file has a relatively
simple format then one can write a simple I/O library and use the NumPy
fromfile() function and .tofile() method to read and write NumPy arrays
directly (mind your byteorder though!) If a good C or C++ library exists that
read the data, one can wrap that library with a variety of techniques though
that certainly is much more work and requires significantly more advanced
knowledge to interface with C or C++.

Use of Special Libraries
------------------------

There are libraries that can be used to generate arrays for special purposes
and it isn't possible to enumerate all of them. The most common uses are use
of the many array generation functions in random that can generate arrays of
random values, and some utility functions to generate special matrices (e.g.
diagonal).


