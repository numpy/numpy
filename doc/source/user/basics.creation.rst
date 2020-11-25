.. _arrays.creation:

**************
Array creation
**************

.. seealso:: :ref:`Array creation routines <routines.array-creation>`

Introduction
============

There are 6 general mechanisms for creating arrays:

1) Conversion from other Python structures (i.e. lists and tuples)
2) Intrinsic NumPy array creation objects (e.g. arange, ones, zeros,
   etc.)
3) Replicating, joining, or mutating existing arrays
4) Reading arrays from disk, either from standard or custom formats
5) Creating arrays from raw bytes through the use of strings or buffers
6) Use of special library functions (e.g., random)

You can use these methods to create object or structured arrays, but
this application is outside the scope of this document. 

1) Converting Python structures to NumPy Arrays
===============================================

Python structures such as lists and tuples. Lists and
tuples are defined using ``[...]`` and ``(...)``, respectively. Lists
and tuples can define ND-array creation:
* a list of numbers will create a 1D array, 
* a list of lists will create a 2D array, 
* and further nested lists will create ND arrays

::
 a1D = array([1, 2, 3, 4])
 a2D = array([ [1, 2], [3, 4] ])
 a3D = array([ [ [1, 2], [3, 4]],
               [ [5, 6], [7, 8]]
             ])


When you use :ref:`array` to define a new array, its important to
consider the `dtype` of the elements in the array. 

NumPy arrays enable you to specify the ``dtype`` for elements in the
array. This feature gives you more control over the underlying data
structures and how the elements are handled in C/C++ functions. If you
are not careful with ``dtype`` assignments, you can get unwanted
overflow, as such 

::
  >>> a = array([127,128,129], dtype = int8)
  >>> print(a)
  [ 127 -128 -127 ]

An 8-bit signed integer can represent integers from -128 to 127.
Assigning the ``int8`` array to integers outside of this range results
in overflow. This feature can often be misunderstood if you use it to
perform calculations without matching ``dtypes``, for example

::
    >>> a = array([2, 3, 4], dtype = uint32)
    >>> b = array([5, 6, 7], dtype = uint32)
    >>> c_unsigned32 = a - b
    >>> print('unsigned c:', c_unsigned32, c_unsigned32.dtype)
    unsigned c: [4294967293 4294967293 4294967293] uint32
    >>> c_signed32 = a - b.astype(int32)
    >>> print('signed c:', c_signed32, c_signed32.dtype)
    signed c: [-3 -3 -3] int64

Notice when you perform operations with two arrays of the same
``dtype``: ``uint32``, the resulting array is the same type. When you
perform operations with different ``dtype``, NumPy will 
assign a new type that satisfies all of the array elements involved in
the computation, here ``uint32`` and ``int32`` can both be represented in
the ``int64``. 

The default NumPy behavior is to create arrays in either 64-bit integers
or double precision floating point numbers. If you expect your arrays to
be a certain type, then it is important to specify the ``dtype`` while
you create the array. 

2) Intrinsic NumPy array creation objects
=========================================

NumPy has over XX built-in functions for creating arrays as laid
out in the :ref:`Array creation routines <routines.array-creation>`.
NumPy array creation objects can be seen as:
1. 1D array creation objects
2. 2D array creation objects
3. ND array creation objects

1 - 1D array creation objects
-----------------------------

The 1D array creation objects e.g. ``linspace`` and ``arange`` typically
need at least two inputs: start and end.  

arange() will create arrays
with regularly incrementing values. Check the docstring for complete
information on the various ways it can be used. A few examples are
shown: ::

 >>> arange(10)
 array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
 >>> arange(2, 10, dtype=float)
 array([ 2., 3., 4., 5., 6., 7., 8., 9.])
 >>> arange(2, 3, 0.1)
 array([ 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9])

Note: there are some subtleties regarding the last usage that the user
should be aware of that are described in the arange docstring.

linspace() will create arrays with a specified number of elements, and
spaced equally between the specified beginning and end values. For
example: ::

 >>> linspace(1., 4., 6)
 array([ 1. ,  1.6,  2.2,  2.8,  3.4,  4. ])

The advantage of this creation function is that one can guarantee the
number of elements and the starting and end point, which arange()
generally will not do for arbitrary start, stop, and step values.

2 - 2D array creation objects
-----------------------------

The 2D array creation objects e.g. ``eye``, ``diag``, and ``vander``
define properties of special matrices represented as 2D arrays. 

``eye(n,m)`` defines a 2D identity matrix. The elements where i=j are 1
and the rest are 0, as such::

 >>> eye(3)
 array([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])
 >>> eye(3,5)
 array([[1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0.]])

``diag()`` can define either a square 2D array with given values along
the diagonal _or_ if given a 2D array returns a 1D array that is
only the diagonal elements. The two array creations can be helpful while
doing linear algebra as such::
 
 >>> diag([1,2,3])
 array([[1, 0, 0],
       [0, 2, 0],
       [0, 0, 3]])
 >>> diag([1,2,3],1)
 array([[0, 1, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 3],
        [0, 0, 0, 0]])
 >>> a = np.array([[1, 2], [3, 4]])
 >>> diag(a)
 array([1, 4])

``vander(x,n)`` defines a Vandermonde matrix as a 2D NumPy array. Each column
of the Vandermonde matrix is a decreasing power of the input 1D array or
list or tuple,
``x`` where the highest polynomial order is ``n-1``. This array creation
routine is helpful in generating linear least squares models, as such::
 
 >>> vander(np.linspace(0, 2, 5), 2)
 array([[0.  , 0.  , 1.  ],
        [0.25, 0.5 , 1.  ],
        [1.  , 1.  , 1.  ],
        [2.25, 1.5 , 1.  ],
        [4.  , 2.  , 1.  ]])
 >>> vander([1, 2, 3, 4], 2)
 array([[1, 1],
        [2, 1],
        [3, 1],
        [4, 1]])
 >>> vander((1, 2, 3, 4), 4)
 array([[ 1,  1,  1,  1],
        [ 8,  4,  2,  1],
        [27,  9,  3,  1],
        [64, 16,  4,  1]])
 
3 - ND array creation objects
-----------------------------

The ND array creation objects e.g. ``ones``, ``zeros``, and
``default_rng.random`` define arrays based upon the desired shape. ND
array creation objects can create arrays with any dimension by
specifying how many dimensions and length along that dimension in a
tuple or list. 

``zeros(shape)`` will create an array filled with 0 values with the specified
shape. The default dtype is float64. ::

 >>> np.zeros((2, 3))
 array([[ 0., 0., 0.], 
        [ 0., 0., 0.]])
 >>> np.zeros((2, 3, 2))
 array([[[0., 0.],
         [0., 0.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.]]])

``ones(shape)`` will create an array filled with 1 values. It is identical to
zeros in all other respects as such, ::

 >>> np.ones((2, 3))
 array([[ 1., 1., 1.], 
        [ 1., 1., 1.]])
 >>> np.ones((2, 3, 2))
 array([[[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]]])


indices() will create a set of arrays (stacked as a one-higher dimensioned
array), one per dimension with each representing variation in that dimension.
An example illustrates much better than a verbal description: ::

 >>> np.indices((3,3))
 array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]])

This is particularly useful for evaluating functions of multiple dimensions on
a regular grid.

3) Replicating, joining, or mutating existing arrays
====================================================

Once you have created arrays, you can replicate, join, or mutate those
existing arrays to create new arrays. When you assign an array or its
elements to a new variable, you have to explicitly ``copy`` the array,
otherwise the variable is a view into the original array. Consider the
following example, ::

 >>> a = array([1, 2, 3, 4, 5, 6])
 >>> b = a[:2]
 >>> b += 1
 >>> print('a = ', a, 'b = ', b)
 a =  [2 3 3 4 5 6] b =  [2 3]

In this example, you did not create a new array. You created a variable,
``b`` that viewed the first 2 elements of ``a``. When you added 1 to ``b`` you
would get the same result by adding 1 to ``a[:2]``. If you want to create a
_new_ array, use the ``copy`` array creation routine as such, ::

 >>> a = array([1, 2, 3, 4])
 >>> b = a[:2].copy()
 >>> b += 1
 >>> print('a = ', a, 'b = ', b)
 a =  [1 2 3 4 5 6] b =  [2 3]

There are a number of routines to join existing arrays e.g. ``vstack``,
``hstack``, and ``block``. Here is an example of joining four 2-by-2
arrays into a 4-by-4 array using ``block`` ::

 >>> A = ones((2, 2))
 >>> B = eye((2, 2))
 >>> C = zeros((2, 2))
 >>> D = diag((-3, -4))
 >>> block([[A, B], 
            [C, D]])
 array([[ 1.,  1.,  1.,  0. ],
        [ 1.,  1.,  0.,  1. ],
        [ 0.,  0., -3.,  0. ],
        [ 0.,  0.,  0., -4. ]])

Other routines use similar syntax to join ND arrays, check the
routine's documentation for further examples and syntax. 

4) Reading arrays from disk, either from standard or custom formats
========================

This is the most common case of large array creation. The details depend
greatly on the format of data on disk. This section gives
general pointers on how to handle various formats.

Standard Binary Formats
-----------------------

Various fields have standard formats for array data. The following lists the
ones with known python libraries to read them and return NumPy arrays (there
may be others for which it is possible to read and convert to NumPy arrays so
check the last section as well)
::

 HDF5: h5py
 FITS: Astropy
 # ?? anything else??

Examples of formats that cannot be read directly but for which it is not hard to
convert are those formats supported by libraries like PIL (able to read and
write many image formats such as jpg, png, etc).

Common ASCII Formats
--------------------

Delimited files such as comma separated value (csv) and tab separated
value (tsv) files are used for programs like Excel and LabView. Python
functions can read and parse these files line-by-line. NumPy has two
standard routines for importing a csvThere are CSV functions in Python
and functions in pylab (part of matplotlib).

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


