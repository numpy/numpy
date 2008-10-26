.. _arrays.ndarray:

******************************************
The N-dimensional array (:class:`ndarray`)
******************************************

.. currentmodule:: numpy

An :class:`ndarray` is a (usually fixed-size) multidimensional
container of items of the same type and size. The number of dimensions
and items in an array is defined by its :attr:`shape <ndarray.shape>`,
which is a :class:`tuple` of *N* integers that specify the sizes of
each dimension. The type of items in the array is specified by a
separate :ref:`data-type object (dtype) <arrays.dtypes>`, one of which
is associated with each ndarray.

As with other container objects in Python, the contents of a
:class:`ndarray` can be accessed and modified by :ref:`indexing or
slicing <arrays.indexing>` the array (using for example *N* integers),
and via the methods and attributes of the :class:`ndarray`.

.. index:: view, base

Different :class:`ndarrays <ndarray>` can share the same data, so that
changes made in one :class:`ndarray` may be visible in another. That
is, an ndarray can be a *"view"* to another ndarray, and the data it
is referring to is taken care of by the *"base"* ndarray. ndarrays can
also be views to memory owned by Python :class:`strings <str>` or
objects implementing the :class:`buffer` or :ref:`array
<arrays.interface>` interfaces.


.. admonition:: Example

   A 2-dimensional array of size 2 x 3, composed of 4-byte integer elements:

   >>> x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
   >>> type(x)
   <type 'numpy.ndarray'>
   >>> x.shape
   (2, 3)
   >>> x.dtype
   dtype('int32')

   The array can be indexed using a Python container-like syntax:
   
   >>> x[1,2]
   6

   For example :ref:`slicing <arrays.indexing>` can produce views of the array:

   >>> y = x[:,1]
   >>> y[0] = 9
   >>> x
   array([[1, 9, 3],
          [4, 5, 6]])


Constructing arrays
===================

New arrays can be constructed using the routines detailed in
:ref:`routines.array-creation`, and also by using the low-level
:class:`ndarray` constructor:

.. autosummary::
   :toctree: generated/ 

   ndarray

.. _arrays.ndarray.indexing:


Indexing arrays
===============

Arrays can be indexed using an extended Python slicing syntax,
``array[selection]``.  Similar syntax is also used for accessing
fields in a :ref:`record array <arrays.dtypes>`.

.. seealso:: :ref:`Array Indexing <arrays.indexing>`.

Internal memory layout of an ndarray
====================================

An instance of class :class:`ndarray` consists of a contiguous
one-dimensional segment of computer memory (owned by the array, or by
some other object), combined with an indexing scheme that maps *N*
integers into the location of an item in the block.  The ranges in
which the indices can vary is specified by the :obj:`shape
<ndarray.shape>` of the array. How many bytes each item takes and how
the bytes are interpreted is defined by the :ref:`data-type object
<arrays.dtypes>` associated with the array.

.. index:: C-order, Fortran-order, row-major, column-major, stride, offset

A segment of memory is inherently 1-dimensional, and there are many
different schemes of arranging the items of an *N*-dimensional array to
a 1-dimensional block. Numpy is flexible, and :class:`ndarray` objects
can accommodate any *strided indexing scheme*. In a strided scheme,
the N-dimensional index :math:`(n_0, n_1, ..., n_{N-1})` corresponds
to the offset (in bytes)

.. math:: n_{\mathrm{offset}} = \sum_{k=0}^{N-1} s_k n_k

from the beginning of the memory block associated with the
array. Here, :math:`s_k` are integers which specify the :obj:`strides
<ndarray.strides>` of the array. The :term:`column-major` order (used
for example in the Fortran language and in *Matlab*) and
:term:`row-major` order (used in C) are special cases of the strided
scheme, and correspond to the strides:

.. math:: 

   s_k^{\mathrm{column}} = \prod_{j=0}^{k-1} d_j , \quad  s_k^{\mathrm{row}} = \prod_{j=k+1}^{N-1} d_j .

.. index:: single-segment, contiguous, non-contiguous

Both the C and Fortran orders are :term:`contiguous`, *i.e.*
:term:`single-segment`, memory layouts, in which every part of the
memory block can be accessed by some combination of the indices.

Data in new :class:`ndarrays <ndarray>` is in the :term:`row-major`
(C) order, unless otherwise specified, but for example :ref:`basic
array slicing <arrays.indexing>` often produces :term:`views <view>`
in a different scheme.

.. seealso: :ref:`Indexing <arrays.ndarray.indexing>`_

.. note:: 

   Several algorithms in NumPy work on arbitrarily strided arrays.
   However, some algorithms require single-segment arrays. When an
   irregularly strided array is passed in to such algorithms, a copy
   is automatically made.


Array attributes
================

Array attributes reflect information that is intrinsic to the array
itself. Generally, accessing an array through its attributes allows
you to get and sometimes set intrinsic properties of the array without
creating a new array. The exposed attributes are the core parts of an
array and only some of them can be reset meaningfully without creating
a new array. Information on each attribute is given below.

Memory layout
-------------

The following attributes contain information about the memory layout
of the array:

.. autosummary::
   :toctree: generated/

   ndarray.flags
   ndarray.shape
   ndarray.strides
   ndarray.ndim
   ndarray.data
   ndarray.size
   ndarray.itemsize
   ndarray.nbytes
   ndarray.base

.. note:: XXX: update and check these docstrings.

Data type
---------

.. seealso:: :ref:`Data type objects <arrays.dtypes>`

The data type object associated with the array can be found in the
:attr:`dtype <ndarray.dtype>` attribute:

.. autosummary::
   :toctree: generated/

   ndarray.dtype

.. note:: XXX: update the dtype attribute docstring: setting etc.

Other attributes
----------------

.. autosummary::
   :toctree: generated/

   ndarray.T
   ndarray.real
   ndarray.imag
   ndarray.flat
   ndarray.ctypes
   __array_priority__


.. _arrays.ndarray.array-interface:

Array interface
---------------

.. seealso:: :ref:`arrays.interface`.

==========================  ===================================
:obj:`__array_interface__`  Python-side of the array interface
:obj:`__array_struct__`     C-side of the array interface
==========================  ===================================

:mod:`ctypes` foreign function interface
----------------------------------------

.. autosummary::
   :toctree: generated/

   ndarray.ctypes

.. note:: XXX: update and check these docstrings.

Array methods
=============

An :class:`ndarray` object has many methods which operate on or with
the array in some fashion, typically returning an array result. These
methods are explained below.

For the following methods there are also corresponding functions in
:mod:`numpy`: :func:`all`, :func:`any`, :func:`argmax`,
:func:`argmin`, :func:`argsort`, :func:`choose`, :func:`clip`,
:func:`compress`, :func:`copy`, :func:`cumprod`, :func:`cumsum`,
:func:`diagonal`, :func:`imag`, :func:`max <amax>`, :func:`mean`,
:func:`min <amin>`, :func:`nonzero`, :func:`prod`, :func:`ptp`, :func:`put`,
:func:`ravel`, :func:`real`, :func:`repeat`, :func:`reshape`,
:func:`round <around>`, :func:`searchsorted`, :func:`sort`, :func:`squeeze`,
:func:`std`, :func:`sum`, :func:`swapaxes`, :func:`take`,
:func:`trace`, :func:`transpose`, :func:`var`.

Array conversion
----------------

.. autosummary::
   :toctree: generated/

   ndarray.item
   ndarray.tolist
   ndarray.itemset
   ndarray.tostring
   ndarray.tofile
   ndarray.dump
   ndarray.dumps
   ndarray.astype
   ndarray.byteswap
   ndarray.copy
   ndarray.view
   ndarray.getfield
   ndarray.setflags
   ndarray.fill

.. note:: XXX: update and check these docstrings.

Shape manipulation
------------------

For reshape, resize, and transpose, the single tuple argument may be
replaced with ``n`` integers which will be interpreted as an n-tuple.

.. autosummary::
   :toctree: generated/

   ndarray.reshape
   ndarray.resize
   ndarray.transpose
   ndarray.swapaxes
   ndarray.flatten
   ndarray.ravel
   ndarray.squeeze

Item selection and manipulation
-------------------------------

For array methods that take an *axis* keyword, it defaults to
:const:`None`. If axis is *None*, then the array is treated as a 1-D
array. Any other value for *axis* represents the dimension along which
the operation should proceed.

.. autosummary::
   :toctree: generated/

   ndarray.take
   ndarray.put
   ndarray.repeat
   ndarray.choose
   ndarray.sort
   ndarray.argsort
   ndarray.searchsorted
   ndarray.nonzero
   ndarray.compress
   ndarray.diagonal

Calculation
-----------

.. index:: axis

Many of these methods take an argument named *axis*. In such cases, 

- If *axis* is *None* (the default), the array is treated as a 1-D
  array and the operation is performed over the entire array. This
  behavior is also the default if self is a 0-dimensional array or
  array scalar.

- If *axis* is an integer, then the operation is done over the given axis
  (for each 1-D subarray that can be created along the given axis). 

The parameter *dtype* specifies the data type over which a reduction
operation (like summing) should take place. The default reduce data
type is the same as the data type of *self*. To avoid overflow, it can
be useful to perform the reduction using a larger data type.

For several methods, an optional *out* argument can also be provided
and the result will be placed into the output array given. The *out*
argument must be an :class:`ndarray` and have the same number of
elements. It can have a different data type in which case casting will
be performed.


.. autosummary::
   :toctree: generated/

   ndarray.argmax
   ndarray.min
   ndarray.argmin
   ndarray.ptp
   ndarray.clip
   ndarray.conj
   ndarray.round
   ndarray.trace
   ndarray.sum
   ndarray.cumsum
   ndarray.mean
   ndarray.var
   ndarray.std
   ndarray.prod
   ndarray.cumprod
   ndarray.all
   ndarray.any

Arithmetic and comparison operations
====================================

.. note:: XXX: write all attributes explicitly here instead of relying on
          the auto\* stuff?

.. index:: comparison, arithmetic, operation, operator

Arithmetic and comparison operations on :class:`ndarrays <ndarray>`
are defined as element-wise operations, and generally yield
:class:`ndarray` objects as results.

Each of the arithmetic operations (``+``, ``-``, ``*``, ``/``, ``//``,
``%``, ``divmod()``, ``**`` or ``pow()``, ``<<``, ``>>``, ``&``,
``^``, ``|``, ``~``) and the comparisons (``==``, ``<``, ``>``,
``<=``, ``>=``, ``!=``) is equivalent to the corresponding
:term:`universal function` (or :term:`ufunc` for short) in Numpy.  For
more information, see the section on :ref:`Universal Functions
<ufuncs>`.

Comparison operators:

.. autosummary::
   :toctree: generated/

   ndarray.__lt__
   ndarray.__le__
   ndarray.__gt__
   ndarray.__ge__
   ndarray.__eq__
   ndarray.__ne__

Truth value of an array (:func:`bool()`):

.. autosummary::
   :toctree: generated/

   ndarray.__nonzero__

.. note::

   Truth-value testing of an array invokes
   :meth:`ndarray.__nonzero__`, which raises an error if the number of
   elements in the the array is larger than 1, because the truth value
   of such arrays is ambiguous. Use :meth:`.any() <ndarray.any>` and
   :meth:`.all() <ndarray.all>` instead to be clear about what is meant in
   such cases. (If the number of elements is 0, the array evaluates to
   ``False``.)


Unary operations:

.. autosummary::
   :toctree: generated/
   
   ndarray.__neg__
   ndarray.__pos__
   ndarray.__abs__
   ndarray.__invert__

Arithmetic:

.. autosummary::
   :toctree: generated/
   
   ndarray.__add__
   ndarray.__sub__
   ndarray.__mul__
   ndarray.__div__
   ndarray.__truediv__
   ndarray.__floordiv__
   ndarray.__mod__
   ndarray.__divmod__
   ndarray.__pow__
   ndarray.__lshift__
   ndarray.__rshift__
   ndarray.__and__
   ndarray.__or__
   ndarray.__xor__

.. note:: 

   - Any third argument to :func:`pow()` is silently ignored,
     as the underlying :func:`ufunc <power>` only takes two arguments.

   - The three division operators are all defined; :obj:`div` is active
     by default, :obj:`truediv` is active when
     :obj:`__future__` division is in effect.

   - Because :class:`ndarray` is a built-in type (written in C), the
     ``__r{op}__`` special methods are not directly defined.

   - The functions called to implement many arithmetic special methods
     for arrays can be modified using :func:`set_numeric_ops`.

Arithmetic, in-place:

.. autosummary::
   :toctree: generated/
   
   ndarray.__iadd__
   ndarray.__isub__
   ndarray.__imul__
   ndarray.__idiv__
   ndarray.__itruediv__
   ndarray.__ifloordiv__
   ndarray.__imod__
   ndarray.__ipow__
   ndarray.__ilshift__
   ndarray.__irshift__
   ndarray.__iand__
   ndarray.__ior__
   ndarray.__ixor__

.. warning::

   In place operations will perform the calculation using the
   precision decided by the data type of the two operands, but will
   silently downcast the result (if necessary) so it can fit back into
   the array.  Therefore, for mixed precision calculations, ``A {op}=
   B`` can be different than ``A = A {op} B``. For example, suppose
   ``a = ones((3,3))``. Then, ``a += 3j`` is different than ``a = a +
   3j``: While they both perform the same computation, ``a += 3``
   casts the result to fit back in ``a``, whereas ``a = a + 3j``
   re-binds the name ``a`` to the result.


Special methods
===============

For standard library functions:

.. autosummary::
   :toctree: generated/

   ndarray.__copy__
   ndarray.__deepcopy__
   ndarray.__reduce__
   ndarray.__setstate__

Basic customization:

.. autosummary::
   :toctree: generated/

   ndarray.__new__
   ndarray.__array__
   ndarray.__array_wrap__

Container customization: (see :ref:`Indexing <arrays.indexing>`)

.. autosummary::
   :toctree: generated/

   ndarray.__len__
   ndarray.__getitem__
   ndarray.__setitem__
   ndarray.__getslice__
   ndarray.__setslice__
   ndarray.__contains__

Conversion; the operations :func:`complex()`, :func:`int()`,
:func:`long()`, :func:`float()`, :func:`oct()`, and
:func:`hex()`. They work only on arrays that have one element in them
and return the appropriate scalar.

.. autosummary::
   :toctree: generated/

   ndarray.__int__
   ndarray.__long__
   ndarray.__float__
   ndarray.__oct__
   ndarray.__hex__

String representations:

.. autosummary::
   :toctree: generated/

   ndarray.__str__
   ndarray.__repr__
