.. _basics.copies-and-views:

****************
Copies and views
****************

When operating on NumPy arrays, it is possible to access the internal data
buffer directly using a :ref:`view <view>` without copying data around. This
ensures good performance but can also cause unwanted problems if the user is
not aware of how this works. Hence, it is important to know the difference
between these two terms and to know which operations return copies and
which return views.

The NumPy array is a data structure consisting of two parts:
the :term:`contiguous` data buffer with the actual data elements and the
metadata that contains information about the data buffer. The metadata
includes data type, strides, and other important information that helps
manipulate the :class:`.ndarray` easily. See the :ref:`numpy-internals`
section for a detailed look.

.. _view:

View
====

It is possible to access the array differently by just changing certain
metadata like :term:`stride` and :term:`dtype` without changing the
data buffer. This creates a new way of looking at the data and these new
arrays are called views. The data buffer remains the same, so any changes made
to a view reflects in the original copy. A view can be forced through the
:meth:`.ndarray.view` method.

Copy
====

When a new array is created by duplicating the data buffer as well as the
metadata, it is called a copy. Changes made to the copy
do not reflect on the original array. Making a copy is slower and
memory-consuming but sometimes necessary. A copy can be forced by using
:meth:`.ndarray.copy`.

.. _indexing-operations:

Indexing operations
===================

.. seealso:: :ref:`basics.indexing`

Views are created when elements can be addressed with offsets and strides
in the original array. Hence, basic indexing always creates views.
For example::

    >>> import numpy as np
    >>> x = np.arange(10)
    >>> x
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> y = x[1:3]  # creates a view
    >>> y
    array([1, 2])
    >>> x[1:3] = [10, 11]
    >>> x
    array([ 0, 10, 11,  3,  4,  5,  6,  7,  8,  9])
    >>> y
    array([10, 11])

Here, ``y`` gets changed when ``x`` is changed because it is a view.

:ref:`advanced-indexing`, on the other hand, always creates copies.
For example::

    >>> import numpy as np
    >>> x = np.arange(9).reshape(3, 3)
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> y = x[[1, 2]]
    >>> y
    array([[3, 4, 5],
           [6, 7, 8]])
    >>> y.base is None
    True

Here, ``y`` is a copy, as signified by the :attr:`base <.ndarray.base>`
attribute. We can also confirm this by assigning new values to ``x[[1, 2]]``
which in turn will not affect ``y`` at all::

    >>> x[[1, 2]] = [[10, 11, 12], [13, 14, 15]]
    >>> x
    array([[ 0,  1,  2],
           [10, 11, 12],
           [13, 14, 15]])
    >>> y
    array([[3, 4, 5],
           [6, 7, 8]])

It must be noted here that during the assignment of ``x[[1, 2]]`` no view
or copy is created as the assignment happens in-place.


Other operations
================

The :func:`numpy.reshape` function creates a view where possible or a copy
otherwise. In most cases, the strides can be modified to reshape the
array with a view. However, in some cases where the array becomes
non-contiguous (perhaps after a :meth:`.ndarray.transpose` operation),
the reshaping cannot be done by modifying strides and requires a copy.

Taking the example of another operation, :func:`numpy.ravel` returns a
contiguous flattened view of the array wherever possible. On the other hand,
:meth:`.ndarray.flatten` always returns a flattened copy of the array.
However, to guarantee a view in most cases, ``x.reshape(-1)`` may be preferable.

How to tell if the array is a view or a copy
============================================

The :attr:`base <.ndarray.base>` attribute of the ndarray makes it easy
to tell if an array is a view or a copy. The base attribute of a view returns
the original array while it returns ``None`` for a copy.

    >>> import numpy as np
    >>> x = np.arange(9)
    >>> x
    array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    >>> y = x.reshape(3, 3)
    >>> y
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> y.base  # .reshape() creates a view
    array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    >>> z = y[[2, 1]]
    >>> z
    array([[6, 7, 8],
           [3, 4, 5]])
    >>> z.base is None  # advanced indexing creates a copy
    True

Note that the ``base`` attribute should not be used to determine
if an ndarray object is *new*; only if it is a view or a copy
of another ndarray.

.. Every entry checked with np.shares_memory against numpy 2.6.0.dev0.

Which operations return views
=============================

The tables below list, for each operation, whether the result shares memory
with its input. "View" includes the cases where an operation returns the input
array itself, because no data is copied either way. "View or copy" means the
answer depends on the input or on an argument, and the note says how.

Changing the shape
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Operation
     - Result
     - Notes
   * - :func:`numpy.reshape`
     - View or copy
     - A view when the elements, read in the requested order, lie at a
       constant stride in memory; otherwise a copy. ``copy=True`` always
       copies, and ``copy=False`` raises rather than copying.
   * - :func:`numpy.ravel`
     - View or copy
     - A view only when the array is contiguous in the requested order, so
       stricter than ``reshape(a, -1)``. ``order='K'`` and ``order='A'``
       accept Fortran-ordered input as well.
   * - :meth:`numpy.ndarray.flatten`
     - Copy
     - Always, unlike :func:`numpy.ravel`.
   * - :attr:`numpy.ndarray.flat`
     - Neither
     - An iterator over the original data, not an array.
   * - :func:`numpy.squeeze`
     - View
     - Returns the input itself when there are no axes of length one.
   * - :func:`numpy.expand_dims`
     - View
     -
   * - :func:`numpy.atleast_1d`, :func:`numpy.atleast_2d`,
       :func:`numpy.atleast_3d`
     - View
     - Returns the input itself when it already has enough dimensions.
   * - :func:`numpy.broadcast_to`
     - View
     - Read-only.
   * - :func:`numpy.broadcast_arrays`
     - View
     - One view per input array.
   * - :func:`numpy.resize`
     - Copy
     - The method :meth:`.ndarray.resize` changes the array in place
       instead.

Transpose-like operations
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Operation
     - Result
     - Notes
   * - :func:`numpy.transpose`, :func:`numpy.permute_dims`,
       :attr:`numpy.ndarray.T`
     - View
     - Only the shape and strides change; no data moves.
   * - :func:`numpy.matrix_transpose`, :attr:`numpy.ndarray.mT`
     - View
     -
   * - :func:`numpy.moveaxis`, :func:`numpy.rollaxis`,
       :func:`numpy.swapaxes`
     - View
     -

Converting and changing dtype
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Operation
     - Result
     - Notes
   * - :func:`numpy.array`
     - View or copy
     - By default it has ``copy=True``. ``copy=None`` and ``copy=False`` 
       both return the input itself when no conversion is needed; when 
       one is needed, ``copy=None`` copies and ``copy=False`` raises.
   * - :func:`numpy.asarray`, :func:`numpy.asanyarray`
     - View or copy
     - Returns the input itself unless a conversion is needed, such as a
       different dtype, in which case it copies. ``copy=True`` always copies.
   * - :func:`numpy.asarray_chkfinite`
     - View or copy
     - Behaves as :func:`numpy.asarray`, after raising on NaN or infinity.
   * - :func:`numpy.ascontiguousarray`
     - View or copy
     - The input itself when it is already C-contiguous, otherwise a copy.
       0-d input gives a new one-element view.
   * - :func:`numpy.asfortranarray`
     - View or copy
     - The input itself when it is already Fortran-contiguous, otherwise a copy. 
       0-d input gives a new one-element view.
   * - :func:`numpy.require`
     - View or copy
     - The input itself when it already meets every requirement, otherwise a
       copy. ``requirements='O'`` copies any array that does not own its data, even a
       contiguous one. 

   * - :func:`numpy.copy`, :meth:`numpy.ndarray.copy`
     - Copy
     -
   * - :meth:`numpy.ndarray.astype`
     - View or copy
     - Copies by default, even when the dtype is unchanged. 
       ``copy=False`` returns the input itself when nothing needs converting, 
       and copies otherwise; unlike ``np.array(..., copy=False)`` it never raises.
   * - :meth:`numpy.ndarray.view`
     - View
     -
   * - :meth:`numpy.ndarray.byteswap`
     - View or copy
     - Copies by default. ``inplace=True`` swaps the bytes in place and
       returns the input itself.
   * - :meth:`numpy.ndarray.conj`, :meth:`numpy.ndarray.conjugate`
     - View or copy
     - Returns the input itself for a real dtype, since conjugating changes
       nothing, and a copy for complex and object dtypes. The function
       :func:`numpy.conj` always copies.

Real and imaginary parts
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Operation
     - Result
     - Notes
   * - :func:`numpy.real`, :attr:`numpy.ndarray.real`
     - View or copy
     - A view of the real parts for a complex dtype. The input itself for a
       real dtype. A read-only copy for an object dtype.
   * - :func:`numpy.imag`, :attr:`numpy.ndarray.imag`
     - View or copy
     - A view of the imaginary parts for a complex dtype. Otherwise a new
       read-only array of zeros, since there is nothing to view.


Diagonals and triangles
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Operation
     - Result
     - Notes
   * - :func:`numpy.diag`
     - View or copy
     - A read-only view of the diagonal for 2-D input. For 1-D input it
       builds a new 2-D array instead.
   * - :func:`numpy.diagonal`, :meth:`numpy.ndarray.diagonal`
     - View
     - Read-only.
   * - :func:`numpy.diagflat`, :func:`numpy.tril`, :func:`numpy.triu`
     - Copy
     -

Rearranging elements
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Operation
     - Result
     - Notes
   * - :func:`numpy.flip`, :func:`numpy.fliplr`, :func:`numpy.flipud`
     - View
     - Reversing an axis only negates a stride. 0-d input returns a scalar.
   * - :func:`numpy.rot90`
     - View
     -
   * - :func:`numpy.roll`
     - Copy
     - Unlike the operations above, elements move between memory positions.

Splitting and joining
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Operation
     - Result
     - Notes
   * - :func:`numpy.split`, :func:`numpy.array_split`,
       :func:`numpy.hsplit`, :func:`numpy.vsplit`, :func:`numpy.dsplit`
     - View
     - Every piece is a view of the original array.
   * - :func:`numpy.concatenate`, :func:`numpy.stack`,
       :func:`numpy.hstack`, :func:`numpy.vstack`, :func:`numpy.dstack`,
       :func:`numpy.column_stack`, :func:`numpy.block`,
       :func:`numpy.append`
     - Copy
     - The result needs new memory to hold the inputs side by side.

Selecting elements
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Operation
     - Result
     - Notes
   * - :func:`numpy.take`, :meth:`numpy.ndarray.take`,
       :func:`numpy.compress`, :func:`numpy.extract`
     - Copy
     - Selecting elements by index or condition works like advanced indexing.
   * - :func:`numpy.repeat`, :func:`numpy.tile`, :func:`numpy.pad`
     - Copy
     -
   * - :func:`numpy.delete`, :func:`numpy.insert`
     - Copy
     -
   * - :func:`numpy.unique`
     - Copy
     -
   * - :func:`numpy.trim_zeros`
     - View
     - Trimming only moves the start and end of the array.


Indexing
--------

.. seealso:: :ref:`basics.indexing`

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Operation
     - Result
     - Notes
   * - Basic slicing, including a step, e.g. ``a[1:3]`` and ``a[::2]``
     - View
     -
   * - ``a[...]``
     - View
     -
   * - ``a[()]``
     - View
     - A scalar for 0-d input.
   * - ``a[np.newaxis]``
     - View
     -
   * - Integer-array (advanced) indexing, e.g. ``a[[0, 1]]``
     - Copy
     -
   * - Boolean indexing, e.g. ``a[a > 1]``
     - Copy
     -
   * - Advanced and basic indexing combined, e.g. ``a[[0, 1], :]``
     - Copy
     -
