.. currentmodule:: numpy

.. _how-to-index:

*****************************************
How to index :class:`ndarrays <.ndarray>`
*****************************************

.. seealso:: :ref:`basics.indexing`

This page tackles common examples. For an in-depth look into indexing, refer
to :ref:`basics.indexing`.

Access specific/arbitrary rows and columns
==========================================

Use :ref:`basic-indexing` features like :ref:`slicing-and-striding`, and
:ref:`dimensional-indexing-tools`.

    >>> a = np.arange(30).reshape(2, 3, 5)
    >>> a
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14]],
    <BLANKLINE>
            [[15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29]]])
    >>> a[0, 2, :]
    array([10, 11, 12, 13, 14])
    >>> a[0, :, 3]
    array([ 3,  8, 13])
    
Note that the output from indexing operations can have different shape from the
original object. To preserve the original dimensions after indexing, you can
use :func:`newaxis`. To use other such tools, refer to
:ref:`dimensional-indexing-tools`.

    >>> a[0, :, 3].shape
    (3,)
    >>> a[0, :, 3, np.newaxis].shape
    (3, 1)
    >>> a[0, :, 3, np.newaxis, np.newaxis].shape
    (3, 1, 1)

Variables can also be used to index::

    >>> y = 0
    >>> a[y, :, y+3]
    array([ 3,  8, 13])

Refer to :ref:`dealing-with-variable-indices` to see how to use
:term:`python:slice` and :py:data:`Ellipsis` in your index variables.

Index columns
-------------

To index columns, you have to index the last axis. Use
:ref:`dimensional-indexing-tools` to get the desired number of dimensions::

    >>> a = np.arange(24).reshape(2, 3, 4)
    >>> a
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    <BLANKLINE>
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
    >>> a[..., 3]
    array([[ 3,  7, 11],
           [15, 19, 23]])

To index specific elements in each column, make use of :ref:`advanced-indexing`
as below::

    >>> arr = np.arange(3*4).reshape(3, 4)
    >>> arr
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    >>> column_indices = [[1, 3], [0, 2], [2, 2]]
    >>> np.arange(arr.shape[0])
    array([0, 1, 2])
    >>> row_indices = np.arange(arr.shape[0])[:, np.newaxis]
    >>> row_indices
    array([[0],
           [1],
           [2]])

Use the ``row_indices`` and ``column_indices`` for advanced
indexing::

    >>> arr[row_indices, column_indices]
    array([[ 1,  3],
           [ 4,  6],
           [10, 10]])

Index along a specific axis
---------------------------

Use :meth:`take`. See also :meth:`take_along_axis` and
:meth:`put_along_axis`.

    >>> a = np.arange(30).reshape(2, 3, 5)
    >>> a
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14]],
    <BLANKLINE>
            [[15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29]]])
    >>> np.take(a, [2, 3], axis=2)
    array([[[ 2,  3],
            [ 7,  8],
            [12, 13]],
    <BLANKLINE>
            [[17, 18],
            [22, 23],
            [27, 28]]])
    >>> np.take(a, [2], axis=1)
    array([[[10, 11, 12, 13, 14]],
    <BLANKLINE>
            [[25, 26, 27, 28, 29]]])

Create subsets of larger matrices
=================================

Use :ref:`slicing-and-striding` to access chunks of a large array::

    >>> a = np.arange(100).reshape(10, 10)
    >>> a
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
            [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
            [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
            [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])
    >>> a[2:5, 2:5]
    array([[22, 23, 24],
           [32, 33, 34],
           [42, 43, 44]])
    >>> a[2:5, 1:3]
    array([[21, 22],
           [31, 32],
           [41, 42]])
    >>> a[:5, :5]
    array([[ 0,  1,  2,  3,  4],
           [10, 11, 12, 13, 14],
           [20, 21, 22, 23, 24],
           [30, 31, 32, 33, 34],
           [40, 41, 42, 43, 44]])

The same thing can be done with advanced indexing in a slightly more complex
way. Remember that
:ref:`advanced indexing creates a copy <indexing-operations>`::

    >>> a[np.arange(5)[:, None], np.arange(5)[None, :]]
    array([[ 0,  1,  2,  3,  4],
           [10, 11, 12, 13, 14],
           [20, 21, 22, 23, 24],
           [30, 31, 32, 33, 34],
           [40, 41, 42, 43, 44]])

You can also use :meth:`mgrid` to generate indices::

    >>> indices = np.mgrid[0:6:2]
    >>> indices
    array([0, 2, 4])
    >>> a[:, indices]
    array([[ 0,  2,  4],
           [10, 12, 14],
           [20, 22, 24],
           [30, 32, 34],
           [40, 42, 44],
           [50, 52, 54],
           [60, 62, 64],
           [70, 72, 74],
           [80, 82, 84],
           [90, 92, 94]])

Filter values
=============

Non-zero elements
-----------------

Use :meth:`nonzero` to get a tuple of array indices of non-zero elements 
corresponding to every dimension::

	>>> z = np.array([[1, 2, 3, 0], [0, 0, 5, 3], [4, 6, 0, 0]])
       >>> z
       array([[1, 2, 3, 0],
              [0, 0, 5, 3],
              [4, 6, 0, 0]])
       >>> np.nonzero(z)
       (array([0, 0, 0, 1, 1, 2, 2]), array([0, 1, 2, 2, 3, 0, 1]))

Use :meth:`flatnonzero` to fetch indices of elements that are non-zero in
the flattened version of the ndarray::

	>>> np.flatnonzero(z)
	array([0, 1, 2, 6, 7, 8, 9])

Arbitrary conditions
--------------------

Use :meth:`where` to generate indices based on conditions and then
use :ref:`advanced-indexing`.

    >>> a = np.arange(30).reshape(2, 3, 5)
    >>> indices = np.where(a % 2 == 0)
    >>> indices
    (array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]), 
    array([0, 0, 0, 1, 1, 2, 2, 2, 0, 0, 1, 1, 1, 2, 2]), 
    array([0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3]))
    >>> a[indices]
    array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])

Or, use :ref:`boolean-indexing`::

    >>> a > 14
    array([[[False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False]],
    <BLANKLINE>
           [[ True,  True,  True,  True,  True],
            [ True,  True,  True,  True,  True],
            [ True,  True,  True,  True,  True]]])
    >>> a[a > 14]
    array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])

Replace values after filtering
------------------------------

Use assignment with filtering to replace desired values::

    >>> p = np.arange(-10, 10).reshape(2, 2, 5)
    >>> p
    array([[[-10,  -9,  -8,  -7,  -6],
            [ -5,  -4,  -3,  -2,  -1]],
    <BLANKLINE>
           [[  0,   1,   2,   3,   4],
            [  5,   6,   7,   8,   9]]])
    >>> q = p < 0
    >>> q
    array([[[ True,  True,  True,  True,  True],
            [ True,  True,  True,  True,  True]],
    <BLANKLINE>
           [[False, False, False, False, False],
            [False, False, False, False, False]]])
    >>> p[q] = 0
    >>> p
    array([[[0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]],
    <BLANKLINE>
           [[0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9]]])

Fetch indices of max/min values
===============================

Use :meth:`argmax` and :meth:`argmin`::

    >>> a = np.arange(30).reshape(2, 3, 5)
    >>> np.argmax(a)
    29
    >>> np.argmin(a)
    0

Use the ``axis`` keyword to get the indices of maximum and minimum
values along a specific axis::

    >>> np.argmax(a, axis=0)
    array([[1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1]])
    >>> np.argmax(a, axis=1)
    array([[2, 2, 2, 2, 2],
           [2, 2, 2, 2, 2]])
    >>> np.argmax(a, axis=2)
    array([[4, 4, 4],
           [4, 4, 4]])
    <BLANKLINE>
    >>> np.argmin(a, axis=1)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> np.argmin(a, axis=2)
    array([[0, 0, 0],
           [0, 0, 0]])

Set ``keepdims`` to ``True`` to keep the axes which are reduced in the
result as dimensions with size one::

    >>> np.argmin(a, axis=2, keepdims=True)
    array([[[0],
            [0],
            [0]],
    <BLANKLINE>
           [[0],
            [0],
            [0]]])
    >>> np.argmax(a, axis=1, keepdims=True)
    array([[[2, 2, 2, 2, 2]],
    <BLANKLINE>
           [[2, 2, 2, 2, 2]]])
	   
To get the indices of each maximum or minimum value for each
(N-1)-dimensional array in an N-dimensional array, use :meth:`reshape`
to reshape the array to a 2D array, apply :meth:`argmax` or :meth:`argmin`
along ``axis=1`` and use :meth:`unravel_index` to recover the index of the
values per slice::

    >>> x = np.arange(2*2*3).reshape(2, 2, 3) % 7  # 3D example array
    >>> x
    array([[[0, 1, 2],
            [3, 4, 5]],
    <BLANKLINE>
           [[6, 0, 1],
            [2, 3, 4]]])
    >>> x_2d = np.reshape(x, (x.shape[0], -1))
    >>> indices_2d = np.argmax(x_2d, axis=1)
    >>> indices_2d
    array([5, 0])
    >>> np.unravel_index(indices_2d, x.shape[1:])
    (array([1, 0]), array([2, 0]))
    
The first array returned contains the indices along axis 1 in the original
array, the second array contains the indices along axis 2. The highest
value in ``x[0]`` is therefore ``x[0, 1, 2]``.

Index the same ndarray multiple times efficiently
=================================================

It must be kept in mind that basic indexing produces :term:`views <view>`
and advanced indexing produces :term:`copies <copy>`, which are
computationally less efficient. Hence, you should take care to use basic
indexing wherever possible instead of advanced indexing.

Further reading
===============

Nicolas Rougier's `100 NumPy exercises <https://github.com/rougier/numpy-100>`_
provide a good insight into how indexing is combined with other operations.
Exercises `6`_, `8`_, `10`_, `15`_, `16`_, `19`_, `20`_, `45`_, `59`_,
`64`_, `65`_, `70`_, `71`_, `72`_, `76`_, `80`_, `81`_, `84`_, `87`_, `90`_,
`93`_, `94`_ are specially focused on indexing. 

.. _6: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#6-create-a-null-vector-of-size-10-but-the-fifth-value-which-is-1-
.. _8: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#8-reverse-a-vector-first-element-becomes-last-
.. _10: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#10-find-indices-of-non-zero-elements-from-120040-
.. _15: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#15-create-a-2d-array-with-1-on-the-border-and-0-inside-
.. _16: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#16-how-to-add-a-border-filled-with-0s-around-an-existing-array-
.. _19: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#19-create-a-8x8-matrix-and-fill-it-with-a-checkerboard-pattern-
.. _20: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#20-consider-a-678-shape-array-what-is-the-index-xyz-of-the-100th-element-
.. _45: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#45-create-random-vector-of-size-10-and-replace-the-maximum-value-by-0-
.. _59: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#59-how-to-sort-an-array-by-the-nth-column-
.. _64: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#64-consider-a-given-vector-how-to-add-1-to-each-element-indexed-by-a-second-vector-be-careful-with-repeated-indices-
.. _65: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#65-how-to-accumulate-elements-of-a-vector-x-to-an-array-f-based-on-an-index-list-i-
.. _70: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#70-consider-the-vector-1-2-3-4-5-how-to-build-a-new-vector-with-3-consecutive-zeros-interleaved-between-each-value-
.. _71: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#71-consider-an-array-of-dimension-553-how-to-mulitply-it-by-an-array-with-dimensions-55-
.. _72: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#72-how-to-swap-two-rows-of-an-array-
.. _76: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#76-consider-a-one-dimensional-array-z-build-a-two-dimensional-array-whose-first-row-is-z0z1z2-and-each-subsequent-row-is--shifted-by-1-last-row-should-be-z-3z-2z-1-
.. _80: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#80-consider-an-arbitrary-array-write-a-function-that-extract-a-subpart-with-a-fixed-shape-and-centered-on-a-given-element-pad-with-a-fill-value-when-necessary-
.. _81: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#81-consider-an-array-z--1234567891011121314-how-to-generate-an-array-r--1234-2345-3456--11121314-
.. _84: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#84-extract-all-the-contiguous-3x3-blocks-from-a-random-10x10-matrix-
.. _87: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#87-consider-a-16x16-array-how-to-get-the-block-sum-block-size-is-4x4-
.. _90: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#90-given-an-arbitrary-number-of-vectors-build-the-cartesian-product-every-combinations-of-every-item-
.. _93: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#93-consider-two-arrays-a-and-b-of-shape-83-and-22-how-to-find-rows-of-a-that-contain-elements-of-each-row-of-b-regardless-of-the-order-of-the-elements-in-b-
.. _94: https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md#94-considering-a-10x3-matrix-extract-rows-with-unequal-values-eg-223-
