.. currentmodule:: numpy

.. _how-to-index.rst:

*****************************************
How to index :class:`ndarrays <.ndarray>`
*****************************************

.. seealso:: :ref:`basics.indexing`

This page tackles common examples. For an in-depth look into indexing, refer
to :ref:`basics.indexing`.

Access specific/random rows and columns
=======================================

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
    >>> a[0, : ,3]
    array([ 3,  8, 13])

Note that the dimensions are not preserved in the above example. To add the
required dimensions, :func:`newaxis` is used here. To use other such tools
refer to :ref:`dimensional-indexing-tools`.

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

Index along a specific axis
---------------------------

Use :meth:`take`. See also :meth:`take_along_axis` and
:meth:`put_along_axis`.

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

Index columns
-------------

Use :ref:`dimensional-indexing-tools` to avoid shape mismatches::

    >>> arr = np.arange(3*4).reshape(3, 4)
    >>> column_indices = [[1, 3], [0, 2], [2, 2]]
    >>> np.arange(arr.shape[0])[:, np.newaxis]
    array([[0],
           [1],
           [2]])
    >>> arr[np.arange(arr.shape[0])[:, np.newaxis], column_indices]
    array([[ 1,  3],
           [ 4,  6],
           [10, 10]])

Create subsets of larger matrices
=================================

Use :ref:`slicing-and-striding` to access chunks of an array.
But if you want to access multiple scattered elements to create
complicated subsets, you have to use :ref:`advanced-indexing`. Use
:func:`ix_` to quickly contruct index arrays::

    >>> indices = np.ix_([0, 1], [0, 2], [2, 4])
    >>> indices
    (array([[[0]],
    <BLANKLINE>
           [[1]]]), 
    array([[[0],
            [2]]]), 
    array([[[2, 4]]]))
    >>> a[indices]
    array([[[ 2,  4],
            [12, 14]],
    <BLANKLINE>
            [[17, 19],
            [27, 29]]])
    >>> indices = np.ix_([0, 1], [0, 1])
    >>> indices
    (array([[0],
           [1]]),
    array([[0, 1]]))
    >>> a[indices]
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9]],
    <BLANKLINE>
            [[15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]]])

Filter values
=============

Use :meth:`where` to generate indices based on conditions and then
use :ref:`advanced-indexing`.

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

Replace values
--------------

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

Index the same ndarray multiple times efficiently
=================================================

It must be kept in mind that basic indexing produces :term:`views <view>`
and advanced indexing produces :term:`copies <copy>`, which takes
more time. Hence, you should take care to use basic indexing wherever
possible instead of advanced indexing. 