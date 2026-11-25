.. _how-to-stack:

*********************
How to join arrays
*********************

Use :func:`numpy.concatenate`, :func:`numpy.stack`, or :func:`numpy.block`
when you need to join arrays. Choose the function based on whether you want
to extend an existing axis, add a new axis, or assemble a nested layout.

Choose a function
=================

Follow this decision guide::

    Do you want a new axis in the result?
    |
    +-- Yes --> np.stack([...], axis=...)
    |
    +-- No --> Do the arrays form a nested layout (such as a block matrix)?
                |
                +-- Yes --> np.block([...])
                |
                +-- No --> np.concatenate([...], axis=...)

* :func:`numpy.concatenate` joins arrays along an **existing axis**. The
  inputs must have the same shape along all other axes.
* :func:`numpy.stack` joins equally shaped arrays along a **new axis**.
* :func:`numpy.block` assembles arrays using **nested lists** that describe
  their arrangement. It concatenates from the innermost lists outward.

Join arrays along an existing axis
==================================

Use :func:`numpy.concatenate` to extend an existing dimension. For example,
2 arrays with 2 rows each can be joined along ``axis=1`` (columns)::

    >>> import numpy as np
    >>> left = np.array([[1, 2], [3, 4]])
    >>> right = np.array([[5], [6]])
    >>> np.concatenate((left, right), axis=1)
    array([[1, 2, 5],
           [3, 4, 6]])

The result has shape ``(2, 3)``. Joining along ``axis=0`` (rows) instead
requires the inputs to have the same number of columns::

    >>> bottom = np.array([[7, 8]])
    >>> np.concatenate((left, bottom), axis=0)
    array([[1, 2],
           [3, 4],
           [7, 8]])

Add a new axis
==============

Use :func:`numpy.stack` when each input has the same shape and you want
an additional dimension::

    >>> first = np.array([1, 2])
    >>> second = np.array([3, 4])
    >>> np.stack((first, second), axis=0)
    array([[1, 2],
           [3, 4]])
    >>> np.stack((first, second), axis=1)
    array([[1, 3],
           [2, 4]])

Both results have shape ``(2, 2)``, but the new axis is inserted in a
different position. In contrast, ``np.concatenate((first, second))``
returns a 1-dimensional array of shape ``(4,)``.

Assemble arrays into blocks
===========================

Use :func:`numpy.block` when the intended result is easier to express as
a layout of subarrays. Supply nested lists: the inner lists describe
horizontal joins (along the last axis), and outer lists describe vertical
joins (along the second-to-last axis)::

    >>> top_left = np.array([[1, 2], [3, 4]])
    >>> top_right = np.array([[5], [6]])
    >>> bottom_left = np.array([[7, 8]])
    >>> bottom_right = np.array([[9]])
    >>> np.block([[top_left, top_right],
    ...           [bottom_left, bottom_right]])
    array([[1, 2, 5],
           [3, 4, 6],
           [7, 8, 9]])

The inner lists first join the left and right blocks; the outer list
then joins the 2 rows of blocks. The result has shape ``(3, 3)``.
Unlike :func:`numpy.stack`, the blocks do not need identical shapes.
They do need compatible dimensions at each join.

A block layout is not necessarily a strict grid of equal-width columns:
concatenation is performed from the innermost lists outward, so different
rows of blocks can have different internal divisions, provided their
combined widths agree. Here the bottom row is a single block::

    >>> bottom = np.array([[7, 8, 9]])
    >>> np.block([[top_left, top_right],
    ...           [bottom]])
    array([[1, 2, 5],
           [3, 4, 6],
           [7, 8, 9]])

To construct a simple block matrix, scalars can also be used::

    >>> np.block([[1, 2], [3, 4]])
    array([[1, 2],
           [3, 4]])

Avoid common shape errors
=========================

Inspect ``array.shape`` before combining arrays to check whether the
operation will preserve, add, or extend an axis. In addition to the shape
requirements described above, :func:`numpy.block` has two rules of its own:

* It can prepend leading dimensions of length one to inputs, but it does
  **not** broadcast arrays to make incompatible blocks fit.
* All nested lists must have the same depth, and none may be empty. The
  error message says which element is at fault::

    >>> np.block([[1, 2], [[3, 4]]])
    Traceback (most recent call last):
    ...
    ValueError: List depths are mismatched. First element was at depth 2, but there is an element at depth 3 (arrays[1][0][0])

.. seealso::

   :func:`numpy.concatenate`, :func:`numpy.stack`, :func:`numpy.block`
