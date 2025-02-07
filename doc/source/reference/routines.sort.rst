Sorting, Searching, and Counting
================================

.. currentmodule:: numpy

Sorting
-------
.. autosummary::
   :toctree: generated/

   sort
   lexsort
   argsort
   ndarray.sort
   sort_complex
   partition
   argpartition

Searching
---------
.. autosummary::
   :toctree: generated/

   argmax
   nanargmax
   argmin
   nanargmin
   argwhere
   nonzero
   flatnonzero
   where
   searchsorted
   extract

searchsorted(a, v, side='left', sorter=None)
------------------------------------------------

The `searchsorted` function returns the indices where elements of `v` should be inserted into the sorted array `a` to maintain the order. 

Parameters:
------------
- `a` : array_like
    A sorted array where values from `v` are inserted to maintain order.
- `v` : array_like
    The values that are to be inserted into `a`.
- `side` : {'left', 'right'}, optional
    If 'left', the returned index is the first position where the element could be inserted. If 'right', it is the last position.
    The default is 'left'.
- `sorter` : array_like, optional
    An array of indices that sort `a`. If provided, `a` does not need to be sorted.

Example:
--------
>>> import numpy as np
>>> a = np.array([1, 3, 4, 6])
>>> v = np.array([2, 5])
>>> np.searchsorted(a, v)
array([1, 3])

Notes:
------
- The array `a` must be sorted for correct results. If `a` is not sorted, the behavior of `searchsorted` may be incorrect.
- If `a` contains `NaN` values, the behavior might depend on the `side` parameter.

Counting
--------
.. autosummary::
   :toctree: generated/

   count_nonzero
