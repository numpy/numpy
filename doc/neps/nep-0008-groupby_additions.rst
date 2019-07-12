=============================================================
NEP 8 —  A proposal for adding groupby functionality to NumPy
=============================================================

:Author: Travis Oliphant
:Contact: oliphant@enthought.com
:Date: 2010-04-27
:Status: Deferred


Executive summary
=================

NumPy provides tools for handling data and doing calculations in much
the same way as relational algebra allows.  However, the common group-by
functionality is not easily handled.  The reduce methods of NumPy's
ufuncs are a natural place to put this groupby behavior.  This NEP
describes two additional methods for ufuncs (reduceby and reducein) and
two additional functions (segment and edges) which can help add this
functionality.

Example Use Case
================
Suppose you have a NumPy structured array containing information about
the number of purchases at several stores over multiple days.  To be clear, the
structured array data-type is::

  dt = [('year', i2), ('month', i1), ('day', i1), ('time', float),
      ('store', i4), ('SKU', 'S6'), ('number', i4)]

Suppose there is a 1-d NumPy array of this data-type and you would like
to compute various statistics (max, min, mean, sum, etc.) on the number
of products sold, by product, by month, by store, etc.

Currently, this could be done by using reduce methods on the number
field of the array, coupled with in-place sorting, unique with
return_inverse=True and bincount, etc.  However, for such a common
data-analysis need, it would be nice to have standard and more direct
ways to get the results.


Ufunc methods proposed
======================

It is proposed to add two new reduce-style methods to the ufuncs:
reduceby and reducein.  The reducein method is intended to be a simpler
to use version of reduceat, while the reduceby method is intended to
provide group-by capability on reductions.

reducein::

        <ufunc>.reducein(arr, indices, axis=0, dtype=None, out=None)

        Perform a local reduce with slices specified by pairs of indices.

        The reduction occurs along the provided axis, using the provided
        data-type to calculate intermediate results, storing the result into
        the array out (if provided).

        The indices array provides the start and end indices for the
        reduction.  If the length of the indices array is odd, then the
        final index provides the beginning point for the final reduction
        and the ending point is the end of arr.

        This generalizes along the given axis, the behavior:

        [<ufunc>.reduce(arr[indices[2*i]:indices[2*i+1]])
                for i in range(len(indices)/2)]

        This assumes indices is of even length

        Example:
           >>> a = [0,1,2,4,5,6,9,10]
           >>> add.reducein(a,[0,3,2,5,-2])
           [3, 11, 19]

           Notice that sum(a[0:3]) = 3; sum(a[2:5]) = 11; and sum(a[-2:]) = 19

reduceby::

        <ufunc>.reduceby(arr, by, dtype=None, out=None)

        Perform a reduction in arr over unique non-negative integers in by.


        Let N=arr.ndim and M=by.ndim.  Then, by.shape[:N] == arr.shape.
        In addition, let I be an N-length index tuple, then by[I]
        contains the location in the output array for the reduction to
        be stored.  Notice that if N == M, then by[I] is a non-negative
        integer, while if N < M, then by[I] is an array of indices into
        the output array.

        The reduction is computed on groups specified by unique indices
        into the output array. The index is either the single
        non-negative integer if N == M or if N < M, the entire
        (M-N+1)-length index by[I] considered as a whole.


Functions proposed
==================

- segment
- edges
