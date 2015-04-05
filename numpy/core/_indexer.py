"""
A Python implementation of orthogonal indexing, to be called from
C code, and used by other libraries as a canonical implementation.

"""
from __future__ import division, absolute_import, print_function

import numpy as np


class OrthogonalIndexer(object):
    """
    """

    def __init__(self, arr):
        if not isinstance(arr, np.ndarray):
            raise TypeError
        self._arr = arr

    def expand_ellipsis(self, key, na_count):
        """Given an indexing tuple 'key', it expands it to have as many
        items as dimensions in 'self._arr', properly replacing Ellipsis
        by slices. Must also be provided with the number of newaxis
        indices in 'key'.
        """
        ndim = self._arr.ndim
        ellipsis_count = sum(k is Ellipsis for k in key)
        if ellipsis_count > 1:
            # Using multiple Ellipsis is deprecated
            raise ValueError('an index can only have a single '
                             'Ellipsis (...); replace all but '
                             'one with slices (:).')
        elif ellipsis_count == 1:
            idx = key.index(Ellipsis)
            new_slices = ndim + na_count - len(key) + 1
            key = key[:idx] + (slice(None),)*new_slices + key[idx+1:]
        if len(key) <= ndim + na_count:
            key += (slice(None),)*(len(key) - ndim - na_count)
        else:
            raise IndexError('too many indices for array')

        return key


    def remap_indices(self, key):
        """
        """
        if not isinstance(key, tuple):
            key = (key,)
        na_count = sum(k is None for k in key)
        ndim = self._arr.ndim
        key = self.expand_ellipsis(key, na_count)

        # Integer indices and their position in 'key' w/o newaxis indices
        ints = []
        int_idx_no_na = []
        # Converted array indices and their position in 'key' w/o
        # newaxis indices and w/o integer indices
        arrays = []
        array_idx_no_na = []
        array_idx_no_ints = []
        # Slice indices and their position in 'key' w/o newaxis indices
        # and w/o integer indices
        slices = []
        slice_idx_no_na = []
        slice_idx_no_ints = []
        # Position of newaxis indices in 'key' w/o integer indices
        newaxis_idx_no_ints = []

        # Number of dimensions so far holding integer indices
        int_offset = 0
        # Number of dimensions so far holding newaxis indices
        na_offset = 0

        for j, k in enumerate(key):
            j_ints = j - int_offset
            j_na = j - na_offset
            if isinstance(k, slice):
                slices.append(k)
                slice_idx_no_na.append(j_na)
                slice_idx_no_ints.append(j_ints)
            elif k is None:
                newaxis_idx_no_ints.append(j_ints)
                na_offset += 1
            else:
                k_indexer = np.asarray(k)
                if k_indexer.ndim == 0:
                    ints.append(int(k_indexer.item()))
                    int_idx_no_na.append(j_na)
                    int_offset += 1
                else:
                    if k_indexer.ndim != 1 or k_indexer.dtype.kind not in 'bi':
                        raise ValueError('only integers, slices (:), '
                                         'ellipsis (...), numpy.newaxis '
                                         '(None) and 1-D integer or boolean '
                                         'arrays are valid indices.')
                    if k_indexer.dtype.kind == 'b':
                        k_indexer, = np.nonzero(k_indexer)
                    arrays.append(k_indexer)
                    array_idx_no_na.append(j_na)
                    array_idx_no_ints.append(j_ints)

        # Append size-1 dimensions to indexing arrays so they broadcast
        # to the orthogonal indexing shape
        for j, a in enumerate(arrays):
            a.shape = (-1,) + (1,) * (len(arrays) - 1 - j)

        # 'new_key' holds all the integer indices, followed by all the
        # array indices, properly broadcasted, followed by all the slice
        # indices.
        new_key = tuple(ints + arrays + slices)
        # Before applying the remapped indices to an array, its
        # dimensions must be reordered according to 'idx_no_na', which
        # holds the positions of those indices, excluding newaxis ones.
        idx_no_na = np.array(int_idx_no_na + array_idx_no_na +
                             slice_idx_no_na, dtype=np.intp)
        # After indexing, the resulting array should have as many size-1
        # dimensions as newaxis indices found prepended, and can then
        # be reorded with the inverse of the mapping in 'idx_no_ints'.
        idx_no_ints = np.array(newaxis_idx_no_ints + array_idx_no_ints +
                               slice_idx_no_ints, dtype=np.intp)

        return new_key, idx_no_na, idx_no_ints, na_offset


    def __getitem__(self, key):
        new_key, idx_no_na, idx_no_ints, na_count = self.remap_indices(key)

        # Remap the axes of 'self._arr' to match 'new_key'
        arr = self._arr.transpose(idx_no_na)
        # Apply the 'new_key' indexing
        ret = arr[new_key]
        # Prepend to the result 'na_count' size-1 dimensions
        ret.shape = (1,)*na_count + ret.shape
        # Invert the permutation defined by 'key_idx_no_ints'
        reorder = np.empty_like(idx_no_ints)
        reorder[idx_no_ints] = np.arange(idx_no_ints.size)
        # Return the result of remapping with the inverse permutation
        return ret.transpose(reorder)


    def __setitem__(self, key, value):
        new_key, idx_no_na, idx_no_ints, na_count = self.remap_indices(key)

        # Remap the axes of 'self._arr' to match 'new_key'
        arr = self._arr.transpose(idx_no_na)
        # Convert 'value' to an array, making sure it has at least as
        # many dimensions as 'idx_no_ints'
        value = np.array(value, copy=False, ndmin=idx_no_ints.size)
        # Before remapping the axes of 'value', we need to make sure it
        # has the right number of dimensions, by removing size-1 ones
        # from the front of the shape.
        # TODO: This will fail if the dimensions removed are not all
        # size-1, but a more informative error is probably in order.
        value.shape = value.shape[-idx_no_ints.size:]
        # Remap the axes of 'value' to match those of 'arr' after indexing
        value = value.transpose(idx_no_ints)
        # Finally, assign the remapped 'value' into the remapped and
        # indexed 'arr'
        arr[new_key] = value
