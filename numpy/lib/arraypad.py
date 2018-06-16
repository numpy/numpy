"""
Functions to pad values onto the edges of an n-dimensional array.
"""


from __future__ import division, absolute_import, print_function

import numpy as np


__all__ = ['pad']


def _slice_at_axis(shape, axis, sl):
    """
    Construct a slice tuple the length of shape, with sl at the specified axis
    """
    slice_tup = (slice(None),)
    return slice_tup * axis + (sl,) + slice_tup * (len(shape) - axis - 1)


def _slice_pad_area(shape, axis, from_index):
    if 0 <= from_index:
        sl = slice(0, from_index)
    else:
        sl = slice(from_index, shape[axis])
    return _slice_at_axis(shape, axis, sl)


def _slice_column(shape, axis, from_index, column_width=1):
    if 0 <= from_index:
        sl = slice(from_index, from_index + column_width)
    else:
        sl = slice(from_index - column_width, from_index)
    return _slice_at_axis(shape, axis, sl)


def _normalize_shape(ndarray, shape, cast_to_int=True):
    """
    Private function which does some checks and normalizes the possibly
    much simpler representations of 'pad_width', 'stat_length',
    'constant_values', 'end_values'.

    Parameters
    ----------
    narray : ndarray
        Input ndarray
    shape : {sequence, array_like, float, int}, optional
        The width of padding (pad_width), the number of elements on the
        edge of the narray used for statistics (stat_length), the constant
        value(s) to use when filling padded regions (constant_values), or the
        endpoint target(s) for linear ramps (end_values).
        ((before_1, after_1), ... (before_N, after_N)) unique number of
        elements for each axis where `N` is rank of `narray`.
        ((before, after),) yields same before and after constants for each
        axis.
        (constant,) or val is a shortcut for before = after = constant for
        all axes.
    cast_to_int : bool, optional
        Controls if values in ``shape`` will be rounded and cast to int
        before being returned.

    Returns
    -------
    normalized_shape : tuple of tuples
        val                               => ((val, val), (val, val), ...)
        [[val1, val2], [val3, val4], ...] => ((val1, val2), (val3, val4), ...)
        ((val1, val2), (val3, val4), ...) => no change
        [[val1, val2], ]                  => ((val1, val2), (val1, val2), ...)
        ((val1, val2), )                  => ((val1, val2), (val1, val2), ...)
        [[val ,     ], ]                  => ((val, val), (val, val), ...)
        ((val ,     ), )                  => ((val, val), (val, val), ...)

    """
    ndims = ndarray.ndim

    # Shortcut shape=None
    if shape is None:
        return ((None, None), ) * ndims

    # Convert any input `info` to a NumPy array
    shape_arr = np.asarray(shape)

    try:
        shape_arr = np.broadcast_to(shape_arr, (ndims, 2))
    except ValueError:
        fmt = "Unable to create correctly shaped tuple from %s"
        raise ValueError(fmt % (shape,))

    # Cast if necessary
    if cast_to_int is True:
        shape_arr = np.round(shape_arr).astype(int)

    # Convert list of lists to tuple of tuples
    return tuple(tuple(axis) for axis in shape_arr.tolist())


def _validate_lengths(narray, number_elements):
    """
    Private function which does some checks and reformats pad_width and
    stat_length using _normalize_shape.

    Parameters
    ----------
    narray : ndarray
        Input ndarray
    number_elements : {sequence, int}, optional
        The width of padding (pad_width) or the number of elements on the edge
        of the narray used for statistics (stat_length).
        ((before_1, after_1), ... (before_N, after_N)) unique number of
        elements for each axis.
        ((before, after),) yields same before and after constants for each
        axis.
        (constant,) or int is a shortcut for before = after = constant for all
        axes.

    Returns
    -------
    _validate_lengths : tuple of tuples
        int                               => ((int, int), (int, int), ...)
        [[int1, int2], [int3, int4], ...] => ((int1, int2), (int3, int4), ...)
        ((int1, int2), (int3, int4), ...) => no change
        [[int1, int2], ]                  => ((int1, int2), (int1, int2), ...)
        ((int1, int2), )                  => ((int1, int2), (int1, int2), ...)
        [[int ,     ], ]                  => ((int, int), (int, int), ...)
        ((int ,     ), )                  => ((int, int), (int, int), ...)

    """
    normshp = _normalize_shape(narray, number_elements)
    for i in normshp:
        chk = [1 if x is None else x for x in i]
        chk = [1 if x >= 0 else -1 for x in chk]
        if (chk[0] < 0) or (chk[1] < 0):
            fmt = "%s cannot contain negative values."
            raise ValueError(fmt % (number_elements,))
    return normshp


def _arange_ndarray(shape, axis, reverse=False):
    """
    Create an ndarray of `shape` with increments along specified `axis`

    Parameters
    ----------
    shape : tuple of ints
        Shape of desired array. Should be equivalent to `arr.shape` except
        `shape[axis]` which may have any positive value.
    axis : int
        Axis to increment along.
    reverse : bool
        If False, increment in a positive fashion from 1 to `shape[axis]`,
        inclusive. If True, the bounds are the same but the order reversed.

    Returns
    -------
    arr : ndarray
        Output array with `shape` that in- or decrements along the given
        `axis`.

    Notes
    -----
    The range is deliberately 1-indexed for this specific use case. Think of
    this algorithm as broadcasting `np.arange` to a single `axis` of an
    arbitrarily shaped ndarray.
    """
    init_shape = (1,) * axis + (shape[axis],) + (1,) * (len(shape) - axis - 1)
    if not reverse:
        arr = np.arange(1, shape[axis] + 1)
    else:
        arr = np.arange(shape[axis], 0, -1)
    arr = arr.reshape(init_shape)
    for i, dim in enumerate(shape):
        if arr.shape[i] != dim:
            arr = arr.repeat(dim, axis=i)
    return arr


def _round_if_needed(arr, dtype):
    """
    Rounds arr inplace if destination dtype is integer.

    Parameters
    ----------
    arr : ndarray
        Input array.
    dtype : dtype
        The dtype of the destination array.

    """
    if np.issubdtype(dtype, np.integer):
        arr.round(out=arr)


def _pad_empty(arr, pad_widths):
    """Pad array with undefined values.

    Parameters
    ----------
    arr : ndarray
        Array to grow.
    pad_widths : sequence of tuple[int, int]
        Pad width on both sides for each dimension in `arr`.

    Returns
    -------
    padded : ndarray
        Larger array with undefined values in padded areas.
    """
    # Allocate grown array
    new_shape = tuple(
        size + left + right
        for size, (left, right) in zip(arr.shape, pad_widths)
    )
    padded = np.empty(new_shape, dtype=arr.dtype)

    # Copy old array into correct space
    old_area = tuple(
        slice(left, left + size)
        for size, (left, right) in zip(arr.shape, pad_widths)
    )
    padded[old_area] = arr

    return padded


def _set_generic(arr, axis, pad_index, values):
    pad_area = _slice_pad_area(arr.shape, axis, pad_index)
    arr[pad_area] = values


def _set_edge(arr, axis, pad_index):
    edge_slice = _slice_column(arr.shape, axis, pad_index)
    edge_arr = arr[edge_slice].repeat(abs(pad_index), axis=axis)
    _set_generic(arr, axis, pad_index, edge_arr)


def _set_linear_ramp(arr, axis, pad_index, end_value):
    pad_shape = arr.shape[:axis] + (abs(pad_index),) + arr.shape[(axis + 1):]
    reverse = True if 0 <= pad_index else False
    linear_ramp = _arange_ndarray(tuple(pad_shape), axis, reverse)

    edge_slice = _slice_column(arr.shape, axis, pad_index)
    edge_arr = arr[edge_slice].repeat(abs(pad_index), axis=axis)

    slope = (end_value - edge_arr) / float(abs(pad_index))
    linear_ramp = linear_ramp * slope
    linear_ramp += edge_arr
    _round_if_needed(linear_ramp, arr.dtype)

    _set_generic(arr, axis, pad_index,
                 linear_ramp.astype(arr.dtype, copy=False))


def pad(array, pad_width, mode, stat_length=None, constant_values=0,
        end_values=0, reflect_type="even", **kwargs):
    if not np.asarray(pad_width).dtype.kind == 'i':
        raise TypeError('`pad_width` must be of integral type.')

    array = np.asarray(array)
    pad_width = _validate_lengths(array, pad_width)

    # Create array with final shape and original values
    padded = _pad_empty(array, pad_width)

    if hasattr(mode, "__call__"):
        # Drop back to old, slower np.apply_along_axis mode for user-supplied
        # vector function
        function = mode
        # Set missing pad values
        for iaxis in range(array.ndim):
            np.apply_along_axis(function,
                                iaxis,
                                padded,
                                pad_width[iaxis],
                                iaxis,
                                kwargs)

    elif mode == "constant":
        constant_values = _normalize_shape(
            array, constant_values, cast_to_int=False)
        for axis, ((left_pad, right_pad), (left_value, right_value)) \
                in enumerate(zip(pad_width, constant_values)):
            _set_generic(padded, axis, left_pad, left_value)
            _set_generic(padded, axis, -right_pad, right_value)

    elif mode == "edge":
        for axis, (left_pad, right_pad) in enumerate(pad_width):
            _set_edge(padded, axis, left_pad)
            _set_edge(padded, axis, -right_pad)

    elif mode == "linear_ramp":
        end_values = _normalize_shape(array, end_values, cast_to_int=False)
        for axis, ((left_pad, right_pad), (left_value, right_value)) \
                in enumerate(zip(pad_width, end_values)):
            _set_linear_ramp(padded, axis, left_pad, left_value)
            _set_linear_ramp(padded, axis, -right_pad, right_value)

    elif mode == "maximum":
        pass

    elif mode == "minimum":
        pass

    elif mode == "mean":
        pass

    elif mode == "median":
        pass

    elif mode == "reflect":
        pass

    elif mode == "symmetric":
        pass

    elif mode == "wrap":
        pass

    else:
        if isinstance(mode, np.compat.basestring):
            raise ValueError("unknown mode '{}'".format(mode))
        else:
            raise TypeError("mode must be string or callable, was type {}"
                            .format(type(mode)))

    return padded
