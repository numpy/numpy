"""
Functions to pad values onto the edges of an n-dimensional array.
"""


from __future__ import division, absolute_import, print_function

import numpy as np


__all__ = ['pad']


###############################################################################
# Private utility functions.


def _linear_ramp(ndim, axis, size, reverse=False):
    """
    Create a linear ramp of `size` in `axis` with `ndim`.

    This algorithm creates a 1-indexed array. The resulting linear ramp is
    broadcastable to any array that matches the ramp in `shape[axis]` and
    `ndim`.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the resulting array. All dimensions except
        the one specified by `axis` whill have the size 1.
    axis : int
        The dimension that contains the linear ramp of `size`.
    size : int
        The size of the linear ramp.
    reverse :
        If False, increment in a positive fashion from 1 to `size`, inclusive.
        If True, the bounds are the same but the order reversed.

    Returns
    -------
    arr : ndarray
        Output array with that in- or decrements along the given `axis`.
    """
    if not reverse:
        arr = np.arange(1, size + 1, 1)
    else:
        arr = np.arange(size, 0, -1)
    init_shape = (1,) * axis + (arr.size,) + (1,) * (ndim - axis - 1)
    arr = arr.reshape(init_shape)
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


def _slice_at_axis(shape, axis, sl):
    """
    Construct a slice tuple the length of shape, with sl at the specified axis
    """
    slice_tup = (slice(None),)
    return slice_tup * axis + (sl,) + slice_tup * (len(shape) - axis - 1)


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
    old_area : tuple
        A tuple of slices pointing to the area of the original array.
    """
    # Allocate grown array
    new_shape = tuple(
        left + size + right
        for size, (left, right) in zip(arr.shape, pad_widths)
    )
    padded = np.empty(new_shape, dtype=arr.dtype)

    # Copy old array into correct space
    old_area = tuple(
        slice(left, left + size)
        for size, (left, right) in zip(arr.shape, pad_widths)
    )
    padded[old_area] = arr

    return padded, old_area


def _set_generic(arr, axis, pad_index, values):
    """
    Set pad area with given values.

    Parameters
    ----------
    arr : ndarray
        Array with pad area which is modified inplace.
    axis : int
        Dimension with the pad area to set.
    pad_index : int
        Index that marks the end (or start) of the pad area in the given
        dimension. If >= 0 the pad area starts at index 0 and ends with this
        value, otherwise it starts at the end of the array and `pad_index` is
        treated as an index counted from the right-hand side.
    values : scalar or ndarray
        Values inserted into the pad area. It must match or be broadcastable
        to the shape of `arr`.
    """
    if pad_index == 0:
        return
    if 0 <= pad_index:
        sl = slice(0, pad_index)
    else:
        sl = slice(pad_index, arr.shape[axis])
    pad_area = _slice_at_axis(arr.shape, axis, sl)
    arr[pad_area] = values


def _set_edge(arr, axis, pad_index):
    if pad_index == 0:
        return

    if pad_index > 0:
        sl = slice(pad_index, pad_index + 1)
    else:
        sl = slice(pad_index - 1, pad_index)
    edge_slice = _slice_at_axis(arr.shape, axis, sl)

    edge_arr = arr[edge_slice].repeat(abs(pad_index), axis=axis)
    _set_generic(arr, axis, pad_index, edge_arr)


def _set_linear_ramp(arr, axis, pad_index, end_value):
    """
    Set pad area with a linear ramp from the edge to the given end value(s).

    Parameters
    ----------
    arr : ndarray
        Array with pad area which is modified inplace.
    axis : int
        Dimension with the pad area to set.
    pad_index : int
        Index that marks the end (or start) of the pad area in the given
        dimension. If >= 0 the pad area starts at index 0 and ends with this
        value, otherwise it starts at the end of the array and `pad_index` is
        treated as an index counted from the right-hand side.
    end_value : scalar
        Constant value to use. For best results should be of type `arr.dtype`;
        if not `arr.dtype` will be cast to `arr.dtype`.
    """
    if pad_index == 0:
        return

    if pad_index > 0:
        ramp = _linear_ramp(arr.ndim, axis, abs(pad_index), True)
        edge_slice = _slice_at_axis(
            arr.shape, axis, slice(pad_index, pad_index + 1))
    else:
        ramp = _linear_ramp(arr.ndim, axis, abs(pad_index), False)
        edge_slice = _slice_at_axis(
            arr.shape, axis, slice(pad_index - 1, pad_index))

    edge_arr = arr[edge_slice].repeat(abs(pad_index), axis=axis)

    # Scale linear ramp to desired linear space
    slope = (end_value - edge_arr) / float(abs(pad_index))
    ramp = ramp * slope
    ramp += edge_arr
    _round_if_needed(ramp, arr.dtype)

    _set_generic(arr, axis, pad_index, ramp.astype(arr.dtype, copy=False))


def _set_stat(arr, axis, pad_index, stat_length, stat_func):
    """Compute statistic and fill the pad area on one side.

    Parameters
    ----------
    arr : ndarray
        Array with pad area which is modified inplace.
    axis : int
        Dimension with the pad area to set.
    pad_index : int
        Index that marks the end (or start) of the pad area in the given
        dimension. If >= 0 the pad area starts at index 0 and ends with this
        value, otherwise it starts at the end of the array and `pad_index` is
        treated as an index counted from the right-hand side.
    stat_length : scalar
        Number of values at edge of each axis used to calculate the statistic
        value.
    stat_func : func
        Function used to compute the statistic.
    """
    if pad_index == 0:
        return
    if stat_length == 1:
        _set_edge(arr, axis, pad_index)
        return

    if 0 <= pad_index:
        start = pad_index
        stop = pad_index + stat_length
    else:
        start = pad_index - stat_length
        stop = pad_index
    stat_area = _slice_at_axis(arr.shape, axis, slice(start, stop))

    stats = stat_func(arr[stat_area], axis=axis, keepdims=True)
    _round_if_needed(stats, arr.dtype)
    #stats = stats.repeat(abs(pad_index), axis=axis)

    _set_generic(arr, axis, pad_index, stats)


def _set_reflect_both(arr, axis, pad_amt, method):
    """
    Pad `axis` of `arr` with reflection.

    Parameters
    ----------
    arr : ndarray
        Input array of arbitrary shape.
    axis : int
        Axis along which to pad `arr`.
    pad_amt : tuple of ints, length 2
        Padding to (prepend, append) along `axis`.
    method : str
        Controls method of reflection; options are 'even' or 'odd'.

    Returns
    -------
    pad_amt : tuple of ints, length 2
        New index positions of padding to do along the `axis`. If these are
        both 0, padding is done in this dimension. See notes on why this is
        necessary.

    Notes
    -----
    This algorithm does not pad with repetition, i.e. the edges are not
    repeated in the reflection. For that behavior, use `mode='symmetric'`.

    The modes 'reflect', 'symmetric', and 'wrap' must be padded with a
    single function, lest the indexing tricks in non-integer multiples of the
    original shape would violate repetition in the final iteration.
    """
    left_pad, right_pad = pad_amt
    period = arr.shape[axis] - right_pad - left_pad - 1

    if left_pad > 0:
        # Reflection of left area
        left_slice = _slice_at_axis(
            arr.shape, axis,
            slice(left_pad + min(period, left_pad), left_pad, -1)
        )
        left_chunk = arr[left_slice]

        if method == "odd":
            edge_slice = _slice_at_axis(
                arr.shape, axis, slice(left_pad, left_pad + 1)
            )
            left_chunk = 2 * arr[edge_slice] - left_chunk

        if left_pad > period:
            left_pad -= period
            pad_area = _slice_at_axis(
                arr.shape, axis, slice(left_pad, left_pad + period))
            arr[pad_area] = left_chunk
        else:
            _set_generic(arr, axis, left_pad, left_chunk)
            left_pad = 0

    if right_pad > 0:
        # Reflection of right area
        right_slice = _slice_at_axis(
            arr.shape, axis,
            slice(-right_pad - 2, -right_pad - min(period, right_pad) - 2, -1)
        )
        right_chunk = arr[right_slice]

        if method == "odd":
            edge_slice = _slice_at_axis(
                arr.shape, axis, slice(-right_pad - 1, -right_pad))
            right_chunk = 2 * arr[edge_slice] - right_chunk

        if right_pad > period:
            right_pad -= period
            pad_area = _slice_at_axis(
                arr.shape, axis, slice(-right_pad - period, -right_pad))
            arr[pad_area] = right_chunk
        else:
            _set_generic(arr, axis, -right_pad, right_chunk)
            right_pad = 0

    return left_pad, right_pad


def _set_symmetric_both(arr, axis, pad_amt, method):
    """
    Pad `axis` of `arr` with symmetry.

    Parameters
    ----------
    arr : ndarray
        Input array of arbitrary shape.
    axis : int
        Axis along which to pad `arr`.
    pad_amt : tuple of ints, length 2
        Padding to (prepend, append) along `axis`.
    method : str
        Controls method of symmetry; options are 'even' or 'odd'.

    Returns
    -------
    pad_amt : tuple of ints, length 2
        New index positions of padding to do along the `axis`. If these are
        both 0, padding is done in this dimension. See notes on why this is
        necessary.

    Notes
    -----
    This algorithm does not pad with repetition, i.e. the edges are not
    repeated in the reflection. For that behavior, use `mode='symmetric'`.

    The modes 'reflect', 'symmetric', and 'wrap' must be padded with a
    single function, lest the indexing tricks in non-integer multiples of the
    original shape would violate repetition in the final iteration.
    """
    left_pad, right_pad = pad_amt
    period = arr.shape[axis] - right_pad - left_pad

    if left_pad > 0:
        left_slice = _slice_at_axis(
            arr.shape, axis,
            slice(left_pad + min(period, left_pad) - 1, left_pad - 1, -1)
        )
        left_chunk = arr[left_slice]

        if method == "odd":
            edge_slice = _slice_at_axis(
                arr.shape, axis, slice(left_pad, left_pad + 1)
            )
            left_chunk = 2 * arr[edge_slice] - left_chunk

        if left_pad > period:
            left_pad -= period
            pad_area = _slice_at_axis(
                arr.shape, axis, slice(left_pad, left_pad + period))
            arr[pad_area] = left_chunk
        else:
            _set_generic(arr, axis, left_pad, left_chunk)
            left_pad = 0

    if right_pad > 0:
        right_slice = _slice_at_axis(
            arr.shape, axis,
            slice(-right_pad - 1, -right_pad - min(period, right_pad) - 1, -1)
        )
        right_chunk = arr[right_slice]

        if method == "odd":
            edge_slice = _slice_at_axis(
                arr.shape, axis, slice(-right_pad - 1, -right_pad))
            right_chunk = 2 * arr[edge_slice] - right_chunk

        if right_pad > period:
            right_pad -= period
            pad_area = _slice_at_axis(
                arr.shape, axis, slice(-right_pad - period, -right_pad))
            arr[pad_area] = right_chunk
        else:
            _set_generic(arr, axis, -right_pad, right_chunk)
            right_pad = 0

    return left_pad, right_pad


def _set_wrap_both(arr, axis, pad_amt):
    """
    Pad `axis` of `arr` with wrapped values.

    Parameters
    ----------
    arr : ndarray
        Input array of arbitrary shape.
    axis : int
        Axis along which to pad `arr`.
    pad_amt : tuple of ints, length 2
        Padding to (prepend, append) along `axis`.

    Returns
    -------
    pad_amt : tuple of ints, length 2
        New index positions of padding to do along the `axis`. If these are
        both 0, padding is done in this dimension. See notes on why this is
        necessary.

    Notes
    -----
    This algorithm does not pad with repetition, i.e. the edges are not
    repeated in the reflection. For that behavior, use `mode='symmetric'`.

    The modes 'reflect', 'symmetric', and 'wrap' must be padded with a
    single function, lest the indexing tricks in non-integer multiples of the
    original shape would violate repetition in the final iteration.
    """
    left_pad, right_pad = pad_amt
    period = arr.shape[axis] - right_pad - left_pad

    # If the current dimension of `arr` doesn't contain enough valid values
    # (not part of the undefined pad area) we need to pad multiple times.
    # Each time the pad area shrinks on both sides which is communicated with
    # these variables.
    new_left_pad = None
    new_right_pad = None

    if left_pad > 0:
        # Pad with wrapped values on left side
        # First slice chunk from right side of the non-pad area.
        # Use min(period, left_pad) to ensure that chunk is not larger than
        # pad area
        right_slice = _slice_at_axis(
            arr.shape, axis,
            slice(-right_pad - min(period, left_pad), -right_pad)
        )
        right_chunk = arr[right_slice]

        if left_pad > period:
            # Chunk is smaller than pad area
            pad_area = _slice_at_axis(
                arr.shape, axis, slice(left_pad - period, left_pad))
            arr[pad_area] = right_chunk
            new_left_pad = left_pad - period
        else:
            # Chunk matches pad area
            _set_generic(arr, axis, left_pad, right_chunk)
            new_left_pad = 0

    if right_pad > 0:
        # Pad with wrapped values on right side
        # First slice chunk from left side of the non-pad area.
        # Use min(period, right_pad) to ensure that chunk is not larger than
        # pad area
        left_slice = _slice_at_axis(
            arr.shape, axis,
            slice(left_pad, left_pad + min(period, right_pad))
        )
        left_chunk = arr[left_slice]

        if right_pad > period:
            # Chunk is smaller than pad area
            pad_area = _slice_at_axis(
                arr.shape, axis, slice(-right_pad, -right_pad + period))
            arr[pad_area] = left_chunk
            new_right_pad = right_pad - period
        else:
            # Chunk matches pad area
            _set_generic(arr, axis, -right_pad, left_chunk)
            new_right_pad = 0

    return new_left_pad, new_right_pad



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


###############################################################################
# Public functions


def pad(array, pad_width, mode, **kwargs):
    """
    Pads an array.

    Parameters
    ----------
    array : array_like of rank N
        Input array
    pad_width : {sequence, array_like, int}
        Number of values padded to the edges of each axis.
        ((before_1, after_1), ... (before_N, after_N)) unique pad widths
        for each axis.
        ((before, after),) yields same before and after pad for each axis.
        (pad,) or int is a shortcut for before = after = pad width for all
        axes.
    mode : str or function
        One of the following string values or a user supplied function.

        'constant'
            Pads with a constant value.
        'edge'
            Pads with the edge values of array.
        'linear_ramp'
            Pads with the linear ramp between end_value and the
            array edge value.
        'maximum'
            Pads with the maximum value of all or part of the
            vector along each axis.
        'mean'
            Pads with the mean value of all or part of the
            vector along each axis.
        'median'
            Pads with the median value of all or part of the
            vector along each axis.
        'minimum'
            Pads with the minimum value of all or part of the
            vector along each axis.
        'reflect'
            Pads with the reflection of the vector mirrored on
            the first and last values of the vector along each
            axis.
        'symmetric'
            Pads with the reflection of the vector mirrored
            along the edge of the array.
        'wrap'
            Pads with the wrap of the vector along the axis.
            The first values are used to pad the end and the
            end values are used to pad the beginning.
        <function>
            Padding function, see Notes.
    stat_length : sequence or int, optional
        Used in 'maximum', 'mean', 'median', and 'minimum'.  Number of
        values at edge of each axis used to calculate the statistic value.

        ((before_1, after_1), ... (before_N, after_N)) unique statistic
        lengths for each axis.

        ((before, after),) yields same before and after statistic lengths
        for each axis.

        (stat_length,) or int is a shortcut for before = after = statistic
        length for all axes.

        Default is ``None``, to use the entire axis.
    constant_values : sequence or int, optional
        Used in 'constant'.  The values to set the padded values for each
        axis.

        ((before_1, after_1), ... (before_N, after_N)) unique pad constants
        for each axis.

        ((before, after),) yields same before and after constants for each
        axis.

        (constant,) or int is a shortcut for before = after = constant for
        all axes.

        Default is 0.
    end_values : sequence or int, optional
        Used in 'linear_ramp'.  The values used for the ending value of the
        linear_ramp and that will form the edge of the padded array.

        ((before_1, after_1), ... (before_N, after_N)) unique end values
        for each axis.

        ((before, after),) yields same before and after end values for each
        axis.

        (constant,) or int is a shortcut for before = after = end value for
        all axes.

        Default is 0.
    reflect_type : {'even', 'odd'}, optional
        Used in 'reflect', and 'symmetric'.  The 'even' style is the
        default with an unaltered reflection around the edge value.  For
        the 'odd' style, the extended part of the array is created by
        subtracting the reflected values from two times the edge value.

    Returns
    -------
    pad : ndarray
        Padded array of rank equal to `array` with shape increased
        according to `pad_width`.

    Notes
    -----
    .. versionadded:: 1.7.0

    For an array with rank greater than 1, some of the padding of later
    axes is calculated from padding of previous axes.  This is easiest to
    think about with a rank 2 array where the corners of the padded array
    are calculated by using padded values from the first axis.

    The padding function, if used, should return a rank 1 array equal in
    length to the vector argument with padded values replaced. It has the
    following signature::

        padding_func(vector, iaxis_pad_width, iaxis, kwargs)

    where

        vector : ndarray
            A rank 1 array already padded with zeros.  Padded values are
            vector[:pad_tuple[0]] and vector[-pad_tuple[1]:].
        iaxis_pad_width : tuple
            A 2-tuple of ints, iaxis_pad_width[0] represents the number of
            values padded at the beginning of vector where
            iaxis_pad_width[1] represents the number of values padded at
            the end of vector.
        iaxis : int
            The axis currently being calculated.
        kwargs : dict
            Any keyword arguments the function requires.

    Examples
    --------
    >>> a = [1, 2, 3, 4, 5]
    >>> np.pad(a, (2,3), 'constant', constant_values=(4, 6))
    array([4, 4, 1, 2, 3, 4, 5, 6, 6, 6])

    >>> np.pad(a, (2, 3), 'edge')
    array([1, 1, 1, 2, 3, 4, 5, 5, 5, 5])

    >>> np.pad(a, (2, 3), 'linear_ramp', end_values=(5, -4))
    array([ 5,  3,  1,  2,  3,  4,  5,  2, -1, -4])

    >>> np.pad(a, (2,), 'maximum')
    array([5, 5, 1, 2, 3, 4, 5, 5, 5])

    >>> np.pad(a, (2,), 'mean')
    array([3, 3, 1, 2, 3, 4, 5, 3, 3])

    >>> np.pad(a, (2,), 'median')
    array([3, 3, 1, 2, 3, 4, 5, 3, 3])

    >>> a = [[1, 2], [3, 4]]
    >>> np.pad(a, ((3, 2), (2, 3)), 'minimum')
    array([[1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [3, 3, 3, 4, 3, 3, 3],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1]])

    >>> a = [1, 2, 3, 4, 5]
    >>> np.pad(a, (2, 3), 'reflect')
    array([3, 2, 1, 2, 3, 4, 5, 4, 3, 2])

    >>> np.pad(a, (2, 3), 'reflect', reflect_type='odd')
    array([-1,  0,  1,  2,  3,  4,  5,  6,  7,  8])

    >>> np.pad(a, (2, 3), 'symmetric')
    array([2, 1, 1, 2, 3, 4, 5, 5, 4, 3])

    >>> np.pad(a, (2, 3), 'symmetric', reflect_type='odd')
    array([0, 1, 1, 2, 3, 4, 5, 5, 6, 7])

    >>> np.pad(a, (2, 3), 'wrap')
    array([4, 5, 1, 2, 3, 4, 5, 1, 2, 3])

    >>> def pad_with(vector, pad_width, iaxis, kwargs):
    ...     pad_value = kwargs.get('padder', 10)
    ...     vector[:pad_width[0]] = pad_value
    ...     vector[-pad_width[1]:] = pad_value
    ...     return vector
    >>> a = np.arange(6)
    >>> a = a.reshape((2, 3))
    >>> np.pad(a, 2, pad_with)
    array([[10, 10, 10, 10, 10, 10, 10],
           [10, 10, 10, 10, 10, 10, 10],
           [10, 10,  0,  1,  2, 10, 10],
           [10, 10,  3,  4,  5, 10, 10],
           [10, 10, 10, 10, 10, 10, 10],
           [10, 10, 10, 10, 10, 10, 10]])
    >>> np.pad(a, 2, pad_with, padder=100)
    array([[100, 100, 100, 100, 100, 100, 100],
           [100, 100, 100, 100, 100, 100, 100],
           [100, 100,   0,   1,   2, 100, 100],
           [100, 100,   3,   4,   5, 100, 100],
           [100, 100, 100, 100, 100, 100, 100],
           [100, 100, 100, 100, 100, 100, 100]])
    """
    if not np.asarray(pad_width).dtype.kind == 'i':
        raise TypeError('`pad_width` must be of integral type.')

    array = np.asarray(array)
    pad_width = _validate_lengths(array, pad_width)

    allowedkwargs = {
        'constant': ['constant_values'],
        'edge': [],
        'linear_ramp': ['end_values'],
        'maximum': ['stat_length'],
        'mean': ['stat_length'],
        'median': ['stat_length'],
        'minimum': ['stat_length'],
        'reflect': ['reflect_type'],
        'symmetric': ['reflect_type'],
        'wrap': [],
        }

    kwdefaults = {
        'stat_length': None,
        'constant_values': 0,
        'end_values': 0,
        'reflect_type': 'even',
        }

    stat_functions = {
        "maximum": np.max,
        "minimum": np.min,
        "mean": np.mean,
        "median": np.median,
    }

    if isinstance(mode, np.compat.basestring):
        # Make sure have allowed kwargs appropriate for mode
        for key in kwargs:
            if key not in allowedkwargs[mode]:
                raise ValueError('%s keyword not in allowed keywords %s' %
                                 (key, allowedkwargs[mode]))

        # Set kwarg defaults
        for kw in allowedkwargs[mode]:
            kwargs.setdefault(kw, kwdefaults[kw])

        # Need to only normalize particular keywords.
        for i in kwargs:
            if i == 'stat_length':
                kwargs[i] = _validate_lengths(array, kwargs[i])
            if i in ['end_values', 'constant_values']:
                kwargs[i] = _normalize_shape(array, kwargs[i],
                                             cast_to_int=False)
    else:
        # Drop back to old, slower np.apply_along_axis mode for user-supplied
        # vector function
        function = mode

        # Create a new padded array
        rank = list(range(array.ndim))
        total_dim_increase = [np.sum(pad_width[i]) for i in rank]
        offset_slices = tuple(
            slice(pad_width[i][0], pad_width[i][0] + array.shape[i])
            for i in rank)
        new_shape = np.array(array.shape) + total_dim_increase
        newmat = np.zeros(new_shape, array.dtype)

        # Insert the original array into the padded array
        newmat[offset_slices] = array

        # This is the core of pad ...
        for iaxis in rank:
            np.apply_along_axis(function,
                                iaxis,
                                newmat,
                                pad_width[iaxis],
                                iaxis,
                                kwargs)
        return newmat

    # Create array with final shape and original values
    padded, _ = _pad_empty(array, pad_width)

    if mode == "constant":
        for axis, ((left_pad, right_pad), (left_value, right_value)) \
                in enumerate(zip(pad_width, kwargs["constant_values"])):
            _set_generic(padded, axis, left_pad, left_value)
            _set_generic(padded, axis, -right_pad, right_value)

    elif mode == "edge":
        for axis, (left_pad, right_pad) in enumerate(pad_width):
            _set_edge(padded, axis, left_pad)
            _set_edge(padded, axis, -right_pad)

    elif mode == "linear_ramp":
        for axis, ((left_pad, right_pad), (left_value, right_value)) \
                in enumerate(zip(pad_width, kwargs["end_values"])):
            _set_linear_ramp(padded, axis, left_pad, left_value)
            _set_linear_ramp(padded, axis, -right_pad, right_value)

    elif mode in stat_functions.keys():
        stat_func = stat_functions[mode]
        for axis, ((left_pad, right_pad), (length_left, length_right)) \
                in enumerate(zip(pad_width, kwargs["stat_length"])):
            max_length = array.shape[axis]

            if length_left is None:
                length_left = max_length
            elif length_left > max_length:
                length_left = max_length
            _set_stat(padded, axis, left_pad, length_left, stat_func)

            if length_right is None:
                length_right = max_length
            elif length_right > max_length:
                length_right = max_length
            _set_stat(padded, axis, -right_pad, length_right, stat_func)

    elif mode == "reflect":
        method = kwargs["reflect_type"]
        for axis, (left_pad, right_pad) in enumerate(pad_width):
            if array.shape[axis] == 0:
                # Axes with non-zero padding cannot be empty.
                if left_pad > 0 or right_pad > 0:
                    raise ValueError("There aren't any elements to reflect"
                                     " in axis {} of `array`".format(axis))
                # Skip zero padding on empty axes.
                continue

            if array.shape[axis] == 1 and (left_pad > 0 or right_pad > 0):
                # Extending singleton dimension for 'reflect' is legacy
                # behavior; it really should raise an error.
                _set_edge(padded, axis, left_pad)
                _set_edge(padded, axis, -right_pad)
                continue

            while left_pad > 0 or right_pad > 0:
                # Iteratively pad until dimension is filled with reflected
                # values. This is necessary if the pad area is larger than
                # the length of the original values in the current dimension.
                left_pad, right_pad = _set_reflect_both(
                    padded, axis, (left_pad, right_pad), method)

    elif mode == "symmetric":
        method = kwargs["reflect_type"]
        for axis, (left_pad, right_pad) in enumerate(pad_width):
            if array.shape[axis] == 0:
                # Axes with non-zero padding cannot be empty.
                if left_pad > 0 or right_pad > 0:
                    raise ValueError("There aren't any elements to reflect"
                                     " in axis {} of `array`".format(axis))
                # Skip zero padding on empty axes.
                continue

            while left_pad > 0 or right_pad > 0:
                # Iteratively pad until dimension is filled with reflected
                # values. This is necessary if the pad area is larger than
                # the length of the original values in the current dimension.
                left_pad, right_pad = _set_symmetric_both(
                    padded, axis, (left_pad, right_pad), method)

    elif mode == "wrap":
        for axis, (left_pad, right_pad) in enumerate(pad_width):
            if array.shape[axis] == 0:
                # Axes with non-zero padding cannot be empty.
                if left_pad > 0 or right_pad > 0:
                    raise ValueError("There aren't any elements to wrap"
                                     " in axis {} of `array`".format(axis))
                # Skip zero padding on empty axes.
                continue

            while left_pad > 0 or right_pad > 0:
                # Iteratively pad until dimension is filled with wrapped
                # values. This is necessary if the pad area is larger than
                # the length of the original values in the current dimension.
                left_pad, right_pad = _set_wrap_both(
                    padded, axis, (left_pad, right_pad))

    return padded
