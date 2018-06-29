"""
Functions to pad values onto the edges of an n-dimensional array.
"""


from __future__ import division, absolute_import, print_function

import numpy as np


__all__ = ['pad']


def _linear_ramp(ndim, axis, start, stop, size, reverse=False, dtype=None):
    """
    Create a linear ramp of `size` in `axis` with `ndim`.

    This algorithm behaves like a vectorized version of `numpy.linspace`.
    The resulting linear ramp is broadcastable to any array that matches the
    ramp in `shape[axis]` and `ndim`.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the resulting array. All dimensions except
        the one specified by `axis` will have the size 1.
    axis : int
        The dimension that contains the linear ramp of `size`.
    start : int or ndarray
        The starting value(s) of the linear ramp. If given as an array, its
        size must match `size`.
    stop : int or ndarray
        The stop value(s) (not included!) of the linear ramp. If given as an
        array, its size must match `size`.
    size : int
        The number of elements in the linear ramp.
    reverse : bool
        If False, increment in a positive fashion, otherwise decrement.

    Returns
    -------
    ramp : ndarray
        Output array that in- or decrements along the given `axis`.

    Examples
    --------
    >>> _linear_ramp(ndim=2, axis=0, start=np.arange(3), stop=10, size=2)
    array([[0. , 1. , 2. ],
           [5. , 5.5, 6. ]])
    """
    # Create initial ramp
    ramp = np.arange(size)
    if reverse:
        ramp = ramp[::-1]

    # Make sure, that ramp is broadcastable
    init_shape = (1,) * axis + (size,) + (1,) * (ndim - axis - 1)
    ramp = ramp.reshape(init_shape)

    # And scale to given start and stop values
    gain = (stop - start) / float(size)
    ramp = ramp * gain
    ramp += start

    if dtype:
        _round_if_needed(ramp, dtype)
    return ramp


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
    Construct tuple of slices to slice an array in the given dimension.

    Parameters
    ----------
    shape : tuple
        Shape of the array to slice as returned by its `shape` attribute.
    axis : int
        The axis to which `sl` is applied. All other dimensions are left
        "unsliced".
    sl : slice
        The slice for the given dimension.

    Returns
    -------
    sl : tuple of slices
        A tuple with slices matching `shape` in length.

    Examples
    --------
    >>> _slice_at_axis((3, 4, 5), 1, slice(None, 3, -1))
    (slice(None, None, None), slice(None, 3, -1), slice(None, None, None))
    """
    slice_tup = (slice(None),)
    return slice_tup * axis + (sl,) + slice_tup * (len(shape) - axis - 1)


def _pad_simple(array, pad_width, fill_value=None):
    """Pad array on all sides with either a single value or undefined values.

    Parameters
    ----------
    array : ndarray
        Array to grow.
    pad_width : sequence of tuple[int, int]
        Pad width on both sides for each dimension in `arr`.
    fill_value : scalar, optional
        If provided the padded area is filled with this value, otherwise
        the pad area left undefined.

    Returns
    -------
    padded : ndarray
        The padded array..
    old_area : tuple
        A tuple of slices pointing to the area of the original array.
    """
    # Allocate grown array
    new_shape = tuple(
        left + size + right
        for size, (left, right) in zip(array.shape, pad_width)
    )
    padded = np.empty(new_shape, dtype=array.dtype)

    if fill_value is not None:
        padded.fill(fill_value)

    # Copy old array into correct space
    old_area = tuple(
        slice(left, left + size)
        for size, (left, right) in zip(array.shape, pad_width)
    )
    padded[old_area] = array

    return padded, old_area


def _set_pad_area(padded, axis, index_pair, value_pair):
    """
    Set empty-padded area in given dimension.

    Parameters
    ----------
    padded : ndarray
        Array with the pad area which is modified inplace.
    axis : int
        Dimension with the pad area to set.
    index_pair : (int, int)
        Pair of indices that mark the end (or start) of the pad area on both 
        sides in the given dimension.
    value_pair : tuple of scalars or ndarrays
        Values inserted into the pad area on each side. It must match or be 
        broadcastable to the shape of `arr`.
    """
    if index_pair[0] > 0:
        # Set pad values on the left
        left_slice = _slice_at_axis(
            padded.shape, axis, slice(None, index_pair[0]))
        padded[left_slice] = value_pair[0]

    if index_pair[1] > 0:
        # Set pad values on the right
        right_slice = _slice_at_axis(
            padded.shape, axis, slice(-index_pair[1], None))
        padded[right_slice] = value_pair[1]


def _get_edges(padded, axis, index_pair):
    """
    Retrieve edge values from empty-padded array in given dimension.
    
    Parameters
    ----------
    padded : ndarray
        Empty-padded array.
    axis : int
        Dimension in which the edges are considered.
    index_pair : (int, int)
        Pair of indices that mark the end (or start) of the pad area on both
        sides in the given dimension.

    Returns
    -------
    left_edge, right_edge : ndarray
        Edge values of the valid area in `padded` in the given dimension.
    """
    if index_pair[0] > 0:
        left_slice = _slice_at_axis(
            padded.shape, axis, slice(index_pair[0], index_pair[0] + 1))
        left_edge = padded[left_slice]
    else:
        left_edge = np.array([], dtype=padded.dtype)

    if index_pair[1] > 0:
        right_slice = _slice_at_axis(
            padded.shape, axis, slice(-index_pair[1] - 1, -index_pair[1]))
        right_edge = padded[right_slice]
    else:
        right_edge = np.array([], dtype=padded.dtype)

    return left_edge, right_edge


def _get_linear_ramps(padded, axis, index_pair, end_value_pair):
    """
    Construct linear ramps for empty-padded array in given dimension.

    Parameters
    ----------
    padded : ndarray
        Empty-padded array.
    axis : int
        Dimension in which the ramps are constructed.
    index_pair : (int, int)
        Pair of indices that mark the end (or start) of the pad area on both
        sides in the given dimension.
    end_value_pair : (scalar, scalar)
        End values for the linear ramps which form the edge of the fully padded
        array. These values are included in the linear ramps.

    Returns
    -------
    left_ramp, right_ramp : ndarray
        Linear ramps to set on both sides of `padded`.
    """
    edge_pair = _get_edges(padded, axis, index_pair)

    if index_pair[0] > 0:
        left_ramp = _linear_ramp(
            padded.ndim, axis,
            start=end_value_pair[0], stop=edge_pair[0], size=index_pair[0],
            dtype=padded.dtype, reverse=False
        )
    else:
        left_ramp = np.array([], dtype=padded.dtype)

    if index_pair[1] > 0:
        right_ramp = _linear_ramp(
            padded.ndim, axis,
            start=end_value_pair[1], stop=edge_pair[1], size=index_pair[1],
            dtype=padded.dtype, reverse=True
        )
    else:
        right_ramp = np.array([], dtype=padded.dtype)

    return left_ramp, right_ramp


def _get_stats(padded, axis, index_pair, length_pair, stat_func):
    """
    Calculate statistic for the empty-padded array in given dimnsion.

    Parameters
    ----------
    padded : ndarray
        Empty-padded array.
    axis : int
        Dimension in which the statistic is calculated.
    index_pair : (int, int)
        Pair of indices that mark the end (or start) of the pad area on both
        sides in the given dimension.
    length_pair : 2-element sequence of None or int
        Gives the number of values in valid area from each side that is
        taken into account when calculating the statistic. If None the entire
        valid area in `padded` is considered.
    stat_func : function
        Function to compute statistic. The expected signature is
        ``stat_func(x: ndarray, axis: int, keepdims: bool) -> ndarray``.

    Returns
    -------
    left_stat, right_stat : ndarray
        Calculated statistic for both sides of `padded`.
    """
    max_length = padded.shape[axis] - index_pair[0] - index_pair[1]

    if index_pair[0] > 0:
        if length_pair[0] is not None and length_pair[0] < max_length:
            left_length = length_pair[0]
        else:
            left_length = max_length
        left_slice = _slice_at_axis(
            padded.shape, axis,
            slice(index_pair[0], index_pair[0] + left_length)
        )
        left_chunk = padded[left_slice]
        left_stat = stat_func(left_chunk, axis=axis, keepdims=True)
        _round_if_needed(left_stat, padded.dtype)
    else:
        left_stat = np.array([], dtype=padded.dtype)

    if index_pair[1] > 0:
        if length_pair[1] is not None and length_pair[1] < max_length:
            right_length = length_pair[1]
        else:
            right_length = max_length
        right_slice = _slice_at_axis(
            padded.shape, axis,
            slice(-index_pair[1] - right_length, -index_pair[1])
        )
        right_chunk = padded[right_slice]
        right_stat = stat_func(right_chunk, axis=axis, keepdims=True)
        _round_if_needed(right_stat, padded.dtype)
    else:
        right_stat = np.array([], dtype=padded.dtype)

    return left_stat, right_stat


def _set_reflect_both(padded, axis, index_pair, method, include_edge=False):
    """
    Pad `axis` of `arr` with reflection.

    Parameters
    ----------
    padded : ndarray
        Input array of arbitrary shape.
    axis : int
        Axis along which to pad `arr`.
    index_pair : tuple of ints, length 2
        Padding to (prepend, append) along `axis`.
    method : str
        Controls method of reflection; options are 'even' or 'odd'.
    include_edge : bool
        If true, edge value is included in reflection, otherwise the edge
        value forms the symmetric axis to the reflection.

    Returns
    -------
    pad_amt : tuple of ints, length 2
        New index positions of padding to do along the `axis`. If these are
        both 0, padding is done in this dimension. See notes on why this is
        necessary.

    Notes
    -----
    The modes 'reflect', 'symmetric', and 'wrap' must be padded with a
    single function, lest the indexing tricks in non-integer multiples of the
    original shape would violate repetition in the final iteration.
    """
    left_pad, right_pad = index_pair
    period = padded.shape[axis] - right_pad - left_pad

    if include_edge:
        offset = 1  # Edge is included, we need to offset the pad amount by 1
    else:
        offset = 0  # Edge is not included, no need to offset pad amount
        period -= 1  # But decrease size of chunk because edge is omitted

    if left_pad > 0:
        # Pad with reflected values on left side
        # First slice chunk from right side of the non-pad area.
        # Use min(period, left_pad) to ensure that chunk is not larger than
        # pad area
        left_index = left_pad - offset
        left_slice = _slice_at_axis(
            padded.shape, axis,
            slice(left_index + min(period, left_pad), left_index, -1)
        )
        left_chunk = padded[left_slice]

        if method == "odd":
            edge_slice = _slice_at_axis(
                padded.shape, axis, slice(left_pad, left_pad + 1)
            )
            left_chunk = 2 * padded[edge_slice] - left_chunk

        if left_pad > period:
            # Chunk is smaller than pad area
            left_pad -= period
            pad_area = _slice_at_axis(
                padded.shape, axis, slice(left_pad, left_pad + period))
        else:
            # Chunk matches pad area
            pad_area = _slice_at_axis(padded.shape, axis, slice(None, left_pad))
            left_pad = 0
        padded[pad_area] = left_chunk

    if right_pad > 0:
        # Pad with reflected values on left side
        # First slice chunk from right side of the non-pad area.
        # Use min(period, right_pad) to ensure that chunk is not larger than
        # pad area
        right_index = -right_pad + offset - 2
        right_slice = _slice_at_axis(
            padded.shape, axis,
            slice(right_index, right_index - min(period, right_pad), -1)
        )
        right_chunk = padded[right_slice]

        if method == "odd":
            edge_slice = _slice_at_axis(
                padded.shape, axis, slice(-right_pad - 1, -right_pad))
            right_chunk = 2 * padded[edge_slice] - right_chunk

        if right_pad > period:
            # Chunk is smaller than pad area
            right_pad -= period
            pad_area = _slice_at_axis(
                padded.shape, axis, slice(-right_pad - period, -right_pad))

        else:
            # Chunk matches pad area
            pad_area = _slice_at_axis(padded.shape, axis, slice(-right_pad, None))
            right_pad = 0
        padded[pad_area] = right_chunk

    return left_pad, right_pad


def _set_wrap_both(padded, axis, index_pair):
    """
    Pad `axis` of `arr` with wrapped values.

    Parameters
    ----------
    padded : ndarray
        Input array of arbitrary shape.
    axis : int
        Axis along which to pad `arr`.
    index_pair : tuple of ints, length 2
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
    left_pad, right_pad = index_pair
    period = padded.shape[axis] - right_pad - left_pad

    # If the current dimension of `arr` doesn't contain enough valid values
    # (not part of the undefined pad area) we need to pad multiple times.
    # Each time the pad area shrinks on both sides which is communicated with
    # these variables.
    new_left_pad = 0
    new_right_pad = 0

    if left_pad > 0:
        # Pad with wrapped values on left side
        # First slice chunk from right side of the non-pad area.
        # Use min(period, left_pad) to ensure that chunk is not larger than
        # pad area
        right_slice = _slice_at_axis(
            padded.shape, axis,
            slice(-right_pad - min(period, left_pad), -right_pad)
        )
        right_chunk = padded[right_slice]

        if left_pad > period:
            # Chunk is smaller than pad area
            pad_area = _slice_at_axis(
                padded.shape, axis, slice(left_pad - period, left_pad))
            new_left_pad = left_pad - period
        else:
            # Chunk matches pad area
            pad_area = _slice_at_axis(padded.shape, axis, slice(None, left_pad))
        padded[pad_area] = right_chunk

    if right_pad > 0:
        # Pad with wrapped values on right side
        # First slice chunk from left side of the non-pad area.
        # Use min(period, right_pad) to ensure that chunk is not larger than
        # pad area
        left_slice = _slice_at_axis(
            padded.shape, axis,
            slice(left_pad, left_pad + min(period, right_pad))
        )
        left_chunk = padded[left_slice]

        if right_pad > period:
            # Chunk is smaller than pad area
            pad_area = _slice_at_axis(
                padded.shape, axis, slice(-right_pad, -right_pad + period))
            new_right_pad = right_pad - period
        else:
            # Chunk matches pad area
            pad_area = _slice_at_axis(padded.shape, axis, slice(-right_pad, None))
        padded[pad_area] = left_chunk

    return new_left_pad, new_right_pad


def _as_pairs(x, ndim, as_index=False, assert_number=False):
    """
    Broadcast `x` to an array with the shape (`ndim`, 2).

    A helper function for `pad` that prepares and validates arguments like
    `pad_width` to be iterated in pairs.

    Parameters
    ----------
    x : {None, scalar, array-like}
        The object to broadcast to the shape (`ndim`, 2). None is only allowed
        as a special case if `as_index` is True.
    ndim : int
        Number of pairs the broadcasted `x` will have.
    as_index : bool, optional
        If `x` is not None, try to round each element of `x` to a non-negative
        integer.
    assert_number : bool, optional
        Raise a TypeError if the dtype of `x` is not a subdtype of `np.number`.

    Returns
    -------
    pairs : nested structure with shape (`ndim`, 2)
        The broadcasted version of `x`.

    Raises
    ------
    TypeError
        If `as_index` is True and `x` is not None and can't be rounded to an
        array of integer type.
        Or if `assert_number` is True and the dtype of `x` is not a subdtype
        of `np.number`.
    ValueError
        If `as_index` is True and `x` contains negative elements.
        Or if `x` is not broadcastable to the shape (`ndim`, 2).
    """
    if as_index and x is None:
        # Pass through None as a special case for indices
        return ((None, None),) * ndim

    x = np.asarray(x)
    if assert_number and not np.issubdtype(x.dtype, np.number):
        raise TypeError("keyword argument must subdtype of np.number for "
                        "the current mode")

    if as_index:
        try:
            x = x.round().astype(np.intp, copy=False)
        except AttributeError:
            raise TypeError("can't cast `x` to int")

    if x.size == 1:
        # Single value case
        x = x.ravel()  # Reduce superfluous dimensions
        if as_index and x < 0:
            raise ValueError("index can't contain negative values")
        return ((x[0], x[0]),) * ndim

    if x.size == 2 and ndim == 2 and x.shape != (2, 1):
        # Pair value case, but except special case when each dimension has a
        # single value which should be broadcasted to a pair
        x = x.ravel()  # Reduce superfluous dimensions
        if as_index and x[0] < 0 and x[1] < 0:
            raise ValueError("index can't contain negative values")
        return ((x[0], x[1]),) * ndim

    if as_index and x.min() < 0:
        raise ValueError("index can't contain negative values")
    try:
        return np.broadcast_to(x, (ndim, 2)).tolist()
    except ValueError:
        raise ValueError("unable to broadcast '{}' to shape {}"
                         .format(x, (ndim, 2)))


def pad(array, pad_width, mode, **kwargs):
    """
    Pad an array.

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
        'empty'
            Pads with undefined values.
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
    array = np.asarray(array)
    pad_width = np.asarray(pad_width)

    if not pad_width.dtype.kind == 'i':
        raise TypeError('`pad_width` must be of integral type.')

    # Broadcast to shape (array.ndim, 2)
    pad_width = _as_pairs(pad_width, array.ndim, as_index=True)

    if not isinstance(mode, np.compat.basestring):
        # Old behavior: Use user-supplied function with np.apply_along_axis
        function = mode
        # Create a new zero padded array
        padded, _ = _pad_simple(array, pad_width, fill_value=0)
        # And apply along each axis
        for axis in range(padded.ndim):
            np.apply_along_axis(function, axis, padded, pad_width[axis],
                                axis, kwargs)
        return padded

    # Make sure that no unsupported keywords were passed for the current mode
    allowed_kwargs = {
        'empty': [], 'edge': [], 'wrap': [],
        'constant': ['constant_values'],
        'linear_ramp': ['end_values'],
        'maximum': ['stat_length'],
        'mean': ['stat_length'],
        'median': ['stat_length'],
        'minimum': ['stat_length'],
        'reflect': ['reflect_type'],
        'symmetric': ['reflect_type'],
    }
    unsupported_kwargs = set(kwargs) - set(allowed_kwargs[mode])
    if unsupported_kwargs:
        raise ValueError("unsupported keyword arguments for mode '{}': {}"
                         .format(mode, unsupported_kwargs))

    stat_functions = {"maximum": np.max, "minimum": np.min,
                      "mean": np.mean, "median": np.median}

    # Create array with final shape and original values
    # (padded area is undefined)
    padded, _ = _pad_simple(array, pad_width)
    # And prepare iteration over all dimensions
    # (zipping may be more readable than using enumerate)
    axes = range(padded.ndim)

    if array.size == 0 and (mode != "constant" or mode != "empty"):
        # Deal with special case: only modes "constant" and "empty" can extend
        # empty axes, all other modes depend on `array` not being empty.
        for axis, index_pair in zip(axes, pad_width):
            if (
                array.shape[axis] == 0
                and (index_pair[0] > 0 or index_pair[1] > 0)
            ):
                raise ValueError(
                    "can't extend empty axis {} using modes other than "
                    "'constant' or 'empty'".format(axis)
                )
        # If pad values in empty axis were zero as well, then _pad_simple
        # already returned the correct result
        return padded

    if mode == "constant":
        values = kwargs.get("constant_values", 0)
        values = _as_pairs(values, padded.ndim, assert_number=True)
        for axis, index_pair, value_pair in zip(axes, pad_width, values):
            _set_pad_area(padded, axis, index_pair, value_pair)

    elif mode == "edge":
        for axis, index_pair in zip(axes, pad_width):
            edge_pair = _get_edges(padded, axis, index_pair)
            _set_pad_area(padded, axis, index_pair, edge_pair)

    elif mode == "linear_ramp":
        end_values = kwargs.get("end_values", 0)
        end_values = _as_pairs(end_values, padded.ndim, assert_number=True)
        for axis, index_pair, value_pair in zip(axes, pad_width, end_values):
            ramp_pair = _get_linear_ramps(padded, axis, index_pair, value_pair)
            _set_pad_area(padded, axis, index_pair, ramp_pair)

    elif mode in stat_functions.keys():
        func = stat_functions[mode]
        length = kwargs.get("stat_length", None)
        length = _as_pairs(length, padded.ndim, as_index=True)
        for axis, index_pair, length_pair in zip(axes, pad_width, length):
            stat_pair = _get_stats(padded, axis, index_pair, length_pair, func)
            _set_pad_area(padded, axis, index_pair, stat_pair)

    elif mode == "reflect" or mode == "symmetric":
        method = kwargs.get("reflect_type", "even")
        include_edge = True if mode == "symmetric" else False
        for axis, (left_index, right_index) in zip(axes, pad_width):
            if array.shape[axis] == 1 and (left_index > 0 or right_index > 0):
                # Extending singleton dimension for 'reflect' is legacy
                # behavior; it really should raise an error.
                edge_pair = _get_edges(padded, axis, (left_index, right_index))
                _set_pad_area(padded, axis, (left_index, right_index), edge_pair)
                continue

            while left_index > 0 or right_index > 0:
                # Iteratively pad until dimension is filled with reflected
                # values. This is necessary if the pad area is larger than
                # the length of the original values in the current dimension.
                left_index, right_index = _set_reflect_both(
                    padded, axis, (left_index, right_index),
                    method, include_edge
                )

    elif mode == "wrap":
        for axis, (left_index, right_index) in zip(axes, pad_width):
            while left_index > 0 or right_index > 0:
                # Iteratively pad until dimension is filled with wrapped
                # values. This is necessary if the pad area is larger than
                # the length of the original values in the current dimension.
                left_index, right_index = _set_wrap_both(
                    padded, axis, (left_index, right_index))

    elif mode == "empty":
        pass  # Do nothing as padded is already prepared

    else:
        raise ValueError("mode '{}' is not supported".format(mode))

    return padded
