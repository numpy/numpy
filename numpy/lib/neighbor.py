
"""
The neighbor module contains a function to calculate a new value
based on neighboring values.
"""

import numpy as np

__all__ = ['neighbor']



################################################################################
# Public functions


def shapeintersect(intersect_arrays):
    """
    Returns the maximally overlaping intersection of input arrays.

    After alignment the intersection is determined by expanding outward from
    the origin/alignment elements to the common maximal extent in all
    arrays in each dimension.

    Parameters
    ----------
    intersect_arrays : sequence of two item sequences
        A sequence of (array, origin) pairs.  The origin index represent the
        alignment element between all of the arrays.

    Returns
    -------
    shapeintersect : list of ndarry
        List of length equal to the number of input arrays.  Each returned
        array is the maximal intesected shape from the input arrays, in turn.

    Examples
    --------
    >>> a = (((7, 3, 5, 1, 5), (1,)), ((4, 5, 6), (1,)))
    >>> shapeintersect(a)
    [array([7, 3, 5]), array([4, 5, 6])]

    >>> a = (((7, 3, 5, 1, 5), (1,)), ((4, 5, 6), (2,)))
    >>> shapeintersect(a)
    [array([7, 3]), array([5, 6])]
    """

    # Bring in just one array to test rank.  Since all arrays must have the
    # same rank, it doesn't matter which one I use.
    tarr = np.array(intersect_arrays[0][0])
    tori = np.array(intersect_arrays[0][1], dtype='int')
    # Might be a better way.  Move through the arrays to collect the minimum
    # lengths between the origins and edges of the arrays.
    for narr, norigin in intersect_arrays:
        oarr = np.array(narr)
        oori = np.array(norigin, dtype='int')
        if tarr.ndim != oarr.ndim:
            raise ValueError((
                    'All input arrays should have the same number '
                    'of dimensions.'))
        if len(tori) != len(oori):
            raise ValueError((
                    'All origins should have the same length.'))
        if len(oori) != oarr.ndim:
            raise ValueError((
                    'The length of the origin should be equal to the number '
                    'of dimensions in array.'))
        # Test to make sure origin is within array by taking advantage of index
        # error.  Error checking is all the following statement is used for.
        a = oarr[oori]
        try:
            lindex = np.minimum(lindex, oori)
        except NameError:
            lindex = np.copy(oori)
        try:
            uindex = np.minimum(uindex, np.array(oarr.shape) - oori)
        except NameError:
            uindex = np.array(oarr.shape) - oori

    # Now lets make the real calculation of the slice indices
    accum_arrays = []
    for narr, norigin in intersect_arrays:
        oarr = np.array(narr)
        oori = np.array(norigin, dtype='int')
        lslice = oori - lindex
        uslice = oori + uindex
        slc = [slice(l, u) for l, u in zip(lslice, uslice)]
        accum_arrays.append(oarr[slc])
    return accum_arrays


def neighbor(array, weight, func, mode='same', weight_origin=None,
        pad='reflect', constant_values=0, end_values=0, stat_length=None,
        reflect_type='even', **kwargs):
    """
    Calculates a new value from a collection of neighboring elements.

    Parameters
    ----------
    array : array_like
        Input array
    weight : array_like
        Array that selects and weights values to be part of the neighborhood.
        The center of the `weight` array is aligned with each element in
        `array`.  The center is exactly determined if the axis is odd, and
        approximated as the nearest lower indexed element if even.  The center
        can be specified with the `weight_origin` keyword.

        Identical rank to `array`.
    func : function, optional
        Function `func` must accept a single rank 1 ndarray as an argument (and
        optionally `kwargs`) and return a scalar.
    mode : {'same', 'valid'}, optional
        'same'     Mode same is the default and returns the same shape as
                   `array`. Boundary effects are still visible, though padding
                   can be used to minimize.
        'valid'    Mode valid returns will return an array of
                   shape = `array.shape` - `weight.shape` + 1
                   The element is only returned where `array` and
                   `weight` overlap completely.
    weight_origin : array_like
        The origin is the element that is aligned with each element in `array`.
        The rank of `weight_origin` must equal the rank of the `weight` array,
        and in each dimension must be less than the length of the corresponding
        dimension in `array`.
    pad : {str, function, None}, optional
        Default is 'reflect'.

        See np.pad documentation for details.

        One of the following string values or a user supplied function.

        'constant'      Pads with a constant value.
        'edge'          Pads with the edge values of array.
        'linear_ramp'   Pads with the linear ramp between end_value and the
                        array edge value.
        'maximum'       Pads with the maximum value of all or part of the
                        vector along each axis.
        'mean'          Pads with the mean value of all or part of the
                        vector along each axis.
        'median'        Pads with the median value of all or part of the
                        vector along each axis.
        'minimum'       Pads with the minimum value of all or part of the
                        vector along each axis.
        'reflect'       Pads with the reflection of the vector mirrored on
                        the first and last values of the vector along each
                        axis.
        'symmetric'     Pads with the reflection of the vector mirrored
                        along the edge of the array.
        'wrap'          Pads with the wrap of the vector along the axis.
                        The first values are used to pad the end and the
                        end values are used to pad the beginning.
        <function>      Padding function, see Notes in the np.pad function.
    constant_values : {sequence, int}
        Optional for `pad` of 'constant'.  Defaults to 0.
        See np.pad documentation for details.
    end_values : {sequence, int}
        Optional for `pad` of 'linear_ramp'. Defaults to 0.
        See np.pad documentation for details.
    stat_length : {sequence, int}
        Optional for `pad` in ['minimum', 'maximum', 'mean', 'median'].
        Defaults to using the entire vector along each axis.
        See np.pad documentation for details.
    reflect_type : {sequence, int}
        Optional for `pad` in ['reflect', 'symmetric'].  Defaults to 'even'.
        See np.pad documentation for details.
    kwargs : varies
        Any additional keyword arguments that should be passed to `func`.

    Returns
    -------
    neighbor : ndarray
        Array with values calculated from neighbors.  Rank will be equivalent
        to rank of `array`, but shape will depend on `mode` keyword.

    Examples
    --------
    >>> a = [7, 3, 5, 1, 5]
    >>> np.lib.neighbor(a, [1]*3, np.mean, pad=None)
    array([5, 5, 3, 3, 3])

    >>> np.lib.neighbor(a, [1]*3, np.min)
    array([3, 3, 1, 1, 1])

    >>> np.lib.neighbor(a, [1]*3, np.max, mode='valid')
    array([7, 5, 5])

    >>> np.lib.neighbor(a, [1]*3, np.max)
    array([7, 7, 5, 5, 5])

    The 'neighbor' and 'convolution' concepts are related.
    See the next two examples.

    >>> np.convolve(a, [2]*3, mode='same')
    array([20, 30, 18, 22, 12])

    To emulate convolution with 'neighbor' use the np.sum function.

    >>> np.lib.neighbor(a, [2]*3, np.sum, pad='constant')
    array([20, 30, 18, 22, 12])

    Example of 'neighbor' support for n-dimensional arrays:

    >>> a = np.arange(25).reshape((5, 5))
    >>> kernel = np.ones((3, 3))
    >>> np.lib.neighbor(a, kernel, np.min)
    array([[ 0,  0,  1,  2,  3],
           [ 0,  0,  1,  2,  3],
           [ 5,  5,  6,  7,  8],
           [10, 10, 11, 12, 13],
           [15, 15, 16, 17, 18]])

    Create your own function to use in the neighborhood.
    >>> def counter(arr):
    ...    # here 'arr' will always be a 1-D array.
    ...    return len(arr)

    >>> np.lib.neighbor(a, kernel, counter, pad=None)
    array([[4, 6, 6, 6, 4],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [4, 6, 6, 6, 4]])
    """

    narray = np.array(array)
    weight_array = np.array(weight)

    # Error checks
    if narray.ndim != weight_array.ndim:
        raise ValueError((
                'The "array" (rank %d) and "weight" (rank %d) arguments '
                'must have the same rank.') %
                (narray.ndim, weight_array.ndim))

    if mode not in ['same', 'valid']:
        raise ValueError(
                'The "mode" keyword must be one of ["same", "valid"]')

    # Find the origin of the weight array
    if weight_origin:
        weight_center = np.array(weight_origin)
        if np.any(np.array(weight_center.shape) > np.array(weight_array.shape)):
            raise ValueError((
                'The "weight_origin" keyword must less than "weight" array '
                'in each dimension. Instead %s !< %s') %
                (weight_center.shape, weight_array.shape))
        if np.any(weight_center < 0):
            raise ValueError((
                'All values in array_like "weight_origin" keyword must be'
                'positive. Instead have %s') %
                (weight_center))
    else:
        weight_center = (np.array(weight_array.shape) - 1)/2

    if mode == 'valid':
        pad = None

    if pad:
        # Pad width dependent on weight.
        pad_width = zip(weight_center, 
                        np.array(weight_array.shape) - weight_center - 1)
        # Different modes only allow certain keywords.
        padkw = {}
        if pad == 'constant':
            padkw['constant_values'] = constant_values
        if pad == 'linear_ramp':
            padkw['linear_ramp'] = end_values
        if pad in ['mean', 'minimum', 'maximum', 'median']:
            padkw['stat_length'] = stat_length
        if pad in ['reflect', 'symmetric']:
            padkw['reflect_type'] = reflect_type
        narray = np.pad(narray, pad_width, mode=pad, **padkw)

    if pad or mode == 'valid':
        array_offset = weight_center
    else:
        array_offset = np.zeros(weight_center.shape)

    if mode == 'valid':
        # Need to convert element numbering to slice notation, by subtracting 1
        # for the lower index and adding 1 to the upper index.  Then need to
        # make sure the the lower index is greater than zero, and the upper
        # index is less than shape of the input array.
        lindex = np.array(weight_center) - 1
        lindex[lindex < 0] = 0
        uindex = np.array(weight_array.shape) - lindex
        uindex = np.where(uindex > weight_array.shape,
                          np.array(weight_array.shape) + 1, uindex)
        valid_slices = tuple(slice(l, u) for l, u in zip(lindex, uindex))
    else:
        valid_slices = (slice(None))

    out_array = np.empty_like(np.array(array)[valid_slices])
    for mindex in np.ndindex(*out_array.shape):
        offset_index = array_offset + np.array(mindex)
        tarray, tweight = shapeintersect(((narray, offset_index),
                                          (weight_array, weight_center)))
        tarray = tarray*tweight
        if kwargs:
            out_array[mindex] = func(np.extract(np.isfinite(tweight), tarray),
                                     kwargs)
        else:
            out_array[mindex] = func(np.extract(np.isfinite(tweight), tarray))
    return out_array
