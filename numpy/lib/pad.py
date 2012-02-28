#!/usr/bin/env python
"""The pad.py module contains a group of functions to pad values onto the edges
of an n-dimensional array.
"""

# Make sure this line is here such that epydoc 3 can parse the docstrings for
# auto-generated documentation.
__docformat__ = "restructuredtext en"

#===imports======================
import numpy as np

#===globals======================

#---other---
__all__ = [
           'pad_minimum',
           'pad_maximum',
           'pad_mean',
           'pad_median',
           'pad_linear_ramp',
           'pad_reflect',
           'pad_constant',
           'pad_wrap',
           'pad_edge',
           ]


########
# Exception classes


class PadWidthWrongNumberOfValues(Exception):
    '''
    Error class for the wrong number of parameters to define the pad width.
    '''
    def __init__(self, rnk, pdw):
        self.rnk = rnk
        self.pdw = pdw

    def __str__(self):
        return """

            For pad_width should get a list/tuple with length
            equal to rank (%i) of lists/tuples with length of 2.
            Instead have %s
            """ % (self.rnk, self.pdw)


class NegativePadWidth(Exception):
    '''
    Error class for the negative pad width.
    '''
    def __str__(self):
        return "\n\nCannot have negative values for the pad_width."


class IncorrectKeywordValue(Exception):
    '''
    Error class for the negative pad width.
    '''
    def __init__(self, keyword, value):
        self.keyword = keyword
        self.value = str(value)

    def __str__(self):
        return """

            The keyword '%s' cannot have the value '%s'.
            """ % (self.keyword, self.value)


########
# Private utility functions.


def _create_vector(vector, pad_tuple, before_val, after_val):
    '''
    Private function which creates the padded vector to pad_mean, pad_maximum,
    pad_minimum, and pad_median.
    '''
    vector[:pad_tuple[0]] = before_val
    if pad_tuple[1] > 0:
        vector[-pad_tuple[1]:] = after_val
    return vector


def _normalize_shape(vector, shape):
    '''
    Private function to normalize all the possible ways to input a tuple of
    tuples to define before and after vectors or the stat_len used.

    int                               => ((int, int), (int, int), ...)
    [[int1, int2], [int3, int4], ...] => ((int1, int2), (int3, int4), ...)
    ((int1, int2), (int3, int4), ...) => no change
    [[int1, int2], ]                  => ((int1, int2), (int1, int2), ...]
    ((int1, int2), )                  => ((int1, int2), (int1, int2), ...)
    [[int ,     ), )                  => ((int, int), (int, int), ...)
    ((int ,     ), )                  => ((int, int), (int, int), ...)

    The length of the returned tuple is the rank of vector.
    '''
    normshp = None
    shapelen = len(np.shape(vector))
    if (isinstance(shape, int)):
        normshp = ((shape, shape), ) * shapelen
    elif (isinstance(shape, (tuple, list))
            and isinstance(shape[0], (tuple, list))
            and len(shape) == shapelen):
        normshp = shape
    elif (isinstance(shape, (tuple, list))
            and isinstance(shape[0], (int, float, long))
            and len(shape) == 1):
        normshp = ((shape[0], shape[0]), ) * shapelen
    elif (isinstance(shape, (tuple, list))
            and isinstance(shape[0], (int, float, long))
            and len(shape) == 2):
        normshp = (shape, ) * shapelen
    return normshp


def _validate_pad_width(vector, pad_width):
    '''
    Private function which does some checks and reformats the pad_width
    tuple.
    '''
    shapelen = len(np.shape(vector))
    normshp = _normalize_shape(vector, pad_width)
    if normshp == None:
        raise PadWidthWrongNumberOfValues(shapelen, pad_width)
    for i in normshp:
        if len(i) != 2:
            raise PadWidthWrongNumberOfValues(shapelen, normshp)
        if i[0] < 0 or i[1] < 0:
            raise NegativePadWidth()
    return normshp


def _loop_across(matrix, pad_width, function, **kwds):
    '''
    Private function to prepare the data for the np.apply_along_axis command
    to move through the matrix.
    '''
    nmatrix = np.array(matrix)
    pad_width = _validate_pad_width(nmatrix, pad_width)
    # Need to only normalize particular keywords.
    for i in kwds:
        if i in ['stat_len', 'end_values', 'constant_values']:
            kwds[i] = _normalize_shape(nmatrix, kwds[i])
    rank = range(len(nmatrix.shape))
    total_dim_increase = [np.sum(pad_width[i]) for i in rank]
    offset_slices = [slice(pad_width[i][0],
                           pad_width[i][0] + nmatrix.shape[i])
                     for i in rank]
    new_shape = np.array(nmatrix.shape) + total_dim_increase
    newmat = np.zeros(new_shape).astype(nmatrix.dtype)
    newmat[offset_slices] = nmatrix

    original_shape = []
    for iaxis in rank:
        np.apply_along_axis(function,
                            iaxis,
                            newmat,
                            pad_width[iaxis],
                            iaxis,
                            kwds)
        original_shape.append(slice(pad_width[iaxis][0], -pad_width[iaxis][1]))
    return newmat


def _create_stat_vectors(vector, pad_tuple, iaxis, kwds):
    '''
    Returns the portion of the vector required for any statistic.
    '''
    pt1 = -pad_tuple[1]
    if pt1 == 0:
        pt1 = None
    sbvec = vector[pad_tuple[0]:pt1]
    savec = vector[pad_tuple[0]:pt1]
    if kwds['stat_len']:
        stat_len = kwds['stat_len'][iaxis]
        sbvec = np.arange(1)
        savec = np.arange(1)
        if pad_tuple[0] > 0:
            sbvec = vector[pad_tuple[0]:pad_tuple[0] + stat_len[0]]
        if pad_tuple[1] > 0:
            savec = vector[-pad_tuple[1] - stat_len[1]:-pad_tuple[1]]
    return (sbvec, savec)


def _maximum(vector, pad_tuple, iaxis, kwds):
    '''
    Private function to calculate the before/after vectors.
    '''
    sbvec, savec = _create_stat_vectors(vector, pad_tuple, iaxis, kwds)

    bvec = np.zeros(pad_tuple[0])
    avec = np.zeros(pad_tuple[1])
    bvec[:] = max(sbvec)
    avec[:] = max(savec)
    return _create_vector(vector, pad_tuple, bvec, avec)


def _minimum(vector, pad_tuple, iaxis, kwds):
    '''
    Private function to calculate the before/after vectors.
    '''
    sbvec, savec = _create_stat_vectors(vector, pad_tuple, iaxis, kwds)

    bvec = np.zeros(pad_tuple[0])
    avec = np.zeros(pad_tuple[1])
    bvec[:] = min(sbvec)
    avec[:] = min(savec)
    return _create_vector(vector, pad_tuple, bvec, avec)


def _median(vector, pad_tuple, iaxis, kwds):
    '''
    Private function to calculate the before/after vectors.
    '''
    sbvec, savec = _create_stat_vectors(vector, pad_tuple, iaxis, kwds)

    bvec = np.zeros(pad_tuple[0])
    avec = np.zeros(pad_tuple[1])
    bvec[:] = np.median(sbvec)
    avec[:] = np.median(savec)
    return _create_vector(vector, pad_tuple, bvec, avec)


def _mean(vector, pad_tuple, iaxis, kwds):
    '''
    Private function to calculate the before/after vectors.
    '''
    sbvec, savec = _create_stat_vectors(vector, pad_tuple, iaxis, kwds)

    bvec = np.zeros(pad_tuple[0])
    avec = np.zeros(pad_tuple[1])
    bvec[:] = np.average(sbvec)
    avec[:] = np.average(savec)
    return _create_vector(vector, pad_tuple, bvec, avec)


def _constant(vector, pad_tuple, iaxis, kwds):
    '''
    Private function to calculate the before/after vectors.
    '''
    nconstant = kwds['constant_values'][iaxis]
    bvec = np.zeros(pad_tuple[0])
    avec = np.zeros(pad_tuple[1])
    bvec[:] = nconstant[0]
    avec[:] = nconstant[1]
    return _create_vector(vector, pad_tuple, bvec, avec)


def _linear_ramp(vector, pad_tuple, iaxis, kwds):
    '''
    Private function to calculate the before/after vectors.
    '''
    end_values = kwds['end_values'][iaxis]
    before_delta = ((vector[pad_tuple[0]] - end_values[0])
                    / float(pad_tuple[0]))
    after_delta = ((vector[-pad_tuple[1] - 1] - end_values[1])
                   / float(pad_tuple[1]))

    before_vector = np.ones((pad_tuple[0], )) * end_values[0]
    before_vector = before_vector.astype(vector.dtype)
    for i in range(len(before_vector)):
        before_vector[i] = before_vector[i] + i * before_delta

    after_vector = np.ones((pad_tuple[1], )) * end_values[1]
    after_vector = after_vector.astype(vector.dtype)
    for i in range(len(after_vector)):
        after_vector[i] = after_vector[i] + i * after_delta
    after_vector = after_vector[::-1]

    return _create_vector(vector, pad_tuple, before_vector, after_vector)


def _reflect(vector, pad_tuple, iaxis, kwds):
    '''
    Private function to calculate the before/after vectors.
    '''
    before_vector = (vector[pad_tuple[0] + 1:2 * pad_tuple[0] + 1])[::-1]
    after_vector = (vector[-2 * pad_tuple[1] - 1: -pad_tuple[1] - 1])[::-1]
    if kwds['reflect_type'] == 'even':
        pass
    elif kwds['reflect_type'] == 'odd':
        before_vector = 2 * vector[pad_tuple[0]] - before_vector
        after_vector = 2 * vector[-pad_tuple[-1] - 1] - after_vector
    else:
        raise IncorrectKeywordValue('reflect_type', kwds['reflect_type'])
    return _create_vector(vector, pad_tuple, before_vector, after_vector)


def _wrap(vector, pad_tuple, iaxis, kwds):
    '''
    Private function to calculate the before/after vectors.
    '''
    before_vector = vector[-(pad_tuple[1] + pad_tuple[0]):-pad_tuple[1]]
    after_vector = vector[pad_tuple[0]:pad_tuple[0] + pad_tuple[1]]
    return _create_vector(vector, pad_tuple, before_vector, after_vector)


def _edge(vector, pad_tuple, iaxis, kwds):
    '''
    Private function to calculate the before/after vectors.
    '''
    bvec = np.zeros(pad_tuple[0])
    avec = np.zeros(pad_tuple[1])
    bvec[:] = vector[pad_tuple[0]]
    avec[:] = vector[-pad_tuple[1] - 1]
    return _create_vector(vector, pad_tuple, bvec, avec)

########
# Public functions


def pad_maximum(matrix, pad_width=(1, ), stat_len=None):
    """
    Pads with the maximum value of all or part of the vector along each
    axis.

    Parameters
    ----------
    matrix : array_like of rank N
        Input array
    pad_width : {tuple of N tuples(before, after), tuple(pad,), int}, optional
        How many values padded to each end of the vector for each axis.
        ((before, after),) * np.rank(`matrix`)
        (pad,) or int is a shortcut for before = after = pad for all axes
        Default is (1, ).
    stat_len : {tuple of N tuples(before, after), tuple(len,), int}, optional
        How many values at each end of vector to determine the statistic.
        ((before_len, after_len),) * np.rank(`matrix`)
        (len,) or int is a shortcut for before = after = len for all dimensions
        ``None`` uses the entire vector.
        Default is ``None``.

    Returns
    -------
    out : ndarray of rank N
        Padded array.

    See Also
    --------
    pad_minimum
    pad_median
    pad_mean
    pad_constant
    pad_linear_ramp
    pad_reflect
    pad_wrap

    Examples
    --------
    >>> a = [1, 2, 3, 4, 5]
    >>> np.lib.pad_maximum(a, (2,))
    array([5, 5, 1, 2, 3, 4, 5, 5, 5])

    >>> np.lib.pad_maximum(a, (1, 7))
    array([5, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5])

    >>> np.lib.pad_maximum(a, (0, 7))
    array([1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5])

    """
    return _loop_across(matrix, pad_width, _maximum, stat_len=stat_len)


def pad_minimum(matrix, pad_width=(1, ), stat_len=None):
    """
    Pads with the minimum value of all or part of the vector along each
    axis.

    Parameters
    ----------
    matrix : array_like of rank N
        Input array
    pad_width : {tuple of N tuples(before, after), tuple(both,), int}, optional
        How many values padded to each end of the vector for each axis.
        ((before, after),) * np.rank(`matrix`)
        (pad,) or int is a shortcut for before = after = pad for all axes
        Default is (1, ).
    stat_len : {tuple of N tuples(before, after), tuple(both,), int}, optional
        How many values at each end of vector to determine the statistic.
        ((before_len, after_len),) * np.rank(`matrix`)
        (len,) or int is a shortcut for before = after = len for all dimensions
        ``None`` uses the entire vector.
        Default is ``None``.

    Returns
    -------
    out : ndarray of rank N
        Padded array.

    See Also
    --------
    pad_maximum
    pad_median
    pad_mean
    pad_constant
    pad_linear_ramp
    pad_reflect
    pad_wrap

    Examples
    --------
    >>> a = [1, 2, 3, 4, 5, 6]
    >>> np.lib.pad_minimum(a, (2,))
    array([1, 1, 1, 2, 3, 4, 5, 6, 1, 1])

    >>> np.lib.pad_minimum(a, (4, 2))
    array([1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 1, 1])

    >>> a = [[1,2], [3,4]]
    >>> np.lib.pad_minimum(a, ((3, 2), (2, 3)))
    array([[1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [3, 3, 3, 4, 3, 3, 3],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1]])

    """
    return _loop_across(matrix, pad_width, _minimum, stat_len=stat_len)


def pad_median(matrix, pad_width=(1, ), stat_len=None):
    """
    Pads with the median value of all or part of the vector along each axis.

    Parameters
    ----------
    matrix : array_like of rank N
        Input array
    pad_width : {tuple of N tuples(before, after), tuple(both,), int}, optional
        How many values padded to each end of the vector for each axis.
        ((before, after),) * np.rank(`matrix`)
        (pad,) or int is a shortcut for before = after = pad for all axes
        Default is (1, ).
    stat_len : {tuple of N tuples(before, after), tuple(both,), int}, optional
        How many values at each end of vector to determine the statistic.
        ((before_len, after_len),) * np.rank(`matrix`)
        (len,) or int is a shortcut for before = after = len for all dimensions
        ``None`` uses the entire vector.
        Default is ``None``.

    Returns
    -------
    out : ndarray of rank N
        Padded array.

    See Also
    --------
    pad_maximum
    pad_minimum
    pad_mean
    pad_constant
    pad_linear_ramp
    pad_reflect
    pad_wrap

    Examples
    --------
    >>> a = [1, 2, 3, 4, 5]
    >>> np.lib.pad_median(a, (2,))
    array([3, 3, 1, 2, 3, 4, 5, 3, 3])

    >>> np.lib.pad_median(a, (4, 0))
    array([3, 3, 3, 3, 1, 2, 3, 4, 5])

    """
    return _loop_across(matrix, pad_width, _median, stat_len=stat_len)


def pad_mean(matrix, pad_width=(1, ), stat_len=None):
    """
    Pads with the mean value of all or part of the vector along each axis.

    Parameters
    ----------
    matrix : array_like of rank N
        Input array
    pad_width : {tuple of N tuples(before, after), tuple(both,), int}, optional
        How many values padded to each end of the vector for each axis.
        ((before, after),) * np.rank(`matrix`)
        (pad,) or int is a shortcut for before = after = pad for all axes
        Default is (1, ).
    stat_len : {tuple of N tuples(before, after), tuple(both,), int}, optional
        How many values at each end of vector to determine the statistic.
        ((before_len, after_len),) * np.rank(`matrix`)
        (len,) or int is a shortcut for before = after = len for all dimensions
        ``None`` uses the entire vector.
        Default is ``None``.

    Returns
    -------
    out : ndarray of rank N
        Padded array.

    See Also
    --------
    pad_maximum
    pad_minimum
    pad_median
    pad_constant
    pad_linear_ramp
    pad_reflect
    pad_wrap

    Examples
    --------
    >>> a = [1, 2, 3, 4, 5]
    >>> np.lib.pad_mean(a, (2,))
    array([3, 3, 1, 2, 3, 4, 5, 3, 3])

    """
    return _loop_across(matrix, pad_width, _mean, stat_len=stat_len)


def pad_constant(matrix, pad_width=(1, ), constant_values=(0, )):
    """
    Pads with a constant value.

    Parameters
    ----------
    matrix : array_like of rank N
        Input array
    pad_width : {tuple of N tuples(before, after), tuple(both,), int}, optional
        How many values padded to each end of the vector for each axis.
        ((before, after),) * np.rank(`matrix`)
        (pad,) or int is a shortcut for before = after = pad for all axes
        Default is (1, ).
    constant_values : {tuple of N tuples(before, after), tuple(both,), int},
                      optional
        The values to set the padded values to.
        ((before_len, after_len),) * np.rank(`matrix`)
        (len,) or int is a shortcut for before = after = len for all dimensions
        ``None`` uses the entire vector.
        Default is ``None``.

    Returns
    -------
    out : ndarray of rank N
        Padded array.

    See Also
    --------
    pad_maximum
    pad_minimum
    pad_median
    pad_mean
    pad_linear_ramp
    pad_reflect
    pad_wrap

    Examples
    --------
    >>> a = [1, 2, 3, 4, 5]
    >>> np.lib.pad_constant(a, (2,3), (4,6))
    array([4, 4, 1, 2, 3, 4, 5, 6, 6, 6])

    """
    return _loop_across(matrix,
                         pad_width,
                         _constant,
                         constant_values=constant_values)


def pad_linear_ramp(matrix, pad_width=(1, ), end_values=(0, )):
    """
    Pads with the linear ramp between end_value and the begining/end of the
    vector along each axis.

    Parameters
    ----------
    matrix : array_like of rank N
        Input array
    pad_width : {tuple of N tuples(before, after), tuple(both,), int}, optional
        How many values padded to each end of the vector for each axis.
        ((before, after),) * np.rank(`matrix`)
        (pad,) or int is a shortcut for before = after = pad for all axes
        Default is (1, ).
    end_values: {tuple of N tuples(before, after), tuple(both,), int}, optional
        What value should the padded values end with.
        ((before_len, after_len),) * np.rank(`matrix`)
        (len,) or int is a shortcut for before = after = len for all dimensions
        ``None`` uses the entire vector.
        Default is ``None``.

    Returns
    -------
    out : ndarray of rank N
        Padded array.

    See Also
    --------
    pad_maximum
    pad_minimum
    pad_median
    pad_mean
    pad_constant
    pad_reflect
    pad_wrap

    Examples
    --------
    >>> a = [1, 2, 3, 4, 5]
    >>> np.lib.pad_linear_ramp(a, (2,3), (5,-4))
    array([ 5,  3,  1,  2,  3,  4,  5,  2, -1, -4])

    """
    return _loop_across(matrix, pad_width, _linear_ramp, end_values=end_values)


def pad_reflect(matrix, pad_width=(1, ), reflect_type='even'):
    """
    Pads with the reflection of the vector mirrored on the first and last
    values of the vector along each axis.

    Parameters
    ----------
    matrix : array_like of rank N
        Input array
    pad_width : {tuple of N tuples(before, after), tuple(both,), int}, optional
        How many values padded to each end of the vector for each axis.
        ((before, after),) * np.rank(`matrix`)
        (pad,) or int is a shortcut for before = after = pad for all axes
        Default is (1, ).
    reflect_type : str {'even', 'odd'}, optional
        The 'even' style is the default with an unaltered reflection around
        the edge value.  For the 'odd' style, the extented part of the array
        is created by subtracting the reflected values from two times the edge
        value.

    Returns
    -------
    out : ndarray of rank N
        Padded array.

    See Also
    --------
    pad_maximum
    pad_minimum
    pad_median
    pad_mean
    pad_constant
    pad_linear_ramp
    pad_wrap

    Examples
    --------
    >>> a = [1, 2, 3, 4, 5]
    >>> np.lib.pad_reflect(a, (2,3))
    array([3, 2, 1, 2, 3, 4, 5, 4, 3, 2])
    >>> np.lib.pad_reflect(a, (2,3), reflect_type='odd')
    array([-1,  0,  1,  2,  3,  4,  5,  6,  7,  8])

    """
    return _loop_across(matrix, pad_width, _reflect, reflect_type=reflect_type)


def pad_wrap(matrix, pad_width=(1, )):
    """
    Pads with the wrap of the vector along the axis.  The first values are
    used to pad the end and the end values are used to pad the beginning.

    Parameters
    ----------
    matrix : array_like of rank N
        Input array
    pad_width : {tuple of N tuples(before, after), tuple(both,), int}, optional
        How many values padded to each end of the vector for each axis.
        ((before, after),) * np.rank(`matrix`)
        (pad,) or int is a shortcut for before = after = pad for all axes
        Default is (1, ).

    Returns
    -------
    out : ndarray of rank N
        Padded array.

    See Also
    --------
    pad_maximum
    pad_minimum
    pad_median
    pad_mean
    pad_constant
    pad_linear_ramp
    pad_reflect
    pad_wrap

    Examples
    --------
    >>> a = [1, 2, 3, 4, 5]
    >>> np.lib.pad_wrap(a, (2,3))
    array([4, 5, 1, 2, 3, 4, 5, 1, 2, 3])

    """
    return _loop_across(matrix, pad_width, _wrap)


def pad_edge(matrix, pad_width=(1, )):
    """
    Pads with the edge values of the vector along the axis.

    Parameters
    ----------
    matrix : array_like of rank N
        Input array
    pad_width : {tuple of N tuples(before, after), tuple(both,), int}, optional
        How many values padded to each end of the vector for each axis.
        ((before, after),) * np.rank(`matrix`)
        (pad,) or int is a shortcut for before = after = pad for all axes
        Default is (1, ).

    Returns
    -------
    out : ndarray of rank N
        Padded array.

    See Also
    --------
    pad_maximum
    pad_minimum
    pad_median
    pad_mean
    pad_constant
    pad_linear_ramp
    pad_reflect
    pad_wrap

    Examples
    --------
    >>> a = [1, 2, 3, 4, 5]
    >>> np.lib.pad_edge(a, (2,3))
    array([1, 1, 1, 2, 3, 4, 5, 5, 5, 5])

    """
    return _loop_across(matrix, pad_width, _edge)
########


if __name__ == '__main__':
    '''
    This section is just used for testing.  Really you should only import
    this module.
    '''

    import numpy.lib.pad as pad
    import doctest
    doctest.testmod()
    ARR = np.arange(100)
    print ARR
    print pad.pad_median(ARR, (3, ))
    print pad.pad_constant(ARR, (25, 20), (10, 20))
    ARR = np.arange(30)
    ARR = np.reshape(ARR, (6, 5))
    print pad.pad_mean(ARR, pad_width=((2, 3), (3, 2)), stat_len=(3, ))
