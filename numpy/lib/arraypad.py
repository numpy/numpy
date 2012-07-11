"""
The arraypad module contains a group of functions to pad values onto the edges
of an n-dimensional array.
"""

import numpy as np

__all__ = ['pad']


################################################################################
# Private utility functions.


def _create_vector(vector, pad_tuple, before_val, after_val):
    '''
    Private function which creates the padded vector.

    Parameters
    ----------
    vector : ndarray of rank 1, length N + pad_tuple[0] + pad_tuple[1]
        Input vector including blank padded values.  `N` is the lenth of the
        original vector.
    pad_tuple : tuple
        This tuple represents the (before, after) width of the padding along
        this particular iaxis.
    before_val : scalar or ndarray of rank 1, length pad_tuple[0]
        This is the value(s) that will pad the beginning of `vector`.
    after_val : scalar or ndarray of rank 1, length pad_tuple[1]
        This is the value(s) that will pad the end of the `vector`.

    Returns
    -------
    _create_vector : ndarray
        Vector with before_val and after_val replacing the blank pad values.
    '''
    vector[:pad_tuple[0]] = before_val
    if pad_tuple[1] > 0:
        vector[-pad_tuple[1]:] = after_val
    return vector


def _normalize_shape(narray, shape):
    '''
    Private function which does some checks and normalizes the possibly
    much simpler representations of 'pad_width', 'stat_length',
    'constant_values', 'end_values'.

    Parameters
    ----------
    narray : ndarray
        Input ndarray
    shape : {sequence, int}, optional
        The width of padding (pad_width) or the number of elements on the
        edge of the narray used for statistics (stat_length).
        ((before_1, after_1), ... (before_N, after_N)) unique number of
        elements for each axis where `N` is rank of `narray`.
        ((before, after),) yields same before and after constants for each
        axis.
        (constant,) or int is a shortcut for before = after = constant for
        all axes.

    Returns
    -------
    _normalize_shape : tuple of tuples
        int                               => ((int, int), (int, int), ...)
        [[int1, int2], [int3, int4], ...] => ((int1, int2), (int3, int4), ...)
        ((int1, int2), (int3, int4), ...) => no change
        [[int1, int2], ]                  => ((int1, int2), (int1, int2), ...]
        ((int1, int2), )                  => ((int1, int2), (int1, int2), ...)
        [[int ,     ), )                  => ((int, int), (int, int), ...)
        ((int ,     ), )                  => ((int, int), (int, int), ...)
    '''
    normshp = None
    shapelen = len(np.shape(narray))
    if (isinstance(shape, int)):
        normshp = ((shape, shape), ) * shapelen
    elif (isinstance(shape, (tuple, list))
            and isinstance(shape[0], (tuple, list))
            and len(shape) == shapelen):
        normshp = shape
        for i in normshp:
            if len(i) != 2:
                fmt = "Unable to create correctly shaped tuple from %s"
                raise ValueError(fmt % (normshp,))
    elif (isinstance(shape, (tuple, list))
            and isinstance(shape[0], (int, float, long))
            and len(shape) == 1):
        normshp = ((shape[0], shape[0]), ) * shapelen
    elif (isinstance(shape, (tuple, list))
            and isinstance(shape[0], (int, float, long))
            and len(shape) == 2):
        normshp = (shape, ) * shapelen
    if normshp == None:
        fmt = "Unable to create correctly shaped tuple from %s"
        raise ValueError(fmt % (shape,))
    return normshp


def _validate_lengths(narray, number_elements):
    '''
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
        [[int1, int2], ]                  => ((int1, int2), (int1, int2), ...]
        ((int1, int2), )                  => ((int1, int2), (int1, int2), ...)
        [[int ,     ), )                  => ((int, int), (int, int), ...)
        ((int ,     ), )                  => ((int, int), (int, int), ...)
    '''
    shapelen = len(np.shape(narray))
    normshp = _normalize_shape(narray, number_elements)
    for i in normshp:
        if i[0] < 0 or i[1] < 0:
            fmt ="%s cannot contain negative values."
            raise ValueError(fmt % (number_elements,))
    return normshp


def _create_stat_vectors(vector, pad_tuple, iaxis, kwargs):
    '''
    Returns the portion of the vector required for any statistic.

    Parameters
    ----------
    vector : ndarray
        Input vector that already includes empty padded values.
    pad_tuple : tuple
        This tuple represents the (before, after) width of the padding
        along this particular iaxis.
    iaxis : int
        The axis currently being looped across.
    kwargs : keyword arguments
        Keyword arguments.  Only 'stat_length' is used.  'stat_length'
        defaults to the entire vector if not supplied.

    Return
    ------
    _create_stat_vectors : ndarray
        The values from the original vector that will be used to calculate
        the statistic.
    '''

    # Can't have 0 represent the end if a slice... a[1:0] doesnt' work
    pt1 = -pad_tuple[1]
    if pt1 == 0:
        pt1 = None

    # Default is the entire vector from the original array.
    sbvec = vector[pad_tuple[0]:pt1]
    savec = vector[pad_tuple[0]:pt1]

    if kwargs['stat_length']:
        stat_length = kwargs['stat_length'][iaxis]
        sl0 = min(stat_length[0], len(sbvec))
        sl1 = min(stat_length[1], len(savec))
        sbvec = np.arange(0)
        savec = np.arange(0)
        if pad_tuple[0] > 0:
            sbvec = vector[pad_tuple[0]:pad_tuple[0] + sl0]
        if pad_tuple[1] > 0:
            savec = vector[-pad_tuple[1] - sl1:pt1]
    return (sbvec, savec)


def _maximum(vector, pad_tuple, iaxis, kwargs):
    '''
    Private function to calculate the before/after vectors for pad_maximum.

    Parameters
    ----------
    vector : ndarray
        Input vector that already includes empty padded values.
    pad_tuple : tuple
        This tuple represents the (before, after) width of the padding
        along this particular iaxis.
    iaxis : int
        The axis currently being looped across.
    kwargs : keyword arguments
        Keyword arguments.  Only 'stat_length' is used.  'stat_length'
        defaults to the entire vector if not supplied.

    Return
    ------
    _maximum : ndarray
        Padded vector
    '''
    sbvec, savec = _create_stat_vectors(vector, pad_tuple, iaxis, kwargs)
    return _create_vector(vector, pad_tuple, max(sbvec), max(savec))


def _minimum(vector, pad_tuple, iaxis, kwargs):
    '''
    Private function to calculate the before/after vectors for pad_minimum.

    Parameters
    ----------
    vector : ndarray
        Input vector that already includes empty padded values.
    pad_tuple : tuple
        This tuple represents the (before, after) width of the padding
        along this particular iaxis.
    iaxis : int
        The axis currently being looped across.
    kwargs : keyword arguments
        Keyword arguments.  Only 'stat_length' is used.  'stat_length'
        defaults to the entire vector if not supplied.

    Return
    ------
    _minimum : ndarray
        Padded vector
    '''
    sbvec, savec = _create_stat_vectors(vector, pad_tuple, iaxis, kwargs)
    return _create_vector(vector, pad_tuple, min(sbvec), min(savec))


def _median(vector, pad_tuple, iaxis, kwargs):
    '''
    Private function to calculate the before/after vectors for pad_median.

    Parameters
    ----------
    vector : ndarray
        Input vector that already includes empty padded values.
    pad_tuple : tuple
        This tuple represents the (before, after) width of the padding
        along this particular iaxis.
    iaxis : int
        The axis currently being looped across.
    kwargs : keyword arguments
        Keyword arguments.  Only 'stat_length' is used.  'stat_length'
        defaults to the entire vector if not supplied.

    Return
    ------
    _median : ndarray
        Padded vector
    '''
    sbvec, savec = _create_stat_vectors(vector, pad_tuple, iaxis, kwargs)
    return _create_vector(vector, pad_tuple, np.median(sbvec),
                          np.median(savec))


def _mean(vector, pad_tuple, iaxis, kwargs):
    '''
    Private function to calculate the before/after vectors for pad_mean.

    Parameters
    ----------
    vector : ndarray
        Input vector that already includes empty padded values.
    pad_tuple : tuple
        This tuple represents the (before, after) width of the padding
        along this particular iaxis.
    iaxis : int
        The axis currently being looped across.
    kwargs : keyword arguments
        Keyword arguments.  Only 'stat_length' is used.  'stat_length'
        defaults to the entire vector if not supplied.

    Return
    ------
    _mean : ndarray
        Padded vector
    '''
    sbvec, savec = _create_stat_vectors(vector, pad_tuple, iaxis, kwargs)
    return _create_vector(vector, pad_tuple, np.average(sbvec),
                          np.average(savec))


def _constant(vector, pad_tuple, iaxis, kwargs):
    '''
    Private function to calculate the before/after vectors for
    pad_constant.

    Parameters
    ----------
    vector : ndarray
        Input vector that already includes empty padded values.
    pad_tuple : tuple
        This tuple represents the (before, after) width of the padding
        along this particular iaxis.
    iaxis : int
        The axis currently being looped across.
    kwargs : keyword arguments
        Keyword arguments.  Need 'constant_values' keyword argument.

    Return
    ------
    _constant : ndarray
        Padded vector
    '''
    nconstant = kwargs['constant_values'][iaxis]
    return _create_vector(vector, pad_tuple, nconstant[0], nconstant[1])


def _linear_ramp(vector, pad_tuple, iaxis, kwargs):
    '''
    Private function to calculate the before/after vectors for
    pad_linear_ramp.

    Parameters
    ----------
    vector : ndarray
        Input vector that already includes empty padded values.
    pad_tuple : tuple
        This tuple represents the (before, after) width of the padding
        along this particular iaxis.
    iaxis : int
        The axis currently being looped across.  Not used in _linear_ramp.
    kwargs : keyword arguments
        Keyword arguments. Not used in _linear_ramp.

    Return
    ------
    _linear_ramp : ndarray
        Padded vector
    '''
    end_values = kwargs['end_values'][iaxis]
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


def _reflect(vector, pad_tuple, iaxis, kwargs):
    '''
    Private function to calculate the before/after vectors for pad_reflect.

    Parameters
    ----------
    vector : ndarray
        Input vector that already includes empty padded values.
    pad_tuple : tuple
        This tuple represents the (before, after) width of the padding
        along this particular iaxis.
    iaxis : int
        The axis currently being looped across.  Not used in _reflect.
    kwargs : keyword arguments
        Keyword arguments. Not used in _reflect.

    Return
    ------
    _reflect : ndarray
        Padded vector
    '''
    # Can't have pad_tuple[1] be used in the slice if == to 0.
    if pad_tuple[1] == 0:
        after_vector = vector[pad_tuple[0]:None]
    else:
        after_vector = vector[pad_tuple[0]:-pad_tuple[1]]

    reverse = after_vector[::-1]

    before_vector = np.resize(
                        np.concatenate(
                        (after_vector[1:-1], reverse)), pad_tuple[0])[::-1]
    after_vector = np.resize(
                       np.concatenate(
                       (reverse[1:-1], after_vector)), pad_tuple[1])

    if kwargs['reflect_type'] == 'even':
        pass
    elif kwargs['reflect_type'] == 'odd':
        before_vector = 2 * vector[pad_tuple[0]] - before_vector
        after_vector = 2 * vector[-pad_tuple[-1] - 1] - after_vector
    else:
        raise ValueError("The keyword '%s' cannot have the value '%s'."
                          % ('reflect_type', kwargs['reflect_type']))
    return _create_vector(vector, pad_tuple, before_vector, after_vector)


def _symmetric(vector, pad_tuple, iaxis, kwargs):
    '''
    Private function to calculate the before/after vectors for
    pad_symmetric.

    Parameters
    ----------
    vector : ndarray
        Input vector that already includes empty padded values.
    pad_tuple : tuple
        This tuple represents the (before, after) width of the padding
        along this particular iaxis.
    iaxis : int
        The axis currently being looped across.  Not used in _symmetric.
    kwargs : keyword arguments
        Keyword arguments. Not used in _symmetric.

    Return
    ------
    _symmetric : ndarray
        Padded vector
    '''
    if pad_tuple[1] == 0:
        after_vector = vector[pad_tuple[0]:None]
    else:
        after_vector = vector[pad_tuple[0]:-pad_tuple[1]]

    before_vector = np.resize( np.concatenate( (after_vector,
        after_vector[::-1])), pad_tuple[0])[::-1]
    after_vector = np.resize( np.concatenate( (after_vector[::-1],
        after_vector)), pad_tuple[1])

    if kwargs['reflect_type'] == 'even':
        pass
    elif kwargs['reflect_type'] == 'odd':
        before_vector = 2 * vector[pad_tuple[0]] - before_vector
        after_vector = 2 * vector[-pad_tuple[1] - 1] - after_vector
    else:
        raise ValueError("The keyword '%s' cannot have the value '%s'."
                          % ('reflect_type', kwargs['reflect_type']))
    return _create_vector(vector, pad_tuple, before_vector, after_vector)


def _wrap(vector, pad_tuple, iaxis, kwargs):
    '''
    Private function to calculate the before/after vectors for pad_wrap.

    Parameters
    ----------
    vector : ndarray
        Input vector that already includes empty padded values.
    pad_tuple : tuple
        This tuple represents the (before, after) width of the padding
        along this particular iaxis.
    iaxis : int
        The axis currently being looped across.  Not used in _wrap.
    kwargs : keyword arguments
        Keyword arguments. Not used in _wrap.

    Return
    ------
    _wrap : ndarray
        Padded vector
    '''
    if pad_tuple[1] == 0:
        after_vector = vector[pad_tuple[0]:None]
    else:
        after_vector = vector[pad_tuple[0]:-pad_tuple[1]]

    before_vector = np.resize(after_vector[::-1], pad_tuple[0])[::-1]
    after_vector = np.resize(after_vector, pad_tuple[1])

    return _create_vector(vector, pad_tuple, before_vector, after_vector)


def _edge(vector, pad_tuple, iaxis, kwargs):
    '''
    Private function to calculate the before/after vectors for pad_edge.

    Parameters
    ----------
    vector : ndarray
        Input vector that already includes empty padded values.
    pad_tuple : tuple
        This tuple represents the (before, after) width of the padding
        along this particular iaxis.
    iaxis : int
        The axis currently being looped across.  Not used in _edge.
    kwargs : keyword arguments
        Keyword arguments. Not used in _edge.

    Return
    ------
    _edge : ndarray
        Padded vector
    '''
    return _create_vector(vector, pad_tuple, vector[pad_tuple[0]],
                          vector[-pad_tuple[1] - 1])


################################################################################
# Public functions


def pad(array, pad_width, mode=None, **kwargs):
    """
    Pads an array.

    Parameters
    ----------
    array : array_like of rank N
        Input array
    pad_width : {sequence, int}
        Number of values padded to the edges of each axis.
        ((before_1, after_1), ... (before_N, after_N)) unique pad widths
        for each axis.
        ((before, after),) yields same before and after pad for each axis.
        (pad,) or int is a shortcut for before = after = pad width for all
        axes.
    mode : {str, function}
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
        <function>      Padding function, see Notes.
    stat_length : {sequence, int}, optional
        Used in 'maximum', 'mean', 'median', and 'minimum'.  Number of
        values at edge of each axis used to calculate the statistic value.

        ((before_1, after_1), ... (before_N, after_N)) unique statistic
        lengths for each axis.

        ((before, after),) yields same before and after statistic lengths
        for each axis.

        (stat_length,) or int is a shortcut for before = after = statistic
        length for all axes.

        Default is ``None``, to use the entire axis.
    constant_values : {sequence, int}, optional
        Used in 'constant'.  The values to set the padded values for each
        axis.

        ((before_1, after_1), ... (before_N, after_N)) unique pad constants
        for each axis.

        ((before, after),) yields same before and after constants for each
        axis.

        (constant,) or int is a shortcut for before = after = constant for
        all axes.

        Default is 0.
    end_values : {sequence, int}, optional
        Used in 'linear_ramp'.  The values used for the ending value of the
        linear_ramp and that will form the edge of the padded array.

        ((before_1, after_1), ... (before_N, after_N)) unique end values
        for each axis.

        ((before, after),) yields same before and after end values for each
        axis.

        (constant,) or int is a shortcut for before = after = end value for
        all axes.

        Default is 0.
    reflect_type : str {'even', 'odd'}, optional
        Used in 'reflect', and 'symmetric'.  The 'even' style is the
        default with an unaltered reflection around the edge value.  For
        the 'odd' style, the extented part of the array is created by
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
    following signature:

        padding_func(vector, iaxis_pad_width, iaxis, **kwargs)

    where

        vector: ndarray
            A rank 1 array already padded with zeros.  Padded values are
            vector[:pad_tuple[0]] and vector[-pad_tuple[1]:].
        iaxis_pad_width: tuple
            A 2-tuple of ints, iaxis_pad_width[0] represents the number of
            values padded at the beginning of vector where
            iaxis_pad_width[1] represents the number of values padded at
            the end of vector.
        iaxis : int
            The axis currently being calculated.
        kwargs : misc
            Any keyword arguments the function requires.

    Examples
    --------
    >>> a = [1, 2, 3, 4, 5]
    >>> np.lib.pad(a, (2,3), 'constant', constant_values=(4,6))
    array([4, 4, 1, 2, 3, 4, 5, 6, 6, 6])

    >>> np.lib.pad(a, (2,3), 'edge')
    array([1, 1, 1, 2, 3, 4, 5, 5, 5, 5])

    >>> np.lib.pad(a, (2,3), 'linear_ramp', end_values=(5,-4))
    array([ 5,  3,  1,  2,  3,  4,  5,  2, -1, -4])

    >>> np.lib.pad(a, (2,), 'maximum')
    array([5, 5, 1, 2, 3, 4, 5, 5, 5])

    >>> np.lib.pad(a, (2,), 'mean')
    array([3, 3, 1, 2, 3, 4, 5, 3, 3])

    >>> np.lib.pad(a, (2,), 'median')
    array([3, 3, 1, 2, 3, 4, 5, 3, 3])

    >>> a = [[1,2], [3,4]]
    >>> np.lib.pad(a, ((3, 2), (2, 3)), 'minimum')
    array([[1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [3, 3, 3, 4, 3, 3, 3],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1]])

    >>> a = [1, 2, 3, 4, 5]
    >>> np.lib.pad(a, (2,3), 'reflect')
    array([3, 2, 1, 2, 3, 4, 5, 4, 3, 2])

    >>> np.lib.pad(a, (2,3), 'reflect', reflect_type='odd')
    array([-1,  0,  1,  2,  3,  4,  5,  6,  7,  8])

    >>> np.lib.pad(a, (2,3), 'symmetric')
    array([2, 1, 1, 2, 3, 4, 5, 5, 4, 3])

    >>> np.lib.pad(a, (2,3), 'symmetric', reflect_type='odd')
    array([0, 1, 1, 2, 3, 4, 5, 5, 6, 7])

    >>> np.lib.pad(a, (2,3), 'wrap')
    array([4, 5, 1, 2, 3, 4, 5, 1, 2, 3])

    >>> def padwithtens(vector, pad_width, iaxis, kwargs):
    ...     vector[:pad_width[0]] = 10
    ...     vector[-pad_width[1]:] = 10
    ...     return vector

    >>> a = np.arange(6)
    >>> a = a.reshape((2,3))

    >>> np.lib.pad(a, 2, padwithtens)
    array([[10, 10, 10, 10, 10, 10, 10],
           [10, 10, 10, 10, 10, 10, 10],
           [10, 10,  0,  1,  2, 10, 10],
           [10, 10,  3,  4,  5, 10, 10],
           [10, 10, 10, 10, 10, 10, 10],
           [10, 10, 10, 10, 10, 10, 10]])
    """


    narray = np.array(array)
    pad_width = _validate_lengths(narray, pad_width)

    modefunc = {
           'constant': _constant,
           'edge': _edge,
           'linear_ramp': _linear_ramp,
           'maximum': _maximum,
           'mean': _mean,
           'median': _median,
           'minimum': _minimum,
           'reflect': _reflect,
           'symmetric': _symmetric,
           'wrap': _wrap,
           }

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

    if isinstance(mode, str):
        function = modefunc[mode]

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
            if i == 'stat_length' and kwargs[i]:
                kwargs[i] = _validate_lengths(narray, kwargs[i])
            if i in ['end_values', 'constant_values']:
                kwargs[i] = _normalize_shape(narray, kwargs[i])
    elif mode == None:
        raise ValueError('Keyword "mode" must be a function or one of %s.' %
                          (modefunc.keys(),))
    else:
        # User supplied function, I hope
        function = mode

    # Create a new padded array
    rank = range(len(narray.shape))
    total_dim_increase = [np.sum(pad_width[i]) for i in rank]
    offset_slices = [slice(pad_width[i][0],
                           pad_width[i][0] + narray.shape[i])
                     for i in rank]
    new_shape = np.array(narray.shape) + total_dim_increase
    newmat = np.zeros(new_shape).astype(narray.dtype)

    # Insert the original array into the padded array
    newmat[offset_slices] = narray

    # This is the core of pad ...
    for iaxis in rank:
        np.apply_along_axis(function,
                            iaxis,
                            newmat,
                            pad_width[iaxis],
                            iaxis,
                            kwargs)
    return newmat

