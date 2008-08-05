# Docstrings for generated ufuncs

docdict = {}

def get(name):
    return docdict.get(name)

def add_newdoc(place, name, doc):
    docdict['.'.join((place, name))] = doc


add_newdoc('numpy.core.umath', 'absolute',
    """
    Calculate the absolute value elementwise.

    Parameters
    ----------
    x : array_like
        An array-like sequence of values or a scalar.

    Returns
    -------
    res : {ndarray, scalar}
        An ndarray containing the absolute value of
        each element in `x`.  For complex input, ``a + ib``, the
        absolute value is :math:`\\sqrt{ a^2 + b^2 }`.

        Returns a scalar for scalar input.

    Examples
    --------
    >>> x = np.array([-1.2, 1.2])
    >>> np.absolute(x)
    array([ 1.2,  1.2])
    >>> np.absolute(1.2 + 1j)
    1.5620499351813308

    Plot the function over ``[-10, 10]``:

    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(-10, 10, 101)
    >>> plt.plot(x, np.absolute(x))
    >>> plt.show()

    Plot the function over the complex plane:

    >>> xx = x + 1j * x[:, np.newaxis]
    >>> plt.imshow(np.abs(xx), extent=[-10, 10, -10, 10])
    >>> plt.show()

    """)

add_newdoc('numpy.core.umath', 'add',
    """
    Add arguments element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be added.

    Returns
    -------
    y : {ndarray, scalar}
        The sum of `x1` and `x2`, element-wise.  Returns scalar if
        both  `x1` and `x2` are scalars.

    Notes
    -----
    Equivalent to `x1` + `x2` in terms of array broadcasting.

    Examples
    --------
    >>> np.add(1.0, 4.0)
    5.0
    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> np.add(x1, x2)
    array([[  0.,   2.,   4.],
           [  3.,   5.,   7.],
           [  6.,   8.,  10.]])

    """)

add_newdoc('numpy.core.umath', 'arccos',
    """
    Trigonometric inverse cosine, element-wise.

    The inverse of `cos` so that, if ``y = cos(x)``, then ``x = arccos(y)``.

    Parameters
    ----------
    x : array_like
        `x`-coordinate on the unit circle.
        For real arguments, the domain is [-1, 1].

    Returns
    -------
    angle : {ndarray, scalar}
        The angle of the ray intersecting the unit circle at the given
        `x`-coordinate in radians [0, pi]. If `x` is a scalar then a
        scalar is returned, otherwise an array of the same shape as `x`
        is returned.

    See Also
    --------
    cos, arctan, arcsin

    Notes
    -----
    `arccos` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `cos(z) = x`. The convention is to return the
    angle `z` whose real part lies in `[0, pi]`.

    For real-valued input data types, `arccos` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arccos` is a complex analytical function that
    has branch cuts `[-inf, -1]` and `[1, inf]` and is continuous from above
    on the former and from below on the latter.

    The inverse `cos` is also known as `acos` or cos^-1.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 79. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Inverse trigonometric function",
           http://en.wikipedia.org/wiki/Arccos

    Examples
    --------
    We expect the arccos of 1 to be 0, and of -1 to be pi:

    >>> np.arccos([1, -1])
    array([ 0.        ,  3.14159265])

    Plot arccos:

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-1, 1, num=100)
    >>> plt.plot(x, np.arccos(x))
    >>> plt.axis('tight')
    >>> plt.show()

    """)

add_newdoc('numpy.core.umath', 'arccosh',
    """
    Inverse hyperbolic cosine, elementwise.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    out : {ndarray, scalar}
        Array of the same shape and dtype as `x`.

    Notes
    -----
    `arccosh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `cosh(z) = x`. The convention is to return the
    `z` whose imaginary part lies in `[-pi, pi]` and the real part in
    ``[0, inf]``.

    For real-valued input data types, `arccosh` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arccosh` is a complex analytical function that
    has a branch cut `[-inf, 1]` and is continuous from above on it.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 86. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Inverse hyperbolic function",
           http://en.wikipedia.org/wiki/Arccosh

    Examples
    --------
    >>> np.arccosh([np.e, 10.0])
    array([ 1.65745445,  2.99322285])

    """)

add_newdoc('numpy.core.umath', 'arcsin',
    """
    Inverse sine elementwise.

    Parameters
    ----------
    x : array_like
      `y`-coordinate on the unit circle.

    Returns
    -------
    angle : {ndarray, scalar}
      The angle of the ray intersecting the unit circle at the given
      `y`-coordinate in radians ``[-pi, pi]``. If `x` is a scalar then
      a scalar is returned, otherwise an array is returned.

    See Also
    --------
    sin, arctan, arctan2

    Notes
    -----
    `arcsin` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `sin(z) = x`. The convention is to return the
    angle `z` whose real part lies in `[-pi/2, pi/2]`.

    For real-valued input data types, `arcsin` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arcsin` is a complex analytical function that
    has branch cuts `[-inf, -1]` and `[1, inf]` and is continuous from above
    on the former and from below on the latter.

    The inverse sine is also known as `asin` or ``sin^-1``.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 79. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Inverse trigonometric function",
           http://en.wikipedia.org/wiki/Arcsin

    Examples
    --------
    >>> np.arcsin(1)     # pi/2
    1.5707963267948966
    >>> np.arcsin(-1)    # -pi/2
    -1.5707963267948966
    >>> np.arcsin(0)
    0.0

    """)

add_newdoc('numpy.core.umath', 'arcsinh',
    """
    Inverse hyperbolic sine elementwise.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    out : ndarray
        Array of of the same shape as `x`.

    Notes
    -----
    `arcsinh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `sinh(z) = x`. The convention is to return the
    `z` whose imaginary part lies in `[-pi/2, pi/2]`.

    For real-valued input data types, `arcsinh` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arccos` is a complex analytical function that
    has branch cuts `[1j, infj]` and `[-1j, -infj]` and is continuous from
    the right on the former and from the left on the latter.

    The inverse hyperbolic sine is also known as `asinh` or ``sinh^-1``.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 86. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Inverse hyperbolic function",
           http://en.wikipedia.org/wiki/Arcsinh

    Examples
    --------
    >>> np.arcsinh(np.array([np.e, 10.0]))
    array([ 1.72538256,  2.99822295])

    """)

add_newdoc('numpy.core.umath', 'arctan',
    """
    Trigonometric inverse tangent, element-wise.

    The inverse of tan, so that if ``y = tan(x)`` then
    ``x = arctan(y)``.

    Parameters
    ----------
    x : {array_like, scalar}
        Input values.  `arctan` is applied to each element of `x`.

    Returns
    -------
    out : {ndarray, scalar}
        Out has the same shape as `x`.  Its real part is
        in ``[-pi/2, pi/2]``. It is a scalar if `x` is a scalar.

    See Also
    --------
    arctan2 : Calculate the arctan of y/x.

    Notes
    -----
    `arctan` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `tan(z) = x`. The convention is to return the
    angle `z` whose real part lies in `[-pi/2, pi/2]`.

    For real-valued input data types, `arctan` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arctan` is a complex analytical function that
    has branch cuts `[1j, infj]` and `[-1j, -infj]` and is continuous from the
    left on the former and from the right on the latter.

    The inverse tangent is also known as `atan` or ``tan^-1``.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 79. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Inverse trigonometric function",
           http://en.wikipedia.org/wiki/Arctan

    Examples
    --------
    We expect the arctan of 0 to be 0, and of 1 to be :math:`\\pi/4`:

    >>> np.arctan([0, 1])
    array([ 0.        ,  0.78539816])

    >>> np.pi/4
    0.78539816339744828

    Plot arctan:

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-10, 10)
    >>> plt.plot(x, np.arctan(x))
    >>> plt.axis('tight')
    >>> plt.show()

    """)

add_newdoc('numpy.core.umath', 'arctan2',
    """
    Elementwise arc tangent of ``x1/x2`` choosing the quadrant correctly.

    The quadrant (ie. branch) is chosen so that ``arctan2(x1, x2)``
    is the signed angle in radians between the line segments
    ``(0,0) - (1,0)`` and ``(0,0) - (x2,x1)``. This function is defined
    also for `x2` = 0.

    `arctan2` is not defined for complex-valued arguments.

    Parameters
    ----------
    x1 : array-like, real-valued
        y-coordinates.
    x2 : array-like, real-valued
        x-coordinates. `x2` must be broadcastable to match the shape of `x1`,
        or vice versa.

    Returns
    -------
    angle : array-like
        Array of angles in radians, in the range ``[-pi, pi]``.

    See Also
    --------
    arctan, tan

    Notes
    -----
    `arctan2` is identical to the `atan2` function of the underlying
    C library. The following special values are defined in the C standard [2]:

    ====== ====== ================
    `x1`   `x2`   `arctan2(x1,x2)`
    ====== ====== ================
    +/- 0  +0     +/- 0
    +/- 0  -0     +/- pi
     > 0   +/-inf +0 / +pi
     < 0   +/-inf -0 / -pi
    +/-inf +inf   +/- (pi/4)
    +/-inf -inf   +/- (3*pi/4)
    ====== ====== ================

    Note that +0 and -0 are distinct floating point numbers.

    References
    ----------
    .. [1] Wikipedia, "atan2",
           http://en.wikipedia.org/wiki/Atan2
    .. [2] ISO/IEC standard 9899:1999, "Programming language C", 1999.

    Examples
    --------
    Consider four points in different quadrants:

    >>> x = np.array([-1, +1, +1, -1])
    >>> y = np.array([-1, -1, +1, +1])
    >>> np.arctan2(y, x) * 180 / np.pi
    array([-135.,  -45.,   45.,  135.])

    Note the order of the parameters. `arctan2` is defined also when `x2` = 0
    and at several other special points, obtaining values in
    the range ``[-pi, pi]``:

    >>> np.arctan2([1., -1.], [0., 0.])
    array([ 1.57079633, -1.57079633])
    >>> np.arctan2([0., 0., np.inf], [+0., -0., np.inf])
    array([ 0.        ,  3.14159265,  0.78539816])

    """)

add_newdoc('numpy.core.umath', 'arctanh',
    """
    Inverse hyperbolic tangent elementwise.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    out : ndarray
        Array of the same shape as `x`.

    Notes
    -----
    `arctanh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `tanh(z) = x`. The convention is to return the
    `z` whose imaginary part lies in `[-pi/2, pi/2]`.

    For real-valued input data types, `arctanh` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arctanh` is a complex analytical function that
    has branch cuts `[-1, -inf]` and `[1, inf]` and is continuous from
    above on the former and from below on the latter.

    The inverse hyperbolic tangent is also known as `atanh` or ``tanh^-1``.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 86. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Inverse hyperbolic function",
           http://en.wikipedia.org/wiki/Arctanh

    Examples
    --------
    >>> np.arctanh([0, -0.5])
    array([ 0.        , -0.54930614])

    """)

add_newdoc('numpy.core.umath', 'bitwise_and',
    """
    Compute bit-wise AND of two arrays, element-wise.

    When calculating the bit-wise AND between two elements, ``x`` and ``y``,
    each element is first converted to its binary representation (which works
    just like the decimal system, only now we're using 2 instead of 10):

    .. math:: x = \\sum_{i=0}^{W-1} a_i \\cdot 2^i\\\\\n          y = \\sum_{i=0}^{W-1} b_i \\cdot 2^i,

    where ``W`` is the bit-width of the type (i.e., 8 for a byte or uint8),
    and each :math:`a_i` and :math:`b_j` is either 0 or 1.  For example, 13
    is represented as ``00001101``, which translates to
    :math:`2^4 + 2^3 + 2`.

    The bit-wise operator is the result of

    .. math:: z = \\sum_{i=0}^{i=W-1} (a_i \\wedge b_i) \\cdot 2^i,

    where :math:`\\wedge` is the AND operator, which yields one whenever
    both :math:`a_i` and :math:`b_i` are 1.

    Parameters
    ----------
    x1, x2 : array_like
        Only integer types are handled (including booleans).

    Returns
    -------
    out : array_like
        Result.

    See Also
    --------
    bitwise_or, bitwise_xor
    logical_and
    binary_repr :
        Return the binary representation of the input number as a string.

    Examples
    --------
    We've seen that 13 is represented by ``00001101``.  Similary, 17 is
    represented by ``00010001``.  The bit-wise AND of 13 and 17 is
    therefore ``000000001``, or 1:

    >>> np.bitwise_and(13, 17)
    1

    >>> np.bitwise_and(14, 13)
    12
    >>> np.binary_repr(12)
    '1100'
    >>> np.bitwise_and([14,3], 13)
    array([12,  1])

    >>> np.bitwise_and([11,7], [4,25])
    array([0, 1])
    >>> np.bitwise_and(np.array([2,5,255]), np.array([3,14,16]))
    array([ 2,  4, 16])
    >>> np.bitwise_and([True, True], [False, True])
    array([False,  True], dtype=bool)

    """)

add_newdoc('numpy.core.umath', 'bitwise_or',
    """
    Compute bit-wise OR of two arrays, element-wise.

    When calculating the bit-wise OR between two elements, ``x`` and ``y``,
    each element is first converted to its binary representation (which works
    just like the decimal system, only now we're using 2 instead of 10):

    .. math:: x = \\sum_{i=0}^{W-1} a_i \\cdot 2^i\\\\\n          y = \\sum_{i=0}^{W-1} b_i \\cdot 2^i,

    where ``W`` is the bit-width of the type (i.e., 8 for a byte or uint8),
    and each :math:`a_i` and :math:`b_j` is either 0 or 1.  For example, 13
    is represented as ``00001101``, which translates to
    :math:`2^4 + 2^3 + 2`.

    The bit-wise operator is the result of

    .. math:: z = \\sum_{i=0}^{i=W-1} (a_i \\vee b_i) \\cdot 2^i,

    where :math:`\\vee` is the OR operator, which yields one whenever
    either :math:`a_i` or :math:`b_i` is 1.

    Parameters
    ----------
    x1, x2 : array_like
        Only integer types are handled (including booleans).

    Returns
    -------
    out : array_like
        Result.

    See Also
    --------
    bitwise_and, bitwise_xor
    logical_or
    binary_repr :
        Return the binary representation of the input number as a string.

    Examples
    --------
    We've seen that 13 is represented by ``00001101``.  Similary, 16 is
    represented by ``00010000``.  The bit-wise OR of 13 and 16 is
    therefore ``000111011``, or 29:

    >>> np.bitwise_or(13, 16)
    29
    >>> np.binary_repr(29)
    '11101'

    >>> np.bitwise_or(32, 2)
    34
    >>> np.bitwise_or([33, 4], 1)
    array([33,  5])
    >>> np.bitwise_or([33, 4], [1, 2])
    array([33,  6])

    >>> np.bitwise_or(np.array([2, 5, 255]), np.array([4, 4, 4]))
    array([  6,   5, 255])
    >>> np.bitwise_or(np.array([2, 5, 255, 2147483647L], dtype=np.int32), \\\n... np.array([4, 4, 4, 2147483647L], dtype=np.int32))
    array([         6,          5,        255, 2147483647])
    >>> np.bitwise_or([True, True], [False, True])
    array([ True,  True], dtype=bool)

    """)

add_newdoc('numpy.core.umath', 'bitwise_xor',
    """
    Compute bit-wise XOR of two arrays, element-wise.

    When calculating the bit-wise XOR between two elements, ``x`` and ``y``,
    each element is first converted to its binary representation (which works
    just like the decimal system, only now we're using 2 instead of 10):

    .. math:: x = \\sum_{i=0}^{W-1} a_i \\cdot 2^i\\\\\n          y = \\sum_{i=0}^{W-1} b_i \\cdot 2^i,

    where ``W`` is the bit-width of the type (i.e., 8 for a byte or uint8),
    and each :math:`a_i` and :math:`b_j` is either 0 or 1.  For example, 13
    is represented as ``00001101``, which translates to
    :math:`2^4 + 2^3 + 2`.

    The bit-wise operator is the result of

    .. math:: z = \\sum_{i=0}^{i=W-1} (a_i \\oplus b_i) \\cdot 2^i,

    where :math:`\\oplus` is the XOR operator, which yields one whenever
    either :math:`a_i` or :math:`b_i` is 1, but not both.

    Parameters
    ----------
    x1, x2 : array_like
        Only integer types are handled (including booleans).

    Returns
    -------
    out : ndarray
        Result.

    See Also
    --------
    bitwise_and, bitwise_or
    logical_xor
    binary_repr :
        Return the binary representation of the input number as a string.

    Examples
    --------
    We've seen that 13 is represented by ``00001101``.  Similary, 17 is
    represented by ``00010001``.  The bit-wise XOR of 13 and 17 is
    therefore ``00011100``, or 28:

    >>> np.bitwise_xor(13, 17)
    28
    >>> np.binary_repr(28)
    '11100'

    >>> np.bitwise_xor(31, 5)
    26
    >>> np.bitwise_xor([31,3], 5)
    array([26,  6])

    >>> np.bitwise_xor([31,3], [5,6])
    array([26,  5])
    >>> np.bitwise_xor([True, True], [False, True])
    array([ True, False], dtype=bool)

    """)

add_newdoc('numpy.core.umath', 'ceil',
    """
    Return the ceiling of the input, element-wise.

    The ceil of the scalar `x` is the smallest integer `i`, such that
    `i >= x`.  It is often denoted as :math:`\\lceil x \\rceil`.

    Parameters
    ----------
    x : array_like
        Input data.

    Returns
    -------
    y : {ndarray, scalar}
        The ceiling of each element in `x`.

    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.ceil(a)
    array([-1., -1., -0.,  1.,  2.,  2.,  2.])

    """)

add_newdoc('numpy.core.umath', 'conjugate',
    """
    Return the complex conjugate, element-wise.

    The complex conjugate of a complex number is obtained by changing the
    sign of its imaginary part.

    Parameters
    ----------
    x : array_like
        Input value.

    Returns
    -------
    y : {ndarray, scalar}
        The complex conjugate of `x`, with same dtype as `y`.

    Examples
    --------
    >>> np.conjugate(1+2j)
    (1-2j)

    """)

add_newdoc('numpy.core.umath', 'cos',
    """
    Cosine elementwise.

    Parameters
    ----------
    x : array_like
        Input array in radians.

    Returns
    -------
    out : ndarray
        Output array of same shape as `x`.

    Examples
    --------
    >>> np.cos(np.array([0, np.pi/2, np.pi]))
    array([  1.00000000e+00,   6.12303177e-17,  -1.00000000e+00])

    """)

add_newdoc('numpy.core.umath', 'cosh',
    """
    Hyperbolic cosine, element-wise.

    Equivalent to ``1/2 * (np.exp(x) + np.exp(-x))`` and ``np.cos(1j*x)``.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    out : ndarray
        Output array of same shape as `x`.

    Examples
    --------
    >>> np.cosh(0)
    1.0

    The hyperbolic cosine describes the shape of a hanging cable:

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-4, 4, 1000)
    >>> plt.plot(x, np.cosh(x))
    >>> plt.show()

    """)

add_newdoc('numpy.core.umath', 'degrees',
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    x : array-like
      Angle in radians.

    Returns
    -------
    y : {ndarray, scalar}
      The corresponding angle in degrees.


    See Also
    --------
    radians : Convert angles from degrees to radians.
    unwrap : Remove large jumps in angle by wrapping.

    Notes
    -----
    degrees(x) is ``180 * x / pi``.

    Examples
    --------
    >>> np.degrees(np.pi/2)
    90.0

    """)

add_newdoc('numpy.core.umath', 'divide',
    """
    Divide arguments element-wise.

    Parameters
    ----------
    x1 : array_like
        Dividend array.
    x2 : array_like
        Divisor array.

    Returns
    -------
    y : {ndarray, scalar}
        The quotient `x1/x2`, element-wise. Returns a scalar if
        both  `x1` and `x2` are scalars.

    See Also
    --------
    seterr : Set whether to raise or warn on overflow, underflow and division
             by zero.

    Notes
    -----
    Equivalent to `x1` / `x2` in terms of array-broadcasting.

    Behavior on division by zero can be changed using `seterr`.

    When both `x1` and `x2` are of an integer type, `divide` will return
    integers and throw away the fractional part. Moreover, division by zero
    always yields zero in integer arithmetic.

    Examples
    --------
    >>> np.divide(2.0, 4.0)
    0.5
    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> np.divide(x1, x2)
    array([[ NaN,  1. ,  1. ],
           [ Inf,  4. ,  2.5],
           [ Inf,  7. ,  4. ]])

    Note the behavior with integer types:

    >>> np.divide(2, 4)
    0
    >>> np.divide(2, 4.)
    0.5

    Division by zero always yields zero in integer arithmetic, and does not
    raise an exception or a warning:

    >>> np.divide(np.array([0, 1], dtype=int), np.array([0, 0], dtype=int))
    array([0, 0])

    Division by zero can, however, be caught using `seterr`:

    >>> old_err_state = np.seterr(divide='raise')
    >>> np.divide(1, 0)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    FloatingPointError: divide by zero encountered in divide

    >>> ignored_states = np.seterr(**old_err_state)
    >>> np.divide(1, 0)
    0

    """)

add_newdoc('numpy.core.umath', 'equal',
    """
    Returns elementwise x1 == x2 in a bool array.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays of the same shape.

    Returns
    -------
    out : boolean
        The elementwise test `x1` == `x2`.

    """)

add_newdoc('numpy.core.umath', 'exp',
    """
    Calculate the exponential of the elements in the input array.

    Parameters
    ----------
    x : array_like
        Input values.

    Returns
    -------
    out : ndarray
        Element-wise exponential of `x`.

    Notes
    -----
    The irrational number ``e`` is also known as Euler's number.  It is
    approximately 2.718281, and is the base of the natural logarithm,
    ``ln`` (this means that, if :math:`x = \\ln y = \\log_e y`,
    then :math:`e^x = y`. For real input, ``exp(x)`` is always positive.

    For complex arguments, ``x = a + ib``, we can write
    :math:`e^x = e^a e^{ib}`.  The first term, :math:`e^a`, is already
    known (it is the real argument, described above).  The second term,
    :math:`e^{ib}`, is :math:`\\cos b + i \\sin b`, a function with magnitude
    1 and a periodic phase.

    References
    ----------
    .. [1] Wikipedia, "Exponential function",
           http://en.wikipedia.org/wiki/Exponential_function
    .. [2] M. Abramovitz and I. A. Stegun, "Handbook of Mathematical Functions
           with Formulas, Graphs, and Mathematical Tables," Dover, 1964, p. 69,
           http://www.math.sfu.ca/~cbm/aands/page_69.htm

    Examples
    --------
    Plot the magnitude and phase of ``exp(x)`` in the complex plane:

    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(-2*np.pi, 2*np.pi, 100)
    >>> xx = x + 1j * x[:, np.newaxis] # a + ib over complex plane
    >>> out = np.exp(xx)

    >>> plt.subplot(121)
    >>> plt.imshow(np.abs(out),
    ...            extent=[-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi])
    >>> plt.title('Magnitude of exp(x)')

    >>> plt.subplot(122)
    >>> plt.imshow(np.angle(out),
    ...            extent=[-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi])
    >>> plt.title('Phase (angle) of exp(x)')
    >>> plt.show()

    """)

add_newdoc('numpy.core.umath', 'expm1',
    """
    e**x-1 elementwise.

    """)

add_newdoc('numpy.core.umath', 'fabs',
    """
    Compute the absolute values elementwise.

    This function returns the absolute values (positive magnitude) of the data
    in `x`. Complex values are not handled, use `absolute` to find the
    absolute values of complex data.

    Parameters
    ----------
    x : array_like
        The array of numbers for which the absolute values are required. If
        `x` is a scalar, the result `y` will also be a scalar.

    Returns
    -------
    y : {ndarray, scalar}
        The absolute values of `x`, the returned values are always floats.

    See Also
    --------
    absolute : Absolute values including `complex` types.

    Examples
    --------
    >>> np.fabs(-1)
    1.0
    >>> np.fabs([-1.2, 1.2])
    array([ 1.2,  1.2])

    """)

add_newdoc('numpy.core.umath', 'floor',
    """
    Return the floor of the input, element-wise.

    The floor of the scalar `x` is the largest integer `i`, such that
    `i <= x`.  It is often denoted as :math:`\\lfloor x \\rfloor`.

    Parameters
    ----------
    x : array_like
        Input data.

    Returns
    -------
    y : {ndarray, scalar}
        The floor of each element in `x`.

    Notes
    -----
    Some spreadsheet programs calculate the "floor-towards-zero", in other
    words ``floor(-2.5) == -2``.  NumPy, however, uses the a definition of
    `floor` such that `floor(-2.5) == -3``.

    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.floor(a)
    array([-2., -2., -1.,  0.,  1.,  1.,  2.])

    """)

add_newdoc('numpy.core.umath', 'floor_divide',
    """
    Return the largest integer smaller or equal to the division of the inputs.

    Parameters
    ----------
    x1 : array_like
        Numerator.
    x2 : array_like
        Denominator.

    Returns
    -------
    y : ndarray
        y = floor(`x1`/`x2`)


    See Also
    --------
    divide : Standard division.
    floor : Round a number to the nearest integer toward minus infinity.
    ceil : Round a number to the nearest integer toward infinity.

    Examples
    --------
    >>> np.floor_divide(7,3)
    2
    >>> np.floor_divide([1., 2., 3., 4.], 2.5)
    array([ 0.,  0.,  1.,  1.])

    """)

add_newdoc('numpy.core.umath', 'fmod',
    """
    Return the remainder of division.

    This is the NumPy implementation of the C modulo operator `%`.

    Parameters
    ----------
    x1 : array_like
      Dividend.
    x2 : array_like
      Divisor.

    Returns
    -------
    y : array_like
      The remainder of the division of `x1` by `x2`.

    See Also
    --------
    mod : Modulo operation where the quotient is `floor(x1,x2)`.

    Notes
    -----
    The result of the modulo operation for negative dividend and divisors is
    bound by conventions. In `fmod`, the sign of the remainder is the sign of
    the dividend, and the sign of the divisor has no influence on the results.

    Examples
    --------
    >>> np.fmod([-3, -2, -1, 1, 2, 3], 2)
    array([-1,  0, -1,  1,  0,  1])

    >>> np.mod([-3, -2, -1, 1, 2, 3], 2)
    array([1, 0, 1, 1, 0, 1])

    """)

add_newdoc('numpy.core.umath', 'greater',
    """
    Return (x1 > x2) element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays.

    Returns
    -------
    Out : {ndarray, bool}
        Output array of bools, or a single bool if `x1` and `x2` are scalars.

    See Also
    --------
    greater_equal

    Examples
    --------
    >>> np.greater([4,2],[2,2])
    array([ True, False], dtype=bool)

    """)

add_newdoc('numpy.core.umath', 'greater_equal',
    """
    Returns (x1 >= x2) element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays.

    Returns
    -------
    Out : {ndarray, bool}
        Output array of bools, or a single bool if `x1` and `x2` are scalars.

    See Also
    --------
    greater

    Examples
    --------
    >>> np.greater_equal([4,2],[2,2])
    array([ True, True], dtype=bool)

    """)

add_newdoc('numpy.core.umath', 'hypot',
    """
    Given two sides of a right triangle, return its hypotenuse.

    Parameters
    ----------
    x : array-like
      Base of the triangle.
    y : array-like
      Height of the triangle.

    Returns
    -------
    z : {ndarray, scalar}
      Hypotenuse of the triangle: sqrt(x**2 + y**2)


    Examples
    --------
    >>> np.hypot(3,4)
    5.0

    """)

add_newdoc('numpy.core.umath', 'invert',
    """
    Compute bit-wise inversion, or bit-wise NOT, element-wise.

    When calculating the bit-wise NOT of an element ``x``, each element is
    first converted to its binary representation (which works
    just like the decimal system, only now we're using 2 instead of 10):

    .. math:: x = \\sum_{i=0}^{W-1} a_i \\cdot 2^i

    where ``W`` is the bit-width of the type (i.e., 8 for a byte or uint8),
    and each :math:`a_i` is either 0 or 1.  For example, 13 is represented
    as ``00001101``, which translates to :math:`2^4 + 2^3 + 2`.

    The bit-wise operator is the result of

    .. math:: z = \\sum_{i=0}^{i=W-1} (\\lnot a_i) \\cdot 2^i,

    where :math:`\\lnot` is the NOT operator, which yields 1 whenever
    :math:`a_i` is 0 and yields 0 whenever :math:`a_i` is 1.

    For signed integer inputs, the two's complement is returned.
    In a two's-complement system negative numbers are represented by the two's
    complement of the absolute value. This is the most common method of
    representing signed integers on computers [1]_. A N-bit two's-complement
    system can represent every integer in the range
    :math:`-2^{N-1}` to :math:`+2^{N-1}-1`.

    Parameters
    ----------
    x1 : ndarray
        Only integer types are handled (including booleans).

    Returns
    -------
    out : ndarray
        Result.

    See Also
    --------
    bitwise_and, bitwise_or, bitwise_xor
    logical_not
    binary_repr :
        Return the binary representation of the input number as a string.

    Notes
    -----
    `bitwise_not` is an alias for `invert`:

    >>> np.bitwise_not is np.invert
    True

    References
    ----------
    .. [1] Wikipedia, "Two's complement",
        http://en.wikipedia.org/wiki/Two's_complement

    Examples
    --------
    We've seen that 13 is represented by ``00001101``.
    The invert or bit-wise NOT of 13 is then:

    >>> np.invert(np.array([13], dtype=uint8))
    array([242], dtype=uint8)
    >>> np.binary_repr(x, width=8)
    '00001101'
    >>> np.binary_repr(242, width=8)
    '11110010'

    The result depends on the bit-width:

    >>> np.invert(np.array([13], dtype=uint16))
    array([65522], dtype=uint16)
    >>> np.binary_repr(x, width=16)
    '0000000000001101'
    >>> np.binary_repr(65522, width=16)
    '1111111111110010'

    When using signed integer types the result is the two's complement of
    the result for the unsigned type:

    >>> np.invert(np.array([13], dtype=int8))
    array([-14], dtype=int8)
    >>> np.binary_repr(-14, width=8)
    '11110010'

    Booleans are accepted as well:

    >>> np.invert(array([True, False]))
    array([False,  True], dtype=bool)

    """)

add_newdoc('numpy.core.umath', 'isfinite',
    """
    Returns True where x is finite, False otherwise.

    Parameters
    ----------
    x : array_like
      input values

    Returns
    -------
    y : {ndarray, bool}
      array of bools

    Notes
    -----
    `Nan` is considered as non-finite.

    Examples
    --------
    >>> np.isfinite([np.log(-1.),1.,np.log(0)])
    array([False,  True, False], dtype=bool)

    """)

add_newdoc('numpy.core.umath', 'isinf',
    """
    Returns True where x is +inf or -inf, False otherwise.

    Parameters
    ----------
    x : array_like
      input values

    Returns
    -------
    y : {ndarray, bool}
      array of bools

    Examples
    --------
    >>> np.isinf([np.inf, -np.inf, 1.0, np.nan])
    array([ True,  True, False, False], dtype=bool)

    """)

add_newdoc('numpy.core.umath', 'isnan',
    """
    Returns True where elements are Not-A-Number, False otherwise.

    Parameters
    ----------
    x : array_like
      input values.

    Returns
    -------
    y : {ndarray, bool}
      array of bools

    Examples
    --------
    >>> np.isnan([np.log(-1.),1.,np.log(0)])
    array([ True, False, False], dtype=bool)

    """)

add_newdoc('numpy.core.umath', 'left_shift',
    """
    Computes x1 << x2 (x1 shifted to left by x2 bits) elementwise.

    """)

add_newdoc('numpy.core.umath', 'less',
    """
    Returns (x1 < x2) element-wise.

    Parameters
    ----------
    x1, x2 : array-like
        Input arrays.

    Returns
    -------
    Out : {ndarray, bool}
        Output array of bools, or a single bool if `x1` and `x2` are scalars.

    See Also
    --------
    less_equal

    Examples
    --------
    >>> np.less([1,2],[2,2])
    array([ True, False], dtype=bool)

    """)

add_newdoc('numpy.core.umath', 'less_equal',
    """
    Returns (x1 <= x2) element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays.

    Returns
    -------
    Out : {ndarray, bool}
        Output array of bools, or a single bool if `x1` and `x2` are scalars.

    See Also
    --------
    less

    Examples
    --------
    >>> np.less_equal([1,2,3],[2,2,2])
    array([ True,  True, False], dtype=bool)

    """)

add_newdoc('numpy.core.umath', 'log',
    """
    Natural logarithm, element-wise.

    The natural logarithm `log` is the inverse of the exponential function,
    so that `log(exp(x)) = x`. The natural logarithm is logarithm in base `e`.

    Parameters
    ----------
    x : array_like
      Input value.

    Returns
    -------
    y : {ndarray, scalar}
      The natural logarithm of `x`, element-wise.

    See Also
    --------
    log10, log2, log1p

    Notes
    -----
    Logarithm is a multivalued function: for each `x` there is an infinite
    number of `z` such that `exp(z) = x`. The convention is to return the `z`
    whose imaginary part lies in `[-pi, pi]`.

    For real-valued input data types, `log` always returns real output. For
    each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `log` is a complex analytical function that
    has a branch cut `[-inf, 0]` and is continuous from above on it. `log`
    handles the floating-point negative zero as an infinitesimal negative
    number, conforming to the C99 standard.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 67. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Logarithm". http://en.wikipedia.org/wiki/Logarithm

    Examples
    --------
    >>> np.log([1, np.e, np.e**2, 0])
    array([  0.,   1.,   2., -Inf])

    """)

add_newdoc('numpy.core.umath', 'log10',
    """
    Compute the logarithm in base 10 elementwise.

    Parameters
    ----------
    x : array_like
      input values.

    Returns
    -------
    y : {ndarray, scalar}
      base-10 logarithm of `x`.


    Notes
    -----
    Logarithm is a multivalued function: for each `x` there is an infinite
    number of `z` such that `10**z = x`. The convention is to return the `z`
    whose imaginary part lies in `[-pi, pi]`.

    For real-valued input data types, `log10` always returns real output. For
    each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `log10` is a complex analytical function that
    has a branch cut `[-inf, 0]` and is continuous from above on it. `log10`
    handles the floating-point negative zero as an infinitesimal negative
    number, conforming to the C99 standard.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 67. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Logarithm". http://en.wikipedia.org/wiki/Logarithm

    Examples
    --------
    >>> np.log10([1.e-15,-3.])
    array([-15.,  NaN])

    """)

add_newdoc('numpy.core.umath', 'log1p',
    """
    `log(1 + x)` in base `e`, elementwise.

    Parameters
    ----------
    x : array_like
        Input values.

    Returns
    -------
    y : {ndarray, scalar}
        Natural logarithm of `1 + x`, elementwise.

    Notes
    -----
    For real-valued input, `log1p` is accurate also for `x` so small
    that `1 + x == 1` in floating-point accuracy.

    Logarithm is a multivalued function: for each `x` there is an infinite
    number of `z` such that `exp(z) = 1 + x`. The convention is to return
    the `z` whose imaginary part lies in `[-pi, pi]`.

    For real-valued input data types, `log1p` always returns real output. For
    each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `log1p` is a complex analytical function that
    has a branch cut `[-inf, -1]` and is continuous from above on it. `log1p`
    handles the floating-point negative zero as an infinitesimal negative
    number, conforming to the C99 standard.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 67. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Logarithm". http://en.wikipedia.org/wiki/Logarithm

    Examples
    --------
    >>> np.log1p(1e-99)
    1e-99
    >>> np.log(1 + 1e-99)
    0.0

    """)

add_newdoc('numpy.core.umath', 'logical_and',
    """
    Compute the truth value of x1 AND x2 elementwise.

    Parameters
    ----------
    x1, x2 : array_like
        Logical AND is applied to the elements of `x1` and `x2`.
        They have to be of the same shape.


    Returns
    -------
    y : {ndarray, bool}
        Boolean result with the same shape as `x1` and `x2` of the logical
        AND operation on elements of `x1` and `x2`.

    See Also
    --------
    logical_or, logical_not, logical_xor
    bitwise_and

    Examples
    --------
    >>> np.logical_and(True, False)
    False
    >>> np.logical_and([True, False], [False, False])
    array([False, False], dtype=bool)

    >>> x = np.arange(5)
    >>> np.logical_and(x>1, x<4)
    array([False, False,  True,  True, False], dtype=bool)

    """)

add_newdoc('numpy.core.umath', 'logical_not',
    """
    Compute the truth value of NOT x elementwise.

    Parameters
    ----------
    x : array_like
        Logical NOT is applied to the elements of `x`.

    Returns
    -------
    y : {ndarray, bool}
        Boolean result with the same shape as `x` of the NOT operation
        on elements of `x`.

    See Also
    --------
    logical_and, logical_or, logical_xor

    Examples
    --------
    >>> np.logical_not(3)
    False
    >>> np.logical_not([True, False, 0, 1])
    array([False,  True,  True, False], dtype=bool)

    >>> x = np.arange(5)
    >>> np.logical_not(x<3)
    array([False, False, False,  True,  True], dtype=bool)

    """)

add_newdoc('numpy.core.umath', 'logical_or',
    """
    Compute the truth value of x1 OR x2 elementwise.

    Parameters
    ----------
    x1, x2 : array_like
        Logical OR is applied to the elements of `x1` and `x2`.
        They have to be of the same shape.

    Returns
    -------
    y : {ndarray, bool}
        Boolean result with the same shape as `x1` and `x2` of the logical
        OR operation on elements of `x1` and `x2`.

    See Also
    --------
    logical_and, logical_not, logical_xor
    bitwise_or

    Examples
    --------
    >>> np.logical_or(True, False)
    True
    >>> np.logical_or([True, False], [False, False])
    array([ True, False], dtype=bool)

    >>> x = np.arange(5)
    >>> np.logical_or(x < 1, x > 3)
    array([ True, False, False, False,  True], dtype=bool)

    """)

add_newdoc('numpy.core.umath', 'logical_xor',
    """
    Compute the truth value of x1 XOR x2 elementwise.

    Parameters
    ----------
    x1, x2 : array_like
        Logical XOR is applied to the elements of `x1` and `x2`.
        They have to be of the same shape.

    Returns
    -------
    y : {ndarray, bool}
        Boolean result with the same shape as `x1` and `x2` of the logical
        XOR operation on elements of `x1` and `x2`.

    See Also
    --------
    logical_and, logical_or, logical_not
    bitwise_xor

    Examples
    --------
    >>> np.logical_xor(True, False)
    True
    >>> np.logical_xor([True, True, False, False], [True, False, True, False])
    array([False,  True,  True, False], dtype=bool)

    >>> x = np.arange(5)
    >>> np.logical_xor(x < 1, x > 3)
    array([ True, False, False, False,  True], dtype=bool)

    """)

add_newdoc('numpy.core.umath', 'maximum',
    """
    Returns maximum (if x1 > x2: x1;  else: x2) elementwise.

    """)

add_newdoc('numpy.core.umath', 'minimum',
    """
    Returns minimum (if x1 < x2: x1;  else: x2) elementwise

    """)

add_newdoc('numpy.core.umath', 'modf',
    """
    Return the fractional and integral part of a number.

    The fractional and integral parts are negative if the given number is
    negative.

    Parameters
    ----------
    x : array_like
        Input number.

    Returns
    -------
    y1 : ndarray
        Fractional part of `x`.
    y2 : ndarray
        Integral part of `x`.

    Examples
    --------
    >>> np.modf(2.5)
    (0.5, 2.0)
    >>> np.modf(-.4)
    (-0.40000000000000002, -0.0)

    """)

add_newdoc('numpy.core.umath', 'multiply',
    """
    Multiply arguments elementwise.

    Parameters
    ----------
    x1, x2 : array-like
        The arrays to be multiplied.

    Returns
    -------
    y : {ndarray, scalar}
        The product of `x1` and `x2`, elementwise. Returns a scalar if
        both  `x1` and `x2` are scalars.

    Notes
    -----
    Equivalent to `x1` * `x2` in terms of array-broadcasting.

    Examples
    --------
    >>> np.multiply(2.0, 4.0)
    8.0

    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> np.multiply(x1, x2)
    array([[  0.,   1.,   4.],
           [  0.,   4.,  10.],
           [  0.,   7.,  16.]])

    """)

add_newdoc('numpy.core.umath', 'negative',
    """
    Returns an array with the negative of each element of the original array.

    Parameters
    ----------
    x : {array_like, scalar}
        Input array.

    Returns
    -------
    y : {ndarray, scalar}
        Returned array or scalar `y=-x`.

    Examples
    --------
    >>> np.negative([1.,-1.])
    array([-1.,  1.])

    """)

add_newdoc('numpy.core.umath', 'not_equal',
    """
    Return (x1 != x2) element-wise.

    Parameters
    ----------
    x1, x2 : array_like
      Input arrays.
    out : ndarray, optional
      A placeholder the same shape as `x1` to store the result.

    Returns
    -------
    not_equal : ndarray bool, scalar bool
      For each element in `x1, x2`, return True if `x1` is not equal
      to `x2` and False otherwise.


    See Also
    --------
    equal, greater, greater_equal, less, less_equal

    Examples
    --------
    >>> np.not_equal([1.,2.], [1., 3.])
    array([False,  True], dtype=bool)

    """)

add_newdoc('numpy.core.umath', 'ones_like',
    """
    Returns an array of zeros with the same shape and type as a given array.

    Equivalent to ``a.copy().fill(1)``.

    Please refer to the documentation for `zeros_like`.

    See Also
    --------
    zeros_like

    Examples
    --------
    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.ones_like(a)
    array([[1, 1, 1],
           [1, 1, 1]])

    """)

add_newdoc('numpy.core.umath', 'power',
    """
    Computes `x1` ** `x2` elementwise.

    Raise each base in `x1` to the power of the exponents in `x2`. This
    requires that `x1` and `x2` must be broadcastable to the same shape.

    Parameters
    ----------
    x1 : array_like
        The bases.
    x2 : array_like
        The exponents.

    Returns
    -------
    y : ndarray
        The bases in `x1` raised to the exponents in `x2`.

    Examples
    --------
    Cube each element in a list.

    >>> x1 = range(6)
    >>> x1
    [0, 1, 2, 3, 4, 5]
    >>> np.power(x1, 3)
    array([  0,   1,   8,  27,  64, 125])

    Raise the bases to different exponents.

    >>> x2 = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
    >>> np.power(x1, x2)
    array([  0.,   1.,   8.,  27.,  16.,   5.])

    The effect of broadcasting.

    >>> x2 = np.array([[1, 2, 3, 3, 2, 1], [1, 2, 3, 3, 2, 1]])
    >>> x2
    array([[1, 2, 3, 3, 2, 1],
           [1, 2, 3, 3, 2, 1]])
    >>> np.power(x1, x2)
    array([[ 0,  1,  8, 27, 16,  5],
           [ 0,  1,  8, 27, 16,  5]])

    """)

add_newdoc('numpy.core.umath', 'radians',
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    x : array_like
      Angles in degrees.

    Returns
    -------
    y : {ndarray, scalar}
      The corresponding angle in radians.

    See Also
    --------
    degrees : Convert angles from radians to degrees.
    unwrap : Remove large jumps in angle by wrapping.

    Notes
    -----
    ``radians(x)`` is ``x * pi / 180``.

    Examples
    --------
    >>> np.radians(180)
    3.1415926535897931

    """)

add_newdoc('numpy.core.umath', 'reciprocal',
    """
    Compute 1/x.

    Parameters
    ----------
    x : array_like
        Input value.

    Returns
    -------
    y : {ndarray, scalar}
        Return value.

    Examples
    --------
    >>> reciprocal(2.)
    0.5
    >>> reciprocal([1, 2., 3.33])
    array([ 1.       ,  0.5      ,  0.3003003])

    """)

add_newdoc('numpy.core.umath', 'remainder',
    """
    Computes x1-n*x2 where n is floor(x1 / x2)

    Parameters
    ----------
    x1 : array_like
        Dividend array.
    x2 : array_like
        Divisor array.

    Returns
    -------
    y : {ndarray, scalar}
        The quotient `x1/x2`, element-wise. Returns a scalar if
        both  `x1` and `x2` are scalars.

    See Also
    --------
    divide
    floor

    Notes
    -----
    Returns 0 when `x2` is 0.

    Examples
    --------
    >>> np.remainder([4,7],[2,3])
    array([0, 1])

    """)

add_newdoc('numpy.core.umath', 'right_shift',
    """
    Computes x1 >> x2 (x1 shifted to right by x2 bits) elementwise.

    """)

add_newdoc('numpy.core.umath', 'rint',
    """
    Round elements of the array to the nearest integer.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    out : ndarray
        Output array is same shape and type as `x`.

    Examples
    --------
    >>> a = [-4.1, -3.6, -2.5, 0.1, 2.5, 3.1, 3.9]
    >>> np.rint(a)
    array([-4., -4., -2.,  0.,  2.,  3.,  4.])

    """)

add_newdoc('numpy.core.umath', 'sign',
    """
    Return the sign of a number.

    -1 if x < 0, 0 if x==0, 1 if x > 0.

    Parameters
    ----------
    x : array_like
      Input values.

    Returns
    -------
    y : {ndarray, scalar}
      The sign of `x`.

    Examples
    --------
    >>> np.sign([-5., 4.5])
    array([-1.,  1.])
    >>> np.sign(0)
    0

    """)

add_newdoc('numpy.core.umath', 'signbit',
    """
    Returns True where `signbit` of `x` is set (`x<0`).

    Parameters
    ----------
    x: array-like or scalar
      the input value(s).
    output : array-like or scalar
      the returned boolean(s)

    Examples
    --------
    >>> np.signbit(-1.2)
    True
    >>> np.signbit(np.array([1,-2.3,2.1]))
    array([False,  True, False], dtype=bool)

    """)

add_newdoc('numpy.core.umath', 'sin',
    """
    Trigonometric sine, element-wise.

    Parameters
    ----------
    x : array_like
        Angle, in radians (:math:`2 \\pi` rad equals 360 degrees).

    Returns
    -------
    y : array_like
        The sine of each element of x.

    See Also
    --------
    arcsin, sinh, cos

    Notes
    -----
    The sine is one of the fundamental functions of trigonometry
    (the mathematical study of triangles).  Consider a circle of radius
    1 centered on the origin.  A ray comes in from the :math:`+x` axis,
    makes an angle at the origin (measured counter-clockwise from that
    axis), and departs from the origin.  The :math:`y` coordinate of
    the outgoing ray's intersection with the unit circle is the sine
    of that angle.  It ranges from -1 for :math:`x=3\\pi / 2` to
    +1 for :math:`\\pi / 2.`  The function has zeroes where the angle is
    a multiple of :math:`\\pi`.  Sines of angles between :math:`\\pi` and
    :math:`2\\pi` are negative.  The numerous properties of the sine and
    related functions are included in any standard trigonometry text.

    Examples
    --------
    Print sine of one angle:

    >>> np.sin(np.pi/2.)
    1.0

    Print sines of an array of angles given in degrees:

    >>> np.sin(np.array((0., 30., 45., 60., 90.)) * np.pi / 180. )
    array([ 0.        ,  0.5       ,  0.70710678,  0.8660254 ,  1.        ])

    Plot the sine function:

    >>> import matplotlib.pylab as plt
    >>> x = np.linspace(-np.pi, np.pi, 201)
    >>> plt.plot(x, np.sin(x))
    >>> plt.xlabel('Angle [rad]')
    >>> plt.ylabel('sin(x)')
    >>> plt.axis('tight')
    >>> plt.show()

    """)

add_newdoc('numpy.core.umath', 'sinh',
    """
    Hyperbolic sine, element-wise.

    Equivalent to ``1/2 * (np.exp(x) - np.exp(-x))`` or
    ``-1j * np.sin(1j*x)``.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    out : ndarray
        Output array of same shape as `x`.

    """)

add_newdoc('numpy.core.umath', 'sqrt',
    """
    Return the positive square-root of an array, element-wise.

    Parameters
    ----------
    x : array_like
        The square root of each element in this array is calculated.

    Returns
    -------
    y : ndarray
        The square-root of each element in `x`.  If any element in `x`
        is complex, a complex array is returned.  If all of the elements
        of `x` are real, negative elements will return numpy.nan elements.

    See Also
    --------
    numpy.lib.scimath.sqrt
        A version which will return complex numbers when given negative reals.

    Notes
    -----
    `sqrt` has a branch cut ``[-inf, 0)`` and is continuous from above on it.

    Examples
    --------
    >>> np.sqrt([1,4,9])
    array([ 1.,  2.,  3.])

    >>> np.sqrt([4, -1, -3+4J])
    array([ 2.+0.j,  0.+1.j,  1.+2.j])

    >>> np.sqrt([4, -1, numpy.inf])
    array([  2.,  NaN,  Inf])

    """)

add_newdoc('numpy.core.umath', 'square',
    """
    Compute `x` squared, or `x` to the power of two.

    Parameters
    ----------
    x : array_like or scalar
        Input data.

    Returns
    -------
    out : ndarray or scalar
        Element-wise `x*x`, of the same shape and dtype as `x`.
        `out` is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.linalg.matrix_power
    sqrt
    power

    Examples
    --------
    >>> np.square([-1j, 1])
    array([-1.-0.j,  1.+0.j])

    """)

add_newdoc('numpy.core.umath', 'subtract',
    """
    Subtract arguments elementwise.

    Parameters
    ----------
    x1, x2 : {array_like, scalar}
        The arrays to be subtracted from each other.  If type is 'array_like'
        the `x1` and `x2` shapes must be identical.

    Returns
    -------
    y : {ndarray, scalar}
        The difference of `x1` and `x2`, elementwise.  Returns a scalar if
        both  `x1` and `x2` are scalars.

    Notes
    -----
    Equivalent to `x1` - `x2` in terms of array-broadcasting.

    Examples
    --------
    >>> np.subtract(1.0, 4.0)
    -3.0

    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> np.subtract(x1, x2)
    array([[ 0.,  0.,  0.],
           [ 3.,  3.,  3.],
           [ 6.,  6.,  6.]])

    """)

add_newdoc('numpy.core.umath', 'tan',
    """
    Compute tangent elementwise.

    Parameters
    ----------
    x : array_like
      Angles in radians.

    Returns
    -------
    y : ndarray or scalar
      The corresponding tangent values.


    Examples
    --------
    >>> from math import pi
    >>> np.tan(np.array([-pi,pi/2,pi]))
    array([  1.22460635e-16,   1.63317787e+16,  -1.22460635e-16])

    """)

add_newdoc('numpy.core.umath', 'tanh',
    """
    Hyperbolic tangent elementwise.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    y : ndarray or scalar
        The corresponding hyperbolic tangent values.

    """)

add_newdoc('numpy.core.umath', 'true_divide',
    """
    Returns an elementwise, true division of the inputs.

    Instead of the Python traditional 'floor division', this returns a true
    division.  True division adjusts the output type to present the best
    answer, regardless of input types.

    Parameters
    ----------
    x1 : array_like
        Dividend
    x2 : array_like
        Divisor

    Returns
    -------
    out : {ndarray, scalar}
        Result is scalar if both inputs are scalar, ndarray otherwise.

    Notes
    -----
    The floor division operator ('//') was added in Python 2.2 making '//'
    and '/' equivalent operators.  The default floor division operation of
    '/' can be replaced by true division with
    'from __future__ import division'.

    In Python 3.0, '//' will be the floor division operator and '/' will be
    the true division operator.  The 'true_divide(`x1`, `x2`)' function is
    equivalent to true division in Python.

    """)

