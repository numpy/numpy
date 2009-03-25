"""
=========
Constants
=========

Numpy includes several constants:

%(constant_list)s
"""
import textwrap

# Maintain same format as in numpy.add_newdocs
constants = []
def add_newdoc(module, name, doc):
    constants.append((name, doc))

add_newdoc('numpy', 'Inf',
    """
    IEEE 754 floating point representation of (positive) infinity.

    Returns
    -------
    y : A floating point representation of positive infinity.

    Notes
    -----
    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Also that positive infinity is not equivalent to negative infinity. But
    infinity is equivalent to positive infinity.

    Use numpy.inf because Inf, Infinity, PINF, infty are equivalent
    definitions of numpy.inf.

    """)

add_newdoc('numpy', 'Infinity',
    """
    IEEE 754 floating point representation of (positive) infinity.

    Returns
    -------
    y : A floating point representation of positive infinity.

    Notes
    -----
    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Also that positive infinity is not equivalent to negative infinity. But
    infinity is equivalent to positive infinity.

    Use numpy.inf because Inf, Infinity, PINF, infty are equivalent
    definitions of numpy.inf.

    """)

add_newdoc('numpy', 'NAN',
    """
    IEEE 754 floating point representation of Not a Number (NaN).

    Returns
    -------
    y : A floating point representation of Not a Number.

    See Also
    --------
    isnan : Shows which elements are Not a Number.
    isfinite : Shows which elements are finite (not one of
               Not a Number, positive infinity and negative infinity)

    Notes
    -----
    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.

    NaN and NAN are equivalent definitions of numpy.nan. Please use
    numpy.nan instead of numpy.NAN.


    Examples
    --------
    >>> np.NAN
    nan
    >>> np.log(-1)
    nan
    >>> np.log([-1, 1, 2])
    array([        NaN,  0.        ,  0.69314718])

    """)

add_newdoc('numpy', 'NINF',
    """
    IEEE 754 floating point representation of negative infinity.

    Returns
    -------
    y : A floating point representation of negative infinity.

    See Also
    --------
    isinf : Shows which elements are positive or negative infinity

    isposinf : Shows which elements are positive infinity

    isneginf : Shows which elements are negative infinity

    isnan : Shows which elements are Not a Number

    isfinite : Shows which elements are finite (not one of
             Not a Number, positive infinity and negative infinity)

    Notes
    -----
    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Also that positive infinity is not equivalent to negative infinity. But
    infinity is equivalent to positive infinity.


    Examples
    --------
    >>> np.NINF
    -inf
    >>> np.log(0)
    -inf

    """)

add_newdoc('numpy', 'NZERO',
    """
    IEEE 754 floating point representation of negative zero.

    Returns
    -------
    y : A floating point representation of negative zero.

    See Also
    --------
    PZERO : Defines positive zero.

    isinf : Shows which elements are positive or negative infinity.

    isposinf : Shows which elements are positive infinity.

    isneginf : Shows which elements are negative infinity.

    isnan : Shows which elements are Not a Number.

    isfinite : Shows which elements are finite - not one of
               Not a Number, positive infinity and negative infinity.

    Notes
    -----
    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). Negative zero is considered to be a finite number.


    Examples
    --------
    >>> np.NZERO
    -0.0
    >>> np.PZERO
    0.0
    >>> np.isfinite([np.NZERO])
    array([ True], dtype=bool)
    >>> np.isnan([np.NZERO])
    array([False], dtype=bool)
    >>> np.isinf([np.NZERO])
    array([False], dtype=bool)

    """)

add_newdoc('numpy', 'NaN',
    """
    IEEE 754 floating point representation of Not a Number (NaN).

    Returns
    -------
    y : A floating point representation of Not a Number.

    See Also
    --------

    isnan : Shows which elements are Not a Number.
     isfinite : Shows which elements are finite (not one of
               Not a Number, positive infinity and negative infinity)

    Notes
    -----
    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.

    NaN and NAN are equivalent definitions of numpy.nan. Please use
    numpy.nan instead of numpy.NaN.


    Examples
    --------
    >>> np.NaN
    nan
    >>> np.log(-1)
    nan
    >>> np.log([-1, 1, 2])
    array([        NaN,  0.        ,  0.69314718])

    """)

add_newdoc('numpy', 'PINF',
    """
    IEEE 754 floating point representation of (positive) infinity.

    Returns
    -------
    y : A floating point representation of positive infinity.

    Notes
    -----
    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Also that positive infinity is not equivalent to negative infinity. But
    infinity is equivalent to positive infinity.

    Use numpy.inf because Inf, Infinity, PINF, infty are equivalent
    definitions of numpy.inf.

    """)

add_newdoc('numpy', 'PZERO',
    """
    IEEE 754 floating point representation of positive zero.

    Returns
    -------
    y : A floating point representation of positive zero.

    See Also
    --------
    NZERO : Defines negative zero.

    isinf : Shows which elements are positive or negative infinity.

    isposinf : Shows which elements are positive infinity.

    isneginf : Shows which elements are negative infinity.

    isnan : Shows which elements are Not a Number.

    isfinite : Shows which elements are finite - not one of
               Not a Number, positive infinity and negative infinity.

    Notes
    -----
    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754).


    Examples
    --------
    >>> np.PZERO
    0.0
    >>> np.NZERO
    -0.0
    >>> np.isfinite([np.PZERO])
    array([ True], dtype=bool)
    >>> np.isnan([np.PZERO])
    array([False], dtype=bool)
    >>> np.isinf([np.PZERO])
    array([False], dtype=bool)

    """)

add_newdoc('numpy', 'e',
    """
    Euler's constant, base of natural logarithms, Napier's constant.

    `e = 2.71828182845904523536028747135266249775724709369995...`

    See Also
    --------
    exp : Exponential function
    log : Natural logarithm

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Napier_constant

    """)

add_newdoc('numpy', 'inf',
    """
    IEEE 754 floating point representation of (positive) infinity.

    Returns
    -------
    y : A floating point representation of positive infinity.

    See Also
    --------
    isinf : Shows which elements are positive or negative infinity

    isposinf : Shows which elements are positive infinity

    isneginf : Shows which elements are negative infinity

    isnan : Shows which elements are Not a Number

    isfinite : Shows which elements are finite (not one of
               Not a Number, positive infinity and negative infinity)

    Notes
    -----
    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Also that positive infinity is not equivalent to negative infinity. But
    infinity is equivalent to positive infinity.

    Inf, Infinity, PINF, infty are equivalent definitions of numpy.inf.


    Examples
    --------
    >>> np.inf
    inf
    >>> np.array([1])/0.
    array([ Inf])

    """)

add_newdoc('numpy', 'infty',
    """
    IEEE 754 floating point representation of (positive) infinity.

    Returns
    -------
    y : A floating point representation of positive infinity.

    Notes
    -----
    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Also that positive infinity is not equivalent to negative infinity. But
    infinity is equivalent to positive infinity.

    Use numpy.inf because Inf, Infinity, PINF, infty are equivalent
    definitions of numpy.inf.

    """)

add_newdoc('numpy', 'nan',
    """
    IEEE 754 floating point representation of Not a Number (NaN).

    Returns
    -------
    y : A floating point representation of Not a Number.

    See Also
    --------
    isnan : Shows which elements are Not a Number.
    isfinite : Shows which elements are finite (not one of
               Not a Number, positive infinity and negative infinity)

    Notes
    -----
    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.

    NaN and NAN are equivalent definitions of numpy.nan.


    Examples
    --------
    >>> np.nan
    nan
    >>> np.log(-1)
    nan
    >>> np.log([-1, 1, 2])
    array([        NaN,  0.        ,  0.69314718])

    """)

add_newdoc('numpy', 'newaxis',
    """
    See Also
    --------
    `numpy.doc.indexing`

    Examples
    --------
    >>> newaxis is None
    True
    >>> x = np.arange(3)
    >>> x
    array([0, 1, 2])
    >>> x[:,newaxis]
    array([[0],
    [1],
    [2]])
    >>> x[:,newaxis,newaxis]
    array([[[0]],
    [[1]],
    [[2]]])
    >>> x[:,newaxis] * x
    array([[0, 0, 0],
    [0, 1, 2],
    [0, 2, 4]])

    Outer product, same as outer(x,y):

    >>> y = np.arange(3,6)
    >>> x[:,newaxis] * y
    array([[ 0,  0,  0],
    [ 3,  4,  5],
    [ 6,  8, 10]])

    x[newaxis,:] is equivalent to x[newaxis] and x[None]:

    >>> x[newaxis,:].shape
    (1, 3)
    >>> x[newaxis].shape
    (1, 3)
    >>> x[None].shape
    (1, 3)
    >>> x[:,newaxis].shape
    (3, 1)

    """)

if __doc__:
    constants_str = []
    constants.sort()
    for name, doc in constants:
        constants_str.append(""".. const:: %s\n    %s""" % (
            name, textwrap.dedent(doc).replace("\n", "\n    ")))
    constants_str = "\n".join(constants_str)

    __doc__ = __doc__ % dict(constant_list=constants_str)
    del constants_str, name, doc

del constants, add_newdoc
