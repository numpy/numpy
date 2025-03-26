.. currentmodule:: numpy

*********
Constants
*********

NumPy includes several constants:

.. data:: e

    Euler's constant, base of natural logarithms, Napier's constant.

    ``e = 2.71828182845904523536028747135266249775724709369995...``

    .. rubric:: See Also

    exp : Exponential function
    log : Natural logarithm

    .. rubric:: References

    https://en.wikipedia.org/wiki/E_%28mathematical_constant%29


.. data:: euler_gamma

    ``γ = 0.5772156649015328606065120900824024310421...``

    .. rubric:: References

    https://en.wikipedia.org/wiki/Euler%27s_constant


.. data:: inf

    IEEE 754 floating point representation of (positive) infinity.

    .. rubric:: Returns

    y : float
        A floating point representation of positive infinity.

    .. rubric:: See Also

    isinf : Shows which elements are positive or negative infinity

    isposinf : Shows which elements are positive infinity

    isneginf : Shows which elements are negative infinity

    isnan : Shows which elements are Not a Number

    isfinite : Shows which elements are finite (not one of Not a Number,
    positive infinity and negative infinity)

    .. rubric:: Notes

    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Also that positive infinity is not equivalent to negative infinity. But
    infinity is equivalent to positive infinity.

    .. rubric:: Examples

.. try_examples::

    >>> import numpy as np
    >>> np.inf
    inf
    >>> np.array([1]) / 0.
    array([inf])


.. data:: nan

    IEEE 754 floating point representation of Not a Number (NaN).

    .. rubric:: Returns

    y : A floating point representation of Not a Number.

    .. rubric:: See Also

    isnan : Shows which elements are Not a Number.

    isfinite : Shows which elements are finite (not one of
    Not a Number, positive infinity and negative infinity)

    .. rubric:: Notes

    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.

    .. rubric:: Examples

.. try_examples::

    >>> import numpy as np
    >>> np.nan
    nan
    >>> np.log(-1)
    np.float64(nan)
    >>> np.log([-1, 1, 2])
    array([       nan, 0.        , 0.69314718])


.. data:: newaxis

    A convenient alias for None, useful for indexing arrays.

    .. rubric:: Examples

.. try_examples::

    >>> import numpy as np
    >>> np.newaxis is None
    True
    >>> x = np.arange(3)
    >>> x
    array([0, 1, 2])
    >>> x[:, np.newaxis]
    array([[0],
    [1],
    [2]])
    >>> x[:, np.newaxis, np.newaxis]
    array([[[0]],
    [[1]],
    [[2]]])
    >>> x[:, np.newaxis] * x
    array([[0, 0, 0],
        [0, 1, 2],
        [0, 2, 4]])

    Outer product, same as ``outer(x, y)``:

    >>> y = np.arange(3, 6)
    >>> x[:, np.newaxis] * y
    array([[ 0,  0,  0],
        [ 3,  4,  5],
        [ 6,  8, 10]])

    ``x[np.newaxis, :]`` is equivalent to ``x[np.newaxis]`` and ``x[None]``:

    >>> x[np.newaxis, :].shape
    (1, 3)
    >>> x[np.newaxis].shape
    (1, 3)
    >>> x[None].shape
    (1, 3)
    >>> x[:, np.newaxis].shape
    (3, 1)


.. data:: pi

    ``pi = 3.1415926535897932384626433...``

    .. rubric:: References

    https://en.wikipedia.org/wiki/Pi
