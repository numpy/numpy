"""
Create the numpy._core.umath namespace for backward compatibility. In v1.16
the multiarray and umath c-extension modules were merged into a single
_multiarray_umath extension module. So we replicate the old namespace
by importing from the extension module.

"""

from . import _multiarray_umath
from ._multiarray_umath import *  # noqa: F403
# These imports are needed for backward compatibility,
# do not change them. issue gh-11862
# _ones_like is semi-public, on purpose not added to __all__
from ._multiarray_umath import (
    _UFUNC_API, _add_newdoc_ufunc, _ones_like, _get_extobj_dict, _make_extobj,
    _extobj_contextvar)
# These imports are needed for the strip implementation
from ._multiarray_umath import (
    _strip_whitespace, _lstrip_whitespace, _rstrip_whitespace, _strip_chars,
    _lstrip_chars, _rstrip_chars)

__all__ = [
    'absolute', 'add',
    'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh',
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'cbrt', 'ceil', 'conj',
    'conjugate', 'copysign', 'cos', 'cosh', 'bitwise_count', 'deg2rad',
    'degrees', 'divide', 'divmod', 'e', 'equal', 'euler_gamma', 'exp', 'exp2',
    'expm1', 'fabs', 'floor', 'floor_divide', 'float_power', 'fmax', 'fmin',
    'fmod', 'frexp', 'frompyfunc', 'gcd', 'greater', 'greater_equal',
    'heaviside', 'hypot', 'invert', 'isfinite', 'isinf', 'isnan', 'isnat',
    'lcm', 'ldexp', 'left_shift', 'less', 'less_equal', 'log', 'log10',
    'log1p', 'log2', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_not',
    'logical_or', 'logical_xor', 'maximum', 'minimum', 'mod', 'modf',
    'multiply', 'negative', 'nextafter', 'not_equal', 'pi', 'positive',
    'power', 'rad2deg', 'radians', 'reciprocal', 'remainder', 'right_shift',
    'rint', 'sign', 'signbit', 'sin', 'sinh', 'spacing', 'sqrt', 'square',
    'subtract', 'tan', 'tanh', 'true_divide', 'trunc']


def strip(x1, x2=None):
    """
    For each element in `x1`, return a copy with the leading and
    trailing characters removed.

    Parameters
    ----------
    x1 : array_like, with ``bytes_`` or ``unicode_`` dtype
    x2 : array_like, with ``bytes_`` or ``unicode_`` dtype, optional
       The `x2` argument is a string specifying the set of
       characters to be removed. If None, the `x2`
       argument defaults to removing whitespace. The `x2` argument
       is not a prefix or suffix; rather, all combinations of its
       values are stripped.
       $PARAMS

    Returns
    -------
    out : ndarray
        Output array of ``bytes_`` or ``unicode_`` dtype
        $OUT_SCALAR_2

    See Also
    --------
    str.strip

    Examples
    --------
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> c
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.char.strip(c)
    array(['aAaAaA', 'aA', 'abBABba'], dtype='<U7')
    'a' unstripped from c[1] because ws leads
    >>> np.char.strip(c, 'a')
    array(['AaAaA', '  aA  ', 'bBABb'], dtype='<U7')
    # 'A' unstripped from c[1] because ws trails
    >>> np.char.strip(c, 'A')
    array(['aAaAa', '  aA  ', 'abBABba'], dtype='<U7')

    """
    if x2 is not None:
        return _strip_chars(x1, x2)
    return _strip_whitespace(x1)


def lstrip(x1, x2=None):
    """
    For each element in `x1`, return a copy with the leading characters
    removed.

    Parameters
    ----------
    x1 : array_like, with ``bytes_`` or ``unicode_`` dtype
    x2 : array_like, with ``bytes_`` or ``unicode_`` dtype, optional
       The `x2` argument is a string specifying the set of
       characters to be removed. If None, the `x2`
       argument defaults to removing whitespace. The `x2` argument
       is not a prefix or suffix; rather, all combinations of its
       values are stripped.
       $PARAMS

    Returns
    -------
    out : ndarray
        Output array of ``bytes_`` or ``unicode_`` dtype
        $OUT_SCALAR_2

    See Also
    --------
    str.lstrip

    Examples
    --------
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> c
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    The 'a' variable is unstripped from c[1] because whitespace leading.
    >>> np.char.lstrip(c, 'a')
    array(['AaAaA', '  aA  ', 'bBABba'], dtype='<U7')
    >>> np.char.lstrip(c, 'A') # leaves c unchanged
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> (np.char.lstrip(c, ' ') == np.char.lstrip(c, '')).all()
    False
    >>> (np.char.lstrip(c, ' ') == np.char.lstrip(c)).all()
    True

    """
    if x2 is not None:
        return _lstrip_chars(x1, x2)
    return _lstrip_whitespace(x1)


def rstrip(x1, x2=None):
    """
    For each element in `x1`, return a copy with the trailing characters
    removed.

    Parameters
    ----------
    x1 : array_like, with ``bytes_`` or ``unicode_`` dtype
    x2 : scalar with ``bytes_`` or ``unicode_`` dtype, optional
       The `x2` argument is a string specifying the set of
       characters to be removed. If None, the `x2`
       argument defaults to removing whitespace. The `x2` argument
       is not a prefix or suffix; rather, all combinations of its
       values are stripped.
       $PARAMS

    Returns
    -------
    out : ndarray
        Output array of ``bytes_`` or ``unicode_`` dtype
        $OUT_SCALAR_2

    See Also
    --------
    str.rstrip

    Examples
    --------
    >>> c = np.array(['aAaAaA', 'abBABba'], dtype='S7'); c
    array(['aAaAaA', 'abBABba'], dtype='|S7')
    >>> np.char.rstrip(c, b'a')
    array(['aAaAaA', 'abBABb'], dtype='|S7')
    >>> np.char.rstrip(c, b'A')
    array(['aAaAa', 'abBABba'], dtype='|S7')

    """
    if x2 is not None:
        return _rstrip_chars(x1, x2)
    return _rstrip_whitespace(x1)
