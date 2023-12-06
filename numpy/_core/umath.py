"""
Create the numpy._core.umath namespace for backward compatibility. In v1.16
the multiarray and umath c-extension modules were merged into a single
_multiarray_umath extension module. So we replicate the old namespace
by importing from the extension module.

"""

import numpy
from . import _multiarray_umath
from ._multiarray_umath import *  # noqa: F403
from ._multiarray_umath import _replace_impl
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


def replace(x1, x2, x3, x4):
    """
    For each element in `x1`, return a copy of the string with all
    occurrences of substring `x2` replaced by `x3`.
    Parameters
    ----------
    x1 : array_like, with ``bytes_`` or ``unicode_`` dtype
    x2 : array_like, with ``bytes_`` or ``unicode_`` dtype
    x3 : array_like, with ``bytes_`` or ``unicode_`` dtype
    x4 : array_like, with ``int_`` dtype
        If the optional argument `x4` is given, only the first
        `x4` occurrences are replaced.
        $PARAMS
    Returns
    -------
    y : ndarray
        Output array of str or unicode, depending on input type
        $OUT_SCALAR_2
    See Also
    --------
    str.replace
    
    Examples
    --------
    >>> a = np.array(["That is a mango", "Monkeys eat mangos"])
    >>> np.core.umath.replace(a, 'mango', 'banana')
    array(['That is a banana', 'Monkeys eat bananas'], dtype='<U19')
    >>> a = np.array(["The dish is fresh", "This is it"])
    >>> np.core.umath.replace(a, 'is', 'was')
    array(['The dwash was fresh', 'Thwas was it'], dtype='<U19')
    """
    counts = count(x1, x2, 0, numpy.iinfo(numpy.int_).max)
    buffersizes = str_len(x1) + counts * (str_len(x3)-str_len(x2))
    max_buffersize = numpy.max(buffersizes)
    out = numpy.empty(x1.shape, dtype=f"{x1.dtype.char}{max_buffersize}")
    _replace_impl(x1, x2, x3, x4, out=out)
    return out
