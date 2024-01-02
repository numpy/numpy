"""
This module contains a set of functions for vectorized string
operations.
"""

import numpy as np
from numpy import (
    equal, not_equal, less, less_equal, greater, greater_equal,
    add
)
from numpy._core.umath import (
    isalpha, isdigit, isspace, isdecimal, isnumeric,
    str_len, find, rfind, count, startswith, endswith,
    _lstrip_whitespace, _lstrip_chars, _rstrip_whitespace,
    _rstrip_chars, _strip_whitespace, _strip_chars,
    _replace
)


__all__ = [
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "add", "isalpha", "isdigit", "isspace", "isdecimal", "isnumeric",
    "str_len", "find", "rfind", "count", "startswith", "endswith",
    "lstrip", "rstrip", "strip", "replace"
]


def lstrip(x1, x2=None):
    """
    For each element in `x1`, return a copy with the leading characters
    removed.

    Parameters
    ----------
    x1 : array_like, with ``bytes_`` or ``unicode_`` dtype
    x2 : scalar with the same dtype as ``x1``, optional
       The ``x1`` argument is a string specifying the set of
       characters to be removed. If ``None``, the ``x2``
       argument defaults to removing whitespace. The ``x2`` argument
       is not a prefix or suffix; rather, all combinations of its
       values are stripped.

    Returns
    -------
    out : ndarray
        Output array of ``bytes_`` or ``unicode_`` dtype

    See Also
    --------
    str.lstrip

    Examples
    --------
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> c
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    # The 'a' variable is unstripped from c[1] because of leading whitespace.
    >>> np.strings.lstrip(c, 'a')
    array(['AaAaA', '  aA  ', 'bBABba'], dtype='<U7')
    >>> np.strings.lstrip(c, 'A') # leaves c unchanged
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> (np.strings.lstrip(c, ' ') == np.strings.lstrip(c, '')).all()
    np.False_
    >>> (np.strings.lstrip(c, ' ') == np.strings.lstrip(c)).all()
    np.True_

    """
    if x2 is None:
        return _lstrip_whitespace(x1)
    return _lstrip_chars(x1, x2)


def rstrip(x1, x2=None):
    """
    For each element in `x1`, return a copy with the trailing characters
    removed.

    Parameters
    ----------
    x1 : array_like, with ``bytes_`` or ``unicode_`` dtype
    x2 : scalar with the same dtype as ``x1``, optional
       The ``x1`` argument is a string specifying the set of
       characters to be removed. If ``None``, the ``x2``
       argument defaults to removing whitespace. The ``x2`` argument
       is not a prefix or suffix; rather, all combinations of its
       values are stripped.

    Returns
    -------
    out : ndarray
        Output array of ``bytes_`` or ``unicode_`` dtype

    See Also
    --------
    str.rstrip

    Examples
    --------
    >>> c = np.array(['aAaAaA', 'abBABba'])
    >>> c
    array(['aAaAaA', 'abBABba'], dtype='<U7')
    >>> np.strings.rstrip(c, 'a')
    array(['aAaAaA', 'abBABb'], dtype='<U7')
    >>> np.strings.rstrip(c, 'A')
    array(['aAaAa', 'abBABba'], dtype='<U7')

    """
    if x2 is None:
        return _rstrip_whitespace(x1)
    return _rstrip_chars(x1, x2)


def strip(x1, x2=None):
    """
    For each element in `x1`, return a copy with the leading and
    trailing characters removed.

    Parameters
    ----------
    x1 : array_like, with ``bytes_`` or ``unicode_`` dtype
    x2 : scalar with the same dtype as ``x1``, optional
       The ``x1`` argument is a string specifying the set of
       characters to be removed. If ``None``, the ``x2``
       argument defaults to removing whitespace. The ``x2`` argument
       is not a prefix or suffix; rather, all combinations of its
       values are stripped.

    Returns
    -------
    out : ndarray
        Output array of ``bytes_`` or ``unicode_`` dtype

    See Also
    --------
    str.strip

    Examples
    --------
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> c
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.strings.strip(c)
    array(['aAaAaA', 'aA', 'abBABba'], dtype='<U7')
    # 'a' unstripped from c[1] because of leading whitespace.
    >>> np.strings.strip(c, 'a')
    array(['AaAaA', '  aA  ', 'bBABb'], dtype='<U7')
    # 'A' unstripped from c[1] because of trailing whitespace.
    >>> np.strings.strip(c, 'A')
    array(['aAaAa', '  aA  ', 'abBABba'], dtype='<U7')

    """
    if x2 is None:
        return _strip_whitespace(x1)
    return _strip_chars(x1, x2)


def replace(x1, x2, x3, x4):
    """
    For each element in ``x1``, return a copy of the string with all
    occurrences of substring ``x2`` replaced by ``x3``.

    Parameters
    ----------
    x1 : array_like, with ``bytes_`` or ``unicode_`` dtype

    x2, x3 : array_like, with ``bytes_`` or ``unicode_`` dtype

    x4 : int
        If the optional argument ``x4`` is given, only the first
        ``x4`` occurrences are replaced.

    Returns
    -------
    out : ndarray
        Output array of ``str_`` or ``bytes_`` dtype

    See Also
    --------
    str.replace
    
    Examples
    --------
    >>> a = np.array(["That is a mango", "Monkeys eat mangos"])
    >>> np.strings.replace(a, 'mango', 'banana')
    array(['That is a banana', 'Monkeys eat bananas'], dtype='<U19')

    >>> a = np.array(["The dish is fresh", "This is it"])
    >>> np.strings.replace(a, 'is', 'was')
    array(['The dwash was fresh', 'Thwas was it'], dtype='<U19')
    
    """
    x1_arr = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    x3 = np.asanyarray(x3)

    max_int64 = np.iinfo(np.int64).max
    counts = count(x1_arr, x2, 0, max_int64)
    x4 = np.asanyarray(x4)
    counts = np.where(x4 < 0, counts,
                            np.minimum(counts, x4))

    buffersizes = str_len(x1_arr) + counts * (str_len(x3) - str_len(x2))

    # buffersizes is properly broadcast along all inputs.
    out = np.empty_like(x1_arr, shape=buffersizes.shape,
                           dtype=f"{x1_arr.dtype.char}{buffersizes.max()}")
    _replace(x1_arr, x2, x3, counts, out=out)
    return out
