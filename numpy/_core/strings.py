"""
This module contains a set of functions for vectorized string
operations.
"""

from numpy import (
    equal, not_equal, less, less_equal, greater, greater_equal,
    add
)
from numpy._core.umath import (
    isalpha, isdigit, isspace, isdecimal, isnumeric,
    str_len, find, rfind, count, startswith, endswith,
    _lstrip_whitespace, _lstrip_chars, _rstrip_whitespace,
    _rstrip_chars, _strip_whitespace, _strip_chars,
)


__all__ = [
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "add", "isalpha", "isdigit", "isspace", "isdecimal", "isnumeric",
    "str_len", "find", "rfind", "count", "startswith", "endswith",
    "lstrip", "rstrip", "strip",
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
