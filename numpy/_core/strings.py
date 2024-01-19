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
    isalpha,
    isdigit,
    isspace,
    isdecimal,
    isnumeric,
    str_len,
    find as _find_ufunc,
    rfind as _rfind_ufunc,
    count as _count_ufunc,
    startswith as _startswith_ufunc,
    endswith as _endswith_ufunc,
    _lstrip_whitespace,
    _lstrip_chars,
    _rstrip_whitespace,
    _rstrip_chars,
    _strip_whitespace,
    _strip_chars,
    _replace,
)


__all__ = [
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "add", "isalpha", "isdigit", "isspace", "isdecimal", "isnumeric",
    "str_len", "find", "rfind", "count", "startswith", "endswith",
    "lstrip", "rstrip", "strip", "replace"
]


MAX = np.iinfo(np.int64).max


def find(a, sub, start=0, end=None):
    """
    For each element, return the lowest index in the string where
    substring ``sub`` is found, such that ``sub`` is contained in the
    range [``start``, ``end``).

    Parameters
    ----------
    a : array_like, with `np.bytes_` or `np.str_` dtype

    sub : array_like, with `np.bytes_` or `np.str_` dtype
        The substring to search for.

    start, end : array_like, with any integer dtype
        The range to look in, interpreted as in slice notation.

    Returns
    -------
    y : ndarray
        Output array of ints

    See Also
    --------
    str.find

    Examples
    --------
    >>> a = np.array(["NumPy is a Python library"])
    >>> np.strings.find(a, "Python")
    array([11])

    """
    end = end if end is not None else MAX
    return _find_ufunc(a, sub, start, end)


def rfind(a, sub, start=0, end=None):
    """
    For each element, return the highest index in the string where
    substring ``sub`` is found, such that ``sub`` is contained in the
    range [``start``, ``end``).

    Parameters
    ----------
    a : array_like, with `np.bytes_` or `np.str_` dtype

    sub : array_like, with `np.bytes_` or `np.str_` dtype
        The substring to search for.

    start, end : array_like, with any integer dtype
        The range to look in, interpreted as in slice notation.

    Returns
    -------
    y : ndarray
        Output array of ints

    See Also
    --------
    str.rfind

    """
    end = end if end is not None else MAX
    return _rfind_ufunc(a, sub, start, end)


def count(a, sub, start=0, end=None):
    """
    Returns an array with the number of non-overlapping occurrences of
    substring ``sub`` in the range [``start``, ``end``).

    Parameters
    ----------
    a : array_like, with `np.bytes_` or `np.str_` dtype

    sub : array_like, with `np.bytes_` or `np.str_` dtype
       The substring to search for.

    start, end : array_like, with any integer dtype
        The range to look in, interpreted as in slice notation.

    Returns
    -------
    y : ndarray
        Output array of ints

    See Also
    --------
    str.count

    Examples
    --------
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> c
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.strings.count(c, 'A')
    array([3, 1, 1])
    >>> np.strings.count(c, 'aA')
    array([3, 1, 0])
    >>> np.strings.count(c, 'A', start=1, end=4)
    array([2, 1, 1])
    >>> np.strings.count(c, 'A', start=1, end=3)
    array([1, 0, 0])

    """
    end = end if end is not None else MAX
    return _count_ufunc(a, sub, start, end)


def startswith(a, prefix, start=0, end=None):
    """
    Returns a boolean array which is `True` where the string element
    in ``a`` starts with ``prefix``, otherwise `False`.

    Parameters
    ----------
    a : array_like, with `np.bytes_` or `np.str_` dtype

    prefix : array_like, with `np.bytes_` or `np.str_` dtype

    start, end : array_like, with any integer dtype
        With ``start``, test beginning at that position. With ``end``,
        stop comparing at that position.

    Returns
    -------
    out : ndarray
        Output array of bools

    See Also
    --------
    str.startswith

    """
    end = end if end is not None else MAX
    return _startswith_ufunc(a, prefix, start, end)


def endswith(a, suffix, start=0, end=None):
    """
    Returns a boolean array which is `True` where the string element
    in ``a`` ends with ``suffix``, otherwise `False`.

    Parameters
    ----------
    a : array_like, with `np.bytes_` or `np.str_` dtype

    suffix : array_like, with `np.bytes_` or `np.str_` dtype

    start, end : array_like, with any integer dtype
        With ``start``, test beginning at that position. With ``end``,
        stop comparing at that position.

    Returns
    -------
    out : ndarray
        Output array of bools

    See Also
    --------
    str.endswith

    Examples
    --------
    >>> s = np.array(['foo', 'bar'])
    >>> s
    array(['foo', 'bar'], dtype='<U3')
    >>> np.strings.endswith(s, 'ar')
    array([False,  True])
    >>> np.strings.endswith(s, 'a', start=1, end=2)
    array([False,  True])

    """
    end = end if end is not None else MAX
    return _endswith_ufunc(a, suffix, start, end)


def lstrip(a, chars=None):
    """
    For each element in `a`, return a copy with the leading characters
    removed.

    Parameters
    ----------
    a : array_like, with ``bytes_`` or ``unicode_`` dtype
    chars : scalar with the same dtype as ``a``, optional
       The ``chars`` argument is a string specifying the set of
       characters to be removed. If ``None``, the ``chars``
       argument defaults to removing whitespace. The ``chars`` argument
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
    if chars is None:
        return _lstrip_whitespace(a)
    return _lstrip_chars(a, chars)


def rstrip(a, chars=None):
    """
    For each element in `a`, return a copy with the trailing characters
    removed.

    Parameters
    ----------
    a : array_like, with ``bytes_`` or ``unicode_`` dtype
    chars : scalar with the same dtype as ``a``, optional
       The ``chars`` argument is a string specifying the set of
       characters to be removed. If ``None``, the ``chars``
       argument defaults to removing whitespace. The ``chars`` argument
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
    if chars is None:
        return _rstrip_whitespace(a)
    return _rstrip_chars(a, chars)


def strip(a, chars=None):
    """
    For each element in `a`, return a copy with the leading and
    trailing characters removed.

    Parameters
    ----------
    a : array_like, with ``bytes_`` or ``unicode_`` dtype
    chars : scalar with the same dtype as ``a``, optional
       The ``chars`` argument is a string specifying the set of
       characters to be removed. If ``None``, the ``chars``
       argument defaults to removing whitespace. The ``chars`` argument
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
    if chars is None:
        return _strip_whitespace(a)
    return _strip_chars(a, chars)


def replace(a, old, new, count=-1):
    """
    For each element in ``a``, return a copy of the string with
    occurrences of substring ``old`` replaced by ``new``.

    Parameters
    ----------
    a : array_like, with ``bytes_`` or ``str_`` dtype

    old, new : array_like, with ``bytes_`` or ``str_`` dtype

    count : array_like, with ``int_`` dtype
        If the optional argument ``count`` is given, only the first
        ``count`` occurrences are replaced.

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
    from numpy._core.umath import count as count_occurences

    arr = np.asanyarray(a)
    old = np.asanyarray(old)
    new = np.asanyarray(new)

    max_int64 = np.iinfo(np.int64).max
    counts = count_occurences(arr, old, 0, max_int64)
    count = np.asanyarray(count)
    counts = np.where(count < 0, counts, np.minimum(counts, count))

    buffersizes = str_len(arr) + counts * (str_len(new) - str_len(old))

    # buffersizes is properly broadcast along all inputs.
    out = np.empty_like(arr, shape=buffersizes.shape,
                        dtype=f"{arr.dtype.char}{buffersizes.max()}")
    return _replace(arr, old, new, counts, out=out)
