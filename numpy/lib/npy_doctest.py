from doctest import OutputChecker
import doctest
import re
import numpy as np

# This file contains a custom doctest runner for use in downstream projects.
# It makes doctests less sensitive to details of how numpy prints array
# elements.
#
# To use it, you must set doctest's OutputChecker:
#
#    >>> doctest.OutputChecker = np.lib.npy_doctest.FlexNumOutputChecker
#
# and then use the `NPY_FLEX_NUM` doctest option, either when
# initializing doctest or as a doctest directive on individual doctests.
#
# For floating-point values, you can also set the desired maximum relative
# difference using the `fprec` variable:
#
#    >>> np.lib.npy_doctest.fprec = 1e-3
#
# which will make floats a,b be considered equal as long as
# `abs(a-b)/a > fprec`. This is useful to avoid mismatches due to small
# numerical differences on different architectures and platforms.
#
# The runner is implemented so the tests are insensitive to whitespace around
# numbers, and small numerical differences in floating-point numbers. In the
# case of larger changes to array printing in numpy we will try to update the
# doctest runner in tandem to account for these changes, if possible.
#
# This file also has a function `flex_num_match` to compare two
# strings, in the same way as the doctest does.


# Implementation Notes:
# ---------------------
# The doctest module allows us to compare two strings, 'wants' vs 'got', one of
# which is the stdout of a python expression, and the other is the
# result-string in a doctest. Since we do not get access to the python-object
# result of the doctest expression, we have to do some string parsing.
#
# The strategy here is to take care of *all* substrings that look like numbers.
# A downside is we do not cover other numpy types like strings, datetime,
# object, etc.
#
# An alternate strategy could be to look for array-like patterns, eg
# array(xxx) or [xx,xx,xx] and allow for spaces at the right points,
# but this seems tricky because of subclasses and because other python
# objects use this notation (eg, python lists).
#
# While it may have been nice to use a more general parser like pyparsing, this
# implementation uses regex to avoid outside dependencies.


fprec = 1e-3  # this is user visible/customizable

def _float_mismatch(a, b):
    if not np.isfinite(a) or not np.isfinite(b):
        if np.isnan(a) and np.isnan(b):
            return False
        return a != b
    if a == b == 0 or abs(a-b)/a < fprec:
        return False
    return True

def _complex_str_mismatch(a, b):
    # numpy supports spaces betwee real/imag, but not python complex
    a, b = a.replace(' ', ''), b.replace(' ', '')

    a, b = complex(a.replace('...', '')), complex(b.replace('...', ''))
    return _float_mismatch(a.real, b.real) or _float_mismatch(a.imag, b.imag)

def _float_str_mismatch(a, b):
    a, b = float(a.replace('...', '')), float(b.replace('...', ''))
    return _float_mismatch(a, b)


# These patterns detect spaces/punctuation before/after a number
_punct = '[\[\]\(\),]'
_before = '(?:\s+|^|(?<={}))'.format(_punct)
_after = '(?:\s+|$|(?={}))'.format(_punct)
_mkpat = lambda s: re.compile(_before + s + _after)
# Note: We require a digit *before* the decimal point, and don't require any
# after (numpy-specific).
# We allow ellipsis (...) in fractional digits.
# We allow spaces to separate the parts of complex numbers.
_floatpat = '(?:\d+\.\d*(?:\.\.\.)?(?:e[+-]?\d+)?|inf|nan)'
# Define a regex pattern, and a mismatch function, for each type.
_pats = {
        'complex': (
            _mkpat('(?:[+-]?' + _floatpat + ')?\s*[+-]' + _floatpat + 'j'),
            _complex_str_mismatch),
        'float':   (
            _mkpat('[+-]?' + _floatpat),
            _float_str_mismatch),
        'int':     (
            _mkpat('[+-]?\d+'),
            lambda a, b: int(a) != int(b)),
        'bool':    (
            _mkpat('(?:True|False)'),
            lambda a, b: bool(a) == bool(b))}
# WIP: conceivably we might want to include */+-&|%  in _punct. However, if our
# goal is to parse numpy array output, and not arbitrary mathematical
# expressions, the _punct above should be enough.

def flex_num_match(want, got):
    """
    Tests whether the numeric content in the `want` and `got` strings are
    similar, allowing for variations in whitespace around numbers and small
    numeric differences in floating-point values.

    Parameters
    ----------
    want : string
        Expected string. Allows '...' to trail in floating-point fractional
        digits, which match any digits.
    got : string
        String to compare.

    Returns
    -------
    want, got : tuple of strings
        Input strings with numeric patterns (which include surrounding
        whitespace) replaced by a single white-space (' ')

    Raises
    ------
    AssertionError
        If the numeric patterned parts of the strings did not match.
    """
    for tp in ['complex', 'float', 'int', 'bool']:
        pat, mismatch = _pats[tp]

        want_elem = pat.findall(want)
        got_elem = pat.findall(got)

        assert(len(want_elem) == len(got_elem))
        if len(want_elem) == 0:
            continue
        for a,b in zip(want_elem, got_elem):
            assert(not mismatch(a,b))

        # replace with a space to keep further tokens separate
        want = pat.sub(' ', want)
        got = pat.sub(' ', got)
    return want, got

NPY_FLEX_NUMS = doctest.register_optionflag("NPY_FLEX_NUMS")

class FlexNumOutputChecker(doctest.OutputChecker, object):
    def check_output(self, want, got, optionflags):
        if optionflags & NPY_FLEX_NUMS:
            try:
                want, got = flex_num_match(want, got)
            except AssertionError:
                return False
        sup = super(FlexNumOutputChecker, self)
        return sup.check_output(want, got, optionflags)

    # WIP: implement output_difference?
