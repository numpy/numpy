"""
Back compatibility utils module. It will import the appropriate
set of tools

"""
__all__ = [
        'assert_equal', 'assert_almost_equal', 'assert_approx_equal',
        'assert_array_equal', 'assert_array_less', 'assert_string_equal',
        'assert_array_almost_equal', 'assert_raises', 'build_err_msg',
        'decorate_methods', 'jiffies', 'memusage', 'print_assert_equal',
        'raises', 'rand', 'rundocs', 'runstring', 'verbose', 'measure',
        'assert_', 'assert_array_almost_equal_nulp', 'assert_raises_regex',
        'assert_array_max_ulp', 'assert_warns', 'assert_no_warnings',
        'assert_allclose', 'IgnoreException', 'clear_and_catch_warnings',
        'SkipTest', 'KnownFailureException', 'temppath', 'tempdir', 'IS_PYPY',
        'HAS_REFCOUNT', 'suppress_warnings', 'assert_array_compare',
        '_assert_valid_refcount', '_gen_alignment_data',
        ]

from .nose_tools.utils import (
    HAS_REFCOUNT, IS_PYPY, IgnoreException, KnownFailureException,
    KnownFailureTest, SkipTest, StringIO, WarningManager, WarningMessage,
    absolute_import, arange, array, array_repr, assert_, assert_allclose,
    assert_almost_equal, assert_approx_equal, assert_array_almost_equal,
    assert_array_almost_equal_nulp, assert_array_compare, assert_array_equal,
    assert_array_less, assert_array_max_ulp, assert_equal, assert_no_warnings,
    assert_raises, assert_raises_regex, assert_string_equal, assert_warns,
    build_err_msg, clear_and_catch_warnings, contextlib, decorate_methods,
    deprecate, division, empty, float32, gisfinite, gisinf, gisnan,
    import_nose, integer_repr, isnat, jiffies, measure, memusage, mkdtemp,
    mkstemp, ndarray, nulp_diff, operator, os, partial, print_assert_equal,
    print_function, raises, rand, re, rundocs, runstring, shutil,
    suppress_warnings, sys, tempdir, temppath, verbose, warnings, wraps,
    _assert_valid_refcount, _gen_alignment_data)
