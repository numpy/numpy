from __future__ import division, absolute_import, print_function

import numpy as np


def check_dir(module, module_name=None):
    """Returns a mapping of all objects with the wrong __module__ attribute."""
    if module_name is None:
        module_name = module.__name__
    results = {}
    for name in dir(module):
        item = getattr(module, name)
        if (hasattr(item, '__module__') and hasattr(item, '__name__')
                and item.__module__ != module_name):
            results[name] = item.__module__ + '.' + item.__name__
    return results


def test_numpy_namespace():
    # None of these objects are publicly documented.
    whitelist = {
        'Tester': 'numpy.testing._private.nosetester.NoseTester',
        '_add_newdoc_ufunc': 'numpy.core._multiarray_umath._add_newdoc_ufunc',
        'add_docstring': 'numpy.core._multiarray_umath.add_docstring',
        'add_newdoc': 'numpy.core.function_base.add_newdoc',
        'add_newdoc_ufunc': 'numpy.core._multiarray_umath._add_newdoc_ufunc',
        'bool': 'builtins.bool',
        'byte_bounds': 'numpy.lib.utils.byte_bounds',
        'compare_chararrays': 'numpy.core._multiarray_umath.compare_chararrays',
        'complex': 'builtins.complex',
        'deprecate': 'numpy.lib.utils.deprecate',
        'deprecate_with_doc': 'numpy.lib.utils.<lambda>',
        'disp': 'numpy.lib.function_base.disp',
        'fastCopyAndTranspose': 'numpy.core._multiarray_umath._fastCopyAndTranspose',
        'float': 'builtins.float',
        'get_array_wrap': 'numpy.lib.shape_base.get_array_wrap',
        'get_include': 'numpy.lib.utils.get_include',
        'int': 'builtins.int',
        'int_asbuffer': 'numpy.core._multiarray_umath.int_asbuffer',
        'long': 'builtins.int',
        'mafromtxt': 'numpy.lib.npyio.mafromtxt',
        'maximum_sctype': 'numpy.core.numerictypes.maximum_sctype',
        'ndfromtxt': 'numpy.lib.npyio.ndfromtxt',
        'object': 'builtins.object',
        'recfromcsv': 'numpy.lib.npyio.recfromcsv',
        'recfromtxt': 'numpy.lib.npyio.recfromtxt',
        'safe_eval': 'numpy.lib.utils.safe_eval',
        'set_string_function': 'numpy.core.arrayprint.set_string_function',
        'show_config': 'numpy.__config__.show',
        'str': 'builtins.str',
        'unicode': 'builtins.str',
        'who': 'numpy.lib.utils.who',
    }
    bad_results = check_dir(np)
    # pytest gives better error messages with the builtin assert than with
    # assert_equal
    assert bad_results == whitelist


def test_numpy_linalg():
    bad_results = check_dir(np.linalg)
    assert bad_results == {}


def test_numpy_fft():
    bad_results = check_dir(np.fft)
    assert bad_results == {}
