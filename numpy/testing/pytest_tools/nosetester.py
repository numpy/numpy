"""
Nose test running.

This module implements ``test()`` and ``bench()`` functions for NumPy modules.

"""
from __future__ import division, absolute_import, print_function

import os
import sys
import warnings
from numpy.compat import basestring
import numpy as np

from .utils import import_nose, suppress_warnings


__all__ = ['get_package_name', 'run_module_suite', 'get_package_name',
           'import_nose', 'suppress_warnings']


def get_package_name(filepath):
    """
    Given a path where a package is installed, determine its name.

    Parameters
    ----------
    filepath : str
        Path to a file. If the determination fails, "numpy" is returned.

    Examples
    --------
    >>> np.testing.nosetester.get_package_name('nonsense')
    'numpy'

    """

    fullpath = filepath[:]
    pkg_name = []
    while 'site-packages' in filepath or 'dist-packages' in filepath:
        filepath, p2 = os.path.split(filepath)
        if p2 in ('site-packages', 'dist-packages'):
            break
        pkg_name.append(p2)

    # if package name determination failed, just default to numpy/scipy
    if not pkg_name:
        if 'scipy' in fullpath:
            return 'scipy'
        else:
            return 'numpy'

    # otherwise, reverse to get correct order and return
    pkg_name.reverse()

    # don't include the outer egg directory
    if pkg_name[0].endswith('.egg'):
        pkg_name.pop(0)

    return '.'.join(pkg_name)


def run_module_suite(file_to_run=None, argv=None):
    """
    Run a test module.

    Equivalent to calling ``$ nosetests <argv> <file_to_run>`` from
    the command line. This version is for pytest rather than nose.

    Parameters
    ----------
    file_to_run : str, optional
        Path to test module, or None.
        By default, run the module from which this function is called.
    argv : list of strings
        Arguments to be passed to the pytest runner. ``argv[0]`` is
        ignored. All command line arguments accepted by ``pytest``
        will work. If it is the default value None, sys.argv is used.

        .. versionadded:: 1.14.0

    Examples
    --------
    Adding the following::

        if __name__ == "__main__" :
            run_module_suite(argv=sys.argv)

    at the end of a test module will run the tests when that module is
    called in the python interpreter.

    Alternatively, calling::

    >>> run_module_suite(file_to_run="numpy/tests/test_matlib.py")

    from an interpreter will run all the test routine in 'test_matlib.py'.
    """
    import pytest
    if file_to_run is None:
        f = sys._getframe(1)
        file_to_run = f.f_locals.get('__file__', None)
        if file_to_run is None:
            raise AssertionError

    if argv is None:
        argv = sys.argv[1:] + [file_to_run]
    else:
        argv = argv + [file_to_run]

    pytest.main(argv)