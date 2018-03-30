"""
Pytest test running.

This module implements the ``test()`` function for NumPy modules. The usual
boiler plate for doing that is to put the following in the module
``__init__.py`` file::

    from numpy.testing import PytestTester
    test = PytestTester(__name__).test
    del PytestTester


Warnings filtering and other runtime settings should be dealt with in the
``pytest.ini`` file in the numpy repo root. The behavior of the test depends on
whether or not that file is found as follows:

* ``pytest.ini`` is present (develop mode)
    All warnings except those explicily filtered out are raised as error.
* ``pytest.ini`` is absent (release mode)
    DeprecationWarnings and PendingDeprecationWarnings are ignored, other
    warnings are passed through.

In practice, tests run from the numpy repo are run in develop mode. That
includes the standard ``python runtests.py`` invocation.

"""
from __future__ import division, absolute_import, print_function

import sys
import os

__all__ = ['PytestTester']


def _show_numpy_info():
    import numpy as np

    print("NumPy version %s" % np.__version__)
    relaxed_strides = np.ones((10, 1), order="C").flags.f_contiguous
    print("NumPy relaxed strides checking option:", relaxed_strides)


class PytestTester(object):
    """
    Pytest test runner.

    This class is made available in ``numpy.testing``, and a test function
    is typically added to a package's __init__.py like so::

      from numpy.testing import PytestTester
      test = PytestTester(__name__).test
      del PytestTester

    Calling this test function finds and runs all tests associated with the
    module and all its sub-modules.

    Attributes
    ----------
    module_name : str
        Full path to the package to test.

    Parameters
    ----------
    module_name : module name
        The name of the module to test.

    """
    def __init__(self, module_name):
        self.module_name = module_name

    def test(self, label='fast', verbose=1, extra_argv=None,
             doctests=False, coverage=False, timer=0, tests=None):
        """
        Run tests for module using pytest.

        Parameters
        ----------
        label : {'fast', 'full'}, optional
            Identifies the tests to run. When set to 'fast', tests decorated
            with `pytest.mark.slow` are skipped, when 'full', the slow marker
            is ignored.
        verbose : int, optional
            Verbosity value for test outputs, in the range 1-3. Default is 1.
        extra_argv : list, optional
            List with any extra arguments to pass to pytests.
        doctests : bool, optional
            .. note:: Not supported
        coverage : bool, optional
            If True, report coverage of NumPy code. Default is False.
            Requires installation of (pip) pytest-cov.
        timer : int, optional
            If > 0, report the time of the slowest `timer` tests. Default is 0.

        tests : test or list of tests
            Tests to be executed with pytest '--pyargs'

        Returns
        -------
        result : bool
            Return True on success, false otherwise.

        Notes
        -----
        Each NumPy module exposes `test` in its namespace to run all tests for it.
        For example, to run all tests for numpy.lib:

        >>> np.lib.test() #doctest: +SKIP

        Examples
        --------
        >>> result = np.lib.test() #doctest: +SKIP
        Running unit tests for numpy.lib
        ...
        Ran 976 tests in 3.933s

        OK

        >>> result.errors #doctest: +SKIP
        []
        >>> result.knownfail #doctest: +SKIP
        []

        """
        import pytest

        #FIXME This is no longer needed? Assume it was for use in tests.
        # cap verbosity at 3, which is equivalent to the pytest '-vv' option
        #from . import utils
        #verbose = min(int(verbose), 3)
        #utils.verbose = verbose
        #

        module = sys.modules[self.module_name]
        module_path = os.path.abspath(module.__path__[0])

        # setup the pytest arguments
        pytest_args = ['-l']

        if doctests:
            raise ValueError("Doctests not supported")

        if extra_argv:
            pytest_args += list(extra_argv)

        if verbose > 1:
            pytest_args += ["-" + "v"*(verbose - 1)]
        else:
            pytest_args += ["-q"]

        if coverage:
            pytest_args += ["--cov=" + module_path]

        if label == "fast":
            pytest_args += ["-m", "not slow"]
        elif label != "full":
            pytest_args += ["-m", label]

        if timer > 0:
            pytest_args += ["--durations=%s" % timer]

        if tests is None:
            tests = [self.module_name]

        pytest_args += ['--pyargs'] + list(tests)


        # run tests.
        _show_numpy_info()

        try:
            code = pytest.main(pytest_args)
        except SystemExit as exc:
            code = exc.code

        return code == 0
