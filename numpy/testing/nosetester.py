"""
Nose test running.

This module implements ``test()`` and ``bench()`` functions for NumPy modules.

"""
import os
import sys

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

def import_nose():
    """ Import nose only when needed.
    """
    fine_nose = True
    minimum_nose_version = (0,10,0)
    try:
        import nose
        from nose.tools import raises
    except ImportError:
        fine_nose = False
    else:
        if nose.__versioninfo__ < minimum_nose_version:
            fine_nose = False

    if not fine_nose:
        msg = 'Need nose >= %d.%d.%d for tests - see ' \
              'http://somethingaboutorange.com/mrl/projects/nose' % \
              minimum_nose_version

        raise ImportError(msg)

    return nose

def run_module_suite(file_to_run = None):
    if file_to_run is None:
        f = sys._getframe(1)
        file_to_run = f.f_locals.get('__file__', None)
        assert file_to_run is not None

    import_nose().run(argv=['',file_to_run])

# contructs NoseTester method docstrings
def _docmethod(meth, testtype):
    if not meth.__doc__:
        return

    test_header = \
        '''Parameters
        ----------
        label : {'fast', 'full', '', attribute identifer}
            Identifies the %(testtype)ss to run.  This can be a string to
            pass to the nosetests executable with the '-A' option, or one of
            several special values.
            Special values are:
                'fast' - the default - which corresponds to nosetests -A option
                         of 'not slow'.
                'full' - fast (as above) and slow %(testtype)ss as in the
                         no -A option to nosetests - same as ''
            None or '' - run all %(testtype)ss
            attribute_identifier - string passed directly to nosetests as '-A'
        verbose : integer
            verbosity value for test outputs, 1-10
        extra_argv : list
            List with any extra args to pass to nosetests''' \
            % {'testtype': testtype}

    meth.__doc__ = meth.__doc__ % {'test_header':test_header}


class NoseTester(object):
    """
    Nose test runner.

    This class is made available as numpy.testing.Tester, and a test function
    is typically added to a package's __init__.py like so::

      from numpy.testing import Tester
      test = Tester().test

    Calling this test function finds and runs all tests associated with the
    package and all its sub-packages.

    Attributes
    ----------
    package_path : str
        Full path to the package to test.
    package_name : str
        Name of the package to test.

    Parameters
    ----------
    package : module, str or None
        The package to test. If a string, this should be the full path to
        the package. If None (default), `package` is set to the module from
        which `NoseTester` is initialized.

    """

    def __init__(self, package=None):
        ''' Test class init

        Parameters
        ----------
        package : string or module
            If string, gives full path to package
            If None, extract calling module path
            Default is None
        '''
        package_name = None
        if package is None:
            f = sys._getframe(1)
            package_path = f.f_locals.get('__file__', None)
            assert package_path is not None
            package_path = os.path.dirname(package_path)
            package_name = f.f_locals.get('__name__', None)
        elif isinstance(package, type(os)):
            package_path = os.path.dirname(package.__file__)
            package_name = getattr(package, '__name__', None)
        else:
            package_path = str(package)

        self.package_path = package_path

        # find the package name under test; this name is used to limit coverage
        # reporting (if enabled)
        if package_name is None:
            package_name = get_package_name(package_path)
        self.package_name = package_name

    def _test_argv(self, label, verbose, extra_argv):
        ''' Generate argv for nosetest command

        %(test_header)s
        '''
        argv = [__file__, self.package_path, '-s']
        if label and label != 'full':
            if not isinstance(label, basestring):
                raise TypeError, 'Selection label should be a string'
            if label == 'fast':
                label = 'not slow'
            argv += ['-A', label]
        argv += ['--verbosity', str(verbose)]
        if extra_argv:
            argv += extra_argv
        return argv

    def _show_system_info(self):
        nose = import_nose()

        import numpy
        print "NumPy version %s" % numpy.__version__
        npdir = os.path.dirname(numpy.__file__)
        print "NumPy is installed in %s" % npdir

        if 'scipy' in self.package_name:
            import scipy
            print "SciPy version %s" % scipy.__version__
            spdir = os.path.dirname(scipy.__file__)
            print "SciPy is installed in %s" % spdir

        pyversion = sys.version.replace('\n','')
        print "Python version %s" % pyversion
        print "nose version %d.%d.%d" % nose.__versioninfo__


    def prepare_test_args(self, label='fast', verbose=1, extra_argv=None, 
                          doctests=False, coverage=False):
        """
        Run tests for module using nose.

        This method does the heavy lifting for the `test` method. It takes all
        the same arguments, for details see `test`.

        See Also
        --------
        test

        """

        # if doctests is in the extra args, remove it and set the doctest
        # flag so the NumPy doctester is used instead
        if extra_argv and '--with-doctest' in extra_argv:
            extra_argv.remove('--with-doctest')
            doctests = True

        argv = self._test_argv(label, verbose, extra_argv)
        if doctests:
            argv += ['--with-numpydoctest']

        if coverage:
            argv+=['--cover-package=%s' % self.package_name, '--with-coverage',
                   '--cover-tests', '--cover-inclusive', '--cover-erase']

        # bypass these samples under distutils
        argv += ['--exclude','f2py_ext']
        argv += ['--exclude','f2py_f90_ext']
        argv += ['--exclude','gen_ext']
        argv += ['--exclude','pyrex_ext']
        argv += ['--exclude','swig_ext']

        nose = import_nose()

        # construct list of plugins
        import nose.plugins.builtin
        from noseclasses import NumpyDoctest, KnownFailure
        plugins = [NumpyDoctest(), KnownFailure()]
        plugins += [p() for p in nose.plugins.builtin.plugins]
        return argv, plugins

    def test(self, label='fast', verbose=1, extra_argv=None, doctests=False,
             coverage=False):
        """
        Run tests for module using nose.

        Parameters
        ----------
        label : {'fast', 'full', '', attribute identifier}, optional
            Identifies the tests to run. This can be a string to pass to the
            nosetests executable with the '-A' option, or one of
            several special values.
            Special values are:
                'fast' - the default - which corresponds to the ``nosetests -A``
                         option of 'not slow'.
                'full' - fast (as above) and slow tests as in the
                         'no -A' option to nosetests - this is the same as ''.
            None or '' - run all tests.
            attribute_identifier - string passed directly to nosetests as '-A'.
        verbose : int, optional
            Verbosity value for test outputs, in the range 1-10. Default is 1.
        extra_argv : list, optional
            List with any extra arguments to pass to nosetests.
        doctests : bool, optional
            If True, run doctests in module. Default is False.
        coverage : bool, optional
            If True, report coverage of NumPy code. Default is False.
            (This requires the `coverage module:
             <http://nedbatchelder.com/code/modules/coverage.html>`_).

        Returns
        -------
        result : object
            Returns the result of running the tests as a
            ``nose.result.TextTestResult`` object.

        Notes
        -----
        Each NumPy module exposes `test` in its namespace to run all tests for it.
        For example, to run all tests for numpy.lib::

          >>> np.lib.test()

        Examples
        --------
        >>> result = np.lib.test()
        Running unit tests for numpy.lib
        ...
        Ran 976 tests in 3.933s

        OK

        >>> result.errors
        []
        >>> result.knownfail
        []

        """

        # cap verbosity at 3 because nose becomes *very* verbose beyond that
        verbose = min(verbose, 3)

        import utils
        utils.verbose = verbose

        if doctests:
            print "Running unit tests and doctests for %s" % self.package_name
        else:
            print "Running unit tests for %s" % self.package_name

        self._show_system_info()

        # reset doctest state on every run
        import doctest
        doctest.master = None

        argv, plugins = self.prepare_test_args(label, verbose, extra_argv,
                                               doctests, coverage)
        from noseclasses import NumpyTestProgram
        t = NumpyTestProgram(argv=argv, exit=False, plugins=plugins)
        return t.result

    def bench(self, label='fast', verbose=1, extra_argv=None):
        """
        Run benchmarks for module using nose.

        Parameters
        ----------
        label : {'fast', 'full', '', attribute identifier}, optional
            Identifies the tests to run. This can be a string to pass to the
            nosetests executable with the '-A' option, or one of
            several special values.
            Special values are:
                'fast' - the default - which corresponds to the ``nosetests -A``
                         option of 'not slow'.
                'full' - fast (as above) and slow tests as in the
                         'no -A' option to nosetests - this is the same as ''.
            None or '' - run all tests.
            attribute_identifier - string passed directly to nosetests as '-A'.
        verbose : int, optional
            Verbosity value for test outputs, in the range 1-10. Default is 1.
        extra_argv : list, optional
            List with any extra arguments to pass to nosetests.

        Returns
        -------
        success : bool
            Returns True if running the benchmarks works, False if an error
            occurred.

        Notes
        -----
        Benchmarks are like tests, but have names starting with "bench" instead
        of "test", and can be found under the "benchmarks" sub-directory of the
        module.

        Each NumPy module exposes `bench` in its namespace to run all benchmarks
        for it.

        Examples
        --------
        >>> success = np.lib.bench()
        Running benchmarks for numpy.lib
        ...
        using 562341 items:
        unique:
        0.11
        unique1d:
        0.11
        ratio: 1.0
        nUnique: 56230 == 56230
        ...
        OK

        >>> success
        True

        """

        print "Running benchmarks for %s" % self.package_name
        self._show_system_info()

        argv = self._test_argv(label, verbose, extra_argv)
        argv += ['--match', r'(?:^|[\\b_\\.%s-])[Bb]ench' % os.sep]

        nose = import_nose()
        return nose.run(argv=argv)

    # generate method docstrings
    _docmethod(_test_argv, '(testtype)')
    _docmethod(test, 'test')
    _docmethod(bench, 'benchmark')


########################################################################
# Doctests for NumPy-specific nose/doctest modifications

# try the #random directive on the output line
def check_random_directive():
    '''
    >>> 2+2
    <BadExample object at 0x084D05AC>  #random: may vary on your system
    '''

# check the implicit "import numpy as np"
def check_implicit_np():
    '''
    >>> np.array([1,2,3])
    array([1, 2, 3])
    '''

# there's some extraneous whitespace around the correct responses
def check_whitespace_enabled():
    '''
    # whitespace after the 3
    >>> 1+2
    3

    # whitespace before the 7
    >>> 3+4
     7
    '''
