''' Nose test running

Implements test and bench functions for modules.

'''
import os
import sys
import warnings

def get_package_name(filepath):
    # find the package name given a path name that's part of the package
    fullpath = filepath[:]
    pkg_name = []
    while 'site-packages' in filepath:
        filepath, p2 = os.path.split(filepath)
        if p2 == 'site-packages':
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
    """ Nose test runner.

    Usage: NoseTester(<package>).test()

    <package> is package path or module Default for package is None. A
    value of None finds the calling module path.

    This class is made available as numpy.testing.Tester, and a test function
    is typically added to a package's __init__.py like so:

    >>> from numpy.testing import Tester
    >>> test = Tester().test

    Calling this test function finds and runs all tests associated with the
    package and all its subpackages.

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
        if package is None:
            f = sys._getframe(1)
            package = f.f_locals.get('__file__', None)
            assert package is not None
            package = os.path.dirname(package)
        elif isinstance(package, type(os)):
            package = os.path.dirname(package.__file__)
        self.package_path = package

        # find the package name under test; this name is used to limit coverage
        # reporting (if enabled)
        self.package_name = get_package_name(package)

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


    def test(self, label='fast', verbose=1, extra_argv=None, doctests=False,
             coverage=False, **kwargs):
        ''' Run tests for module using nose

        %(test_header)s
        doctests : boolean
            If True, run doctests in module, default False
        coverage : boolean
            If True, report coverage of NumPy code, default False
            (Requires the coverage module:
             http://nedbatchelder.com/code/modules/coverage.html)
        '''

        old_args = set(['level', 'verbosity', 'all', 'sys_argv',
                        'testcase_pattern'])
        unexpected_args = set(kwargs.keys()) - old_args
        if len(unexpected_args) > 0:
            ua = ', '.join(unexpected_args)
            raise TypeError("test() got unexpected arguments: %s" % ua)

        # issue a deprecation warning if any of the pre-1.2 arguments to
        # test are given
        if old_args.intersection(kwargs.keys()):
            warnings.warn("This method's signature will change in the next " \
                          "release; the level, verbosity, all, sys_argv, " \
                          "and testcase_pattern keyword arguments will be " \
                          "removed. Please update your code.",
                          DeprecationWarning, stacklevel=2)

        # Use old arguments if given (where it makes sense)
        # For the moment, level and sys_argv are ignored

        # replace verbose with verbosity
        if kwargs.get('verbosity') is not None:
            verbose = kwargs.get('verbosity')
            # cap verbosity at 3 because nose becomes *very* verbose beyond that
            verbose = min(verbose, 3)

        import utils
        utils.verbose = verbose

        # if all evaluates as True, omit attribute filter and run doctests
        if kwargs.get('all'):
            label = ''
            doctests = True

        # if doctests is in the extra args, remove it and set the doctest
        # flag so the NumPy doctester is used instead
        if extra_argv and '--with-doctest' in extra_argv:
            extra_argv.remove('--with-doctest')
            doctests = True

        argv = self._test_argv(label, verbose, extra_argv)
        if doctests:
            argv += ['--with-numpydoctest']
            print "Running unit tests and doctests for %s" % self.package_name
        else:
            print "Running unit tests for %s" % self.package_name

        if coverage:
            argv+=['--cover-package=%s' % self.package_name, '--with-coverage',
                   '--cover-tests', '--cover-inclusive', '--cover-erase']

        # bypass these samples under distutils
        argv += ['--exclude','f2py_ext']
        argv += ['--exclude','f2py_f90_ext']
        argv += ['--exclude','gen_ext']
        argv += ['--exclude','pyrex_ext']
        argv += ['--exclude','swig_ext']
        argv += ['--exclude','array_from_pyobj']

        self._show_system_info()

        nose = import_nose()

        # Because nose currently discards the test result object, but we need
        # to return it to the user, override TestProgram.runTests to retain
        # the result
        class NumpyTestProgram(nose.core.TestProgram):
            def runTests(self):
                """Run Tests. Returns true on success, false on failure, and
                sets self.success to the same value.
                """
                if self.testRunner is None:
                    self.testRunner = nose.core.TextTestRunner(stream=self.config.stream,
                                                               verbosity=self.config.verbosity,
                                                               config=self.config)
                plug_runner = self.config.plugins.prepareTestRunner(self.testRunner)
                if plug_runner is not None:
                    self.testRunner = plug_runner
                self.result = self.testRunner.run(self.test)
                self.success = self.result.wasSuccessful()
                return self.success

        # reset doctest state on every run
        import doctest
        doctest.master = None

        # construct list of plugins, omitting the existing doctest plugin
        import nose.plugins.builtin
        from noseclasses import numpyDoctest
        plugins = [numpyDoctest()]
        for p in nose.plugins.builtin.plugins:
            plug = p()
            if plug.name == 'doctest':
                continue

            plugins.append(plug)

        t = NumpyTestProgram(argv=argv, exit=False, plugins=plugins)
        return t.result

    def bench(self, label='fast', verbose=1, extra_argv=None):
        ''' Run benchmarks for module using nose

        %(test_header)s'''

        print "Running benchmarks for %s" % self.package_name
        self._show_system_info()

        nose = import_nose()
        argv = self._test_argv(label, verbose, extra_argv)
        argv += ['--match', r'(?:^|[\\b_\\.%s-])[Bb]ench' % os.sep]
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
