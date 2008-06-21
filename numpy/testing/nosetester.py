''' Nose test running

Implements test and bench functions for modules.

'''
import os
import sys
import re
import warnings

def import_nose():
    """ Import nose only when needed.
    """
    fine_nose = True
    try:
        import nose
        from nose.tools import raises
    except ImportError:
        fine_nose = False
    else:
        nose_version = nose.__versioninfo__
        if nose_version[0] < 1 and nose_version[1] < 10:
            fine_nose = False

    if not fine_nose:
        raise ImportError('Need nose >=0.10 for tests - see '
            'http://somethingaboutorange.com/mrl/projects/nose')

    return nose

def run_module_suite(file_to_run = None):
    if file_to_run is None:
        f = sys._getframe(1)
        file_to_run = f.f_locals.get('__file__', None)
        assert file_to_run is not None

    import_nose().run(argv=['',file_to_run])


class NoseTester(object):
    """ Nose test runner.

    Usage: NoseTester(<package>).test()

    <package> is package path or module Default for package is None. A
    value of None finds calling module path.

    Typical call is from module __init__, and corresponds to this:

    >>> test = NoseTester().test

    This class is made available as numpy.testing.Tester:

    >>> from scipy.testing import Tester
    >>> test = Tester().test
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
        pkg_temp = package
        pkg_name = []
        while 'site-packages' in pkg_temp:
            pkg_temp, p2 = os.path.split(pkg_temp)
            if p2 == 'site-packages':
                break
            pkg_name.append(p2)

        # if package name determination failed, just default to numpy/scipy
        if not pkg_name:
            if 'scipy' in self.package_path:
                self.package_name = 'scipy'
            else:
                self.package_name = 'numpy'
        else:
            pkg_name.reverse()
            self.package_name = '.'.join(pkg_name)

    def _add_doc(testtype):
        ''' Decorator to add docstring to functions using test labels

        Parameters
        ----------
        testtype : string
            Type of test for function docstring
        '''
        def docit(func):
            test_header = \
        '''Parameters
        ----------
        label : {'fast', 'full', '', attribute identifer}
            Identifies %(testtype)s to run.  This can be a string to pass to
            the nosetests executable with the'-A' option, or one of
            several special values.
            Special values are:
            'fast' - the default - which corresponds to
                nosetests -A option of
                'not slow'.
            'full' - fast (as above) and slow %(testtype)s as in
                no -A option to nosetests - same as ''
            None or '' - run all %(testtype)ss
            attribute_identifier - string passed directly to
                nosetests as '-A'
        verbose : integer
            verbosity value for test outputs, 1-10
        extra_argv : list
            List with any extra args to pass to nosetests''' \
            % {'testtype': testtype}
            func.__doc__ = func.__doc__ % {
                'test_header': test_header}
            return func
        return docit

    @_add_doc('(testtype)')
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
    
    @_add_doc('test')
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
        old_args = set(['level', 'verbosity', 'all', 'sys_argv', 'testcase_pattern'])
        unexpected_args = set(kwargs.keys()) - old_args
        if len(unexpected_args) > 0:
            ua = ', '.join(unexpected_args)
            raise TypeError("test() got unexpected arguments: %s" % ua)

        # issue a deprecation warning if any of the pre-1.2 arguments to 
        # test are given
        if old_args.intersection(kwargs.keys()):
            warnings.warn("This method's signature will change in the next release; the level, verbosity, all, sys_argv, and testcase_pattern keyword arguments will be removed. Please update your code.", 
                          DeprecationWarning, stacklevel=2)
        
        # Use old arguments if given (where it makes sense)
        # For the moment, level and sys_argv are ignored

        # replace verbose with verbosity
        if kwargs.get('verbosity') is not None:
            verbose = kwargs.get('verbosity')
            # cap verbosity at 3 because nose becomes *very* verbose beyond that
            verbose = min(verbose, 3)

        # if all evaluates as True, omit attribute filter and run doctests
        if kwargs.get('all'):
            label = ''
            doctests = True

        argv = self._test_argv(label, verbose, extra_argv)
        if doctests:
            argv+=['--with-doctest','--doctest-tests']

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

        # Because nose currently discards the test result object, but we need to 
        # return it to the user, override TestProgram.runTests to retain the result
        class NumpyTestProgram(nose.core.TestProgram):
            def runTests(self):
                """Run Tests. Returns true on success, false on failure, and sets
                self.success to the same value.
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
            
        t = NumpyTestProgram(argv=argv, exit=False)
        return t.result

    @_add_doc('benchmark')
    def bench(self, label='fast', verbose=1, extra_argv=None):
        ''' Run benchmarks for module using nose

        %(test_header)s'''
        nose = import_nose()
        argv = self._test_argv(label, verbose, extra_argv)
        argv += ['--match', r'(?:^|[\\b_\\.%s-])[Bb]ench' % os.sep]
        return nose.run(argv=argv)
