''' Nose test running

Implements test and bench functions for modules.

'''
import os
import sys
import warnings

# Patches nose functionality to add NumPy-specific features
# Note: This class should only be instantiated if nose has already
# been successfully imported
class NoseCustomizer:
    __patched = False

    def __init__(self):
        if NoseCustomizer.__patched:
            return

        NoseCustomizer.__patched = True

        # used to monkeypatch the nose doctest classes
        def monkeypatch_method(cls):
            def decorator(func):
                setattr(cls, func.__name__, func)
                return func
            return decorator

        from nose.plugins import doctests as npd
        from nose.plugins.base import Plugin
        from nose.util import src, tolist
        import numpy
        import doctest

        # second-chance checker; if the default comparison doesn't 
        # pass, then see if the expected output string contains flags that
        # tell us to ignore the output
        class NumpyDoctestOutputChecker(doctest.OutputChecker):
            def check_output(self, want, got, optionflags):
                ret = doctest.OutputChecker.check_output(self, want, got, 
                                                         optionflags)
                if not ret:
                    if "#random" in want:
                        return True

                return ret


        # Subclass nose.plugins.doctests.DocTestCase to work around a bug in 
        # its constructor that blocks non-default arguments from being passed
        # down into doctest.DocTestCase
        class NumpyDocTestCase(npd.DocTestCase):
            def __init__(self, test, optionflags=0, setUp=None, tearDown=None,
                         checker=None, obj=None, result_var='_'):
                self._result_var = result_var
                self._nose_obj = obj
                doctest.DocTestCase.__init__(self, test, 
                                             optionflags=optionflags,
                                             setUp=setUp, tearDown=tearDown, 
                                             checker=checker)



        # This will replace the existing loadTestsFromModule method of 
        # nose.plugins.doctests.Doctest.  It turns on whitespace normalization,
        # adds an implicit "import numpy as np" for doctests, and adds a
        # "#random" directive to allow executing a command while ignoring its
        # output.
        @monkeypatch_method(npd.Doctest)
        def loadTestsFromModule(self, module):
            if not self.matches(module.__name__):
                npd.log.debug("Doctest doesn't want module %s", module)
                return
            try:
                tests = self.finder.find(module)
            except AttributeError:
                # nose allows module.__test__ = False; doctest does not and 
                # throws AttributeError
                return
            if not tests:
                return
            tests.sort()
            module_file = src(module.__file__)
            for test in tests:
                if not test.examples:
                    continue
                if not test.filename:
                    test.filename = module_file

                # Each doctest should execute in an environment equivalent to
                # starting Python and executing "import numpy as np"
                #
                # Note: __file__ allows the doctest in NoseTester to run
                # without producing an error
                test.globs = {'__builtins__':__builtins__,
                              '__file__':'__main__', 
                              '__name__':'__main__', 
                              'np':numpy}

                # always use whitespace and ellipsis options
                optionflags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS

                yield NumpyDocTestCase(test, 
                                       optionflags=optionflags,
                                       checker=NumpyDoctestOutputChecker())

        # get original print options
        print_state = numpy.get_printoptions()

        # Add an afterContext method to nose.plugins.doctests.Doctest in order
        # to restore print options to the original state after each doctest
        @monkeypatch_method(npd.Doctest)
        def afterContext(self):
            numpy.set_printoptions(**print_state)

        # Replace the existing wantFile method of nose.plugins.doctests.Doctest
        # so that we can ignore NumPy-specific build files that shouldn't
        # be searched for tests
        old_wantFile = npd.Doctest.wantFile
        ignore_files = ['generate_numpy_api.py', 'scons_support.py',
                        'setupscons.py', 'setup.py']
        def wantFile(self, file):
            bn = os.path.basename(file)
            if bn in ignore_files:
                return False
            return old_wantFile(self, file)

        npd.Doctest.wantFile = wantFile


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
        raise ImportError('Need nose >=%d.%d.%d for tests - see '
            'http://somethingaboutorange.com/mrl/projects/nose' % 
            minimum_nose_version)

    # nose was successfully imported; make customizations for doctests
    NoseCustomizer()

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
        argv += ['--exclude','array_from_pyobj']

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


########################################################################
# Doctests for NumPy-specific doctest modifications

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

