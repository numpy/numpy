import os
import re
import sys
import imp
import glob
import types
import shlex
import unittest
import traceback
import warnings

__all__ = ['set_package_path', 'set_local_path', 'restore_path',
           'IgnoreException', 'NumpyTestCase', 'NumpyTest',
           'ScipyTestCase', 'ScipyTest', # for backward compatibility
           'importall',
           ]

DEBUG=0
from numpy.testing.utils import jiffies
get_frame = sys._getframe

class IgnoreException(Exception):
    "Ignoring this exception due to disabled feature"


def set_package_path(level=1):
    """ Prepend package directory to sys.path.

    set_package_path should be called from a test_file.py that
    satisfies the following tree structure:

      <somepath>/<somedir>/test_file.py

    Then the first existing path name from the following list

      <somepath>/build/lib.<platform>-<version>
      <somepath>/..

    is prepended to sys.path.
    The caller is responsible for removing this path by using

      restore_path()
    """
    from distutils.util import get_platform
    f = get_frame(level)
    if f.f_locals['__name__']=='__main__':
        testfile = sys.argv[0]
    else:
        testfile = f.f_locals['__file__']
    d = os.path.dirname(os.path.dirname(os.path.abspath(testfile)))
    d1 = os.path.join(d,'build','lib.%s-%s'%(get_platform(),sys.version[:3]))
    if not os.path.isdir(d1):
        d1 = os.path.dirname(d)
    if DEBUG:
        print 'Inserting %r to sys.path for test_file %r' % (d1, testfile)
    sys.path.insert(0,d1)
    return


def set_local_path(reldir='', level=1):
    """ Prepend local directory to sys.path.

    The caller is responsible for removing this path by using

      restore_path()
    """
    f = get_frame(level)
    if f.f_locals['__name__']=='__main__':
        testfile = sys.argv[0]
    else:
        testfile = f.f_locals['__file__']
    local_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(testfile)),reldir))
    if DEBUG:
        print 'Inserting %r to sys.path' % (local_path)
    sys.path.insert(0,local_path)
    return


def restore_path():
    if DEBUG:
        print 'Removing %r from sys.path' % (sys.path[0])
    del sys.path[0]
    return


def output_exception(printstream = sys.stdout):
    try:
        type, value, tb = sys.exc_info()
        info = traceback.extract_tb(tb)
        #this is more verbose
        #traceback.print_exc()
        filename, lineno, function, text = info[-1] # last line only
        print>>printstream, "%s:%d: %s: %s (in %s)" %\
                            (filename, lineno, type.__name__, str(value), function)
    finally:
        type = value = tb = None # clean up
    return


class _dummy_stream:
    def __init__(self,stream):
        self.data = []
        self.stream = stream
    def write(self,message):
        if not self.data and not message.startswith('E'):
            self.stream.write(message)
            self.stream.flush()
            message = ''
        self.data.append(message)
    def writeln(self,message):
        self.write(message+'\n')
    def flush(self):
        self.stream.flush()


class NumpyTestCase (unittest.TestCase):

    def measure(self,code_str,times=1):
        """ Return elapsed time for executing code_str in the
        namespace of the caller for given times.
        """
        frame = get_frame(1)
        locs,globs = frame.f_locals,frame.f_globals
        code = compile(code_str,
                       'NumpyTestCase runner for '+self.__class__.__name__,
                       'exec')
        i = 0
        elapsed = jiffies()
        while i<times:
            i += 1
            exec code in globs,locs
        elapsed = jiffies() - elapsed
        return 0.01*elapsed

    def __call__(self, result=None):
        if result is None or not hasattr(result, 'errors') \
                or not hasattr(result, 'stream'):
            return unittest.TestCase.__call__(self, result)

        nof_errors = len(result.errors)
        save_stream = result.stream
        result.stream = _dummy_stream(save_stream)
        unittest.TestCase.__call__(self, result)
        if nof_errors != len(result.errors):
            test, errstr = result.errors[-1][:2]
            if isinstance(errstr, tuple):
                errstr = str(errstr[0])
            elif isinstance(errstr, str):
                errstr = errstr.split('\n')[-2]
            else:
                # allow for proxy classes
                errstr = str(errstr).split('\n')[-2]
            l = len(result.stream.data)
            if errstr.startswith('IgnoreException:'):
                if l==1:
                    assert result.stream.data[-1]=='E', \
                            repr(result.stream.data)
                    result.stream.data[-1] = 'i'
                else:
                    assert result.stream.data[-1]=='ERROR\n', \
                            repr(result.stream.data)
                    result.stream.data[-1] = 'ignoring\n'
                del result.errors[-1]
        map(save_stream.write, result.stream.data)
        save_stream.flush()
        result.stream = save_stream

    def warn(self, message):
        from numpy.distutils.misc_util import yellow_text
        print>>sys.stderr,yellow_text('Warning: %s' % (message))
        sys.stderr.flush()
    def info(self, message):
        print>>sys.stdout, message
        sys.stdout.flush()

    def rundocs(self, filename=None):
        """ Run doc string tests found in filename.
        """
        import doctest
        if filename is None:
            f = get_frame(1)
            filename = f.f_globals['__file__']
        name = os.path.splitext(os.path.basename(filename))[0]
        path = [os.path.dirname(filename)]
        file, pathname, description = imp.find_module(name, path)
        try:
            m = imp.load_module(name, file, pathname, description)
        finally:
            file.close()
        if sys.version[:3]<'2.4':
            doctest.testmod(m, verbose=False)
        else:
            tests = doctest.DocTestFinder().find(m)
            runner = doctest.DocTestRunner(verbose=False)
            for test in tests:
                runner.run(test)
        return

class ScipyTestCase(NumpyTestCase):
    def __init__(self, package=None):
        warnings.warn("ScipyTestCase is now called NumpyTestCase; please update your code",
                         DeprecationWarning, stacklevel=2)
        NumpyTestCase.__init__(self, package)


def _get_all_method_names(cls):
    names = dir(cls)
    if sys.version[:3]<='2.1':
        for b in cls.__bases__:
            for n in dir(b)+_get_all_method_names(b):
                if n not in names:
                    names.append(n)
    return names


# for debug build--check for memory leaks during the test.
class _NumPyTextTestResult(unittest._TextTestResult):
    def startTest(self, test):
        unittest._TextTestResult.startTest(self, test)
        if self.showAll:
            N = len(sys.getobjects(0))
            self._totnumobj = N
            self._totrefcnt = sys.gettotalrefcount()
        return

    def stopTest(self, test):
        if self.showAll:
            N = len(sys.getobjects(0))
            self.stream.write("objects: %d ===> %d;   " % (self._totnumobj, N))
            self.stream.write("refcnts: %d ===> %d\n" % (self._totrefcnt,
                              sys.gettotalrefcount()))
        return

class NumPyTextTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return _NumPyTextTestResult(self.stream, self.descriptions, self.verbosity)


class NumpyTest:
    """ Numpy tests site manager.

    Usage: NumpyTest(<package>).test(level=1,verbosity=1)

    <package> is package name or its module object.

    Package is supposed to contain a directory tests/ with test_*.py
    files where * refers to the names of submodules.  See .rename()
    method to redefine name mapping between test_*.py files and names of
    submodules. Pattern test_*.py can be overwritten by redefining
    .get_testfile() method.

    test_*.py files are supposed to define a classes, derived from
    NumpyTestCase or unittest.TestCase, with methods having names
    starting with test or bench or check. The names of TestCase classes
    must have a prefix test. This can be overwritten by redefining
    .check_testcase_name() method.

    And that is it! No need to implement test or test_suite functions
    in each .py file.

    Old-style test_suite(level=1) hooks are also supported.
    """
    _check_testcase_name = re.compile(r'test.*|Test.*').match
    def check_testcase_name(self, name):
        """ Return True if name matches TestCase class.
        """
        return not not self._check_testcase_name(name)

    testfile_patterns = ['test_%(modulename)s.py']
    def get_testfile(self, module, verbosity = 0):
        """ Return path to module test file.
        """
        mstr = self._module_str
        short_module_name = self._get_short_module_name(module)
        d = os.path.split(module.__file__)[0]
        test_dir = os.path.join(d,'tests')
        local_test_dir = os.path.join(os.getcwd(),'tests')
        if os.path.basename(os.path.dirname(local_test_dir)) \
               == os.path.basename(os.path.dirname(test_dir)):
            test_dir = local_test_dir
        for pat in self.testfile_patterns:
            fn = os.path.join(test_dir, pat % {'modulename':short_module_name})
            if os.path.isfile(fn):
                return fn
        if verbosity>1:
            self.warn('No test file found in %s for module %s' \
                      % (test_dir, mstr(module)))
        return

    def __init__(self, package=None):
        if package is None:
            from numpy.distutils.misc_util import get_frame
            f = get_frame(1)
            package = f.f_locals.get('__name__',f.f_globals.get('__name__',None))
            assert package is not None
        self.package = package
        self._rename_map = {}

    def rename(self, **kws):
        """Apply renaming submodule test file test_<name>.py to
        test_<newname>.py.

        Usage: self.rename(name='newname') before calling the
        self.test() method.

        If 'newname' is None, then no tests will be executed for a given
        module.
        """
        for k,v in kws.items():
            self._rename_map[k] = v
        return

    def _module_str(self, module):
        filename = module.__file__[-30:]
        if filename!=module.__file__:
            filename = '...'+filename
        return '<module %r from %r>' % (module.__name__, filename)

    def _get_method_names(self,clsobj,level):
        names = []
        for mthname in _get_all_method_names(clsobj):
            if mthname[:5] not in ['bench','check'] \
               and mthname[:4] not in ['test']:
                continue
            mth = getattr(clsobj, mthname)
            if type(mth) is not types.MethodType:
                continue
            d = mth.im_func.func_defaults
            if d is not None:
                mthlevel = d[0]
            else:
                mthlevel = 1
            if level>=mthlevel:
                if mthname not in names:
                    names.append(mthname)
            for base in clsobj.__bases__:
                for n in self._get_method_names(base,level):
                    if n not in names:
                        names.append(n)
        return names

    def _get_short_module_name(self, module):
        d,f = os.path.split(module.__file__)
        short_module_name = os.path.splitext(os.path.basename(f))[0]
        if short_module_name=='__init__':
            short_module_name = module.__name__.split('.')[-1]
        short_module_name = self._rename_map.get(short_module_name,short_module_name)
        return short_module_name

    def _get_module_tests(self, module, level, verbosity):
        mstr = self._module_str

        short_module_name = self._get_short_module_name(module)
        if short_module_name is None:
            return []

        test_file = self.get_testfile(module, verbosity)

        if test_file is None:
            return []

        if not os.path.isfile(test_file):
            if short_module_name[:5]=='info_' \
               and short_module_name[5:]==module.__name__.split('.')[-2]:
                return []
            if short_module_name in ['__cvs_version__','__svn_version__']:
                return []
            if short_module_name[-8:]=='_version' \
               and short_module_name[:-8]==module.__name__.split('.')[-2]:
                return []
            if verbosity>1:
                self.warn(test_file)
                self.warn('   !! No test file %r found for %s' \
                          % (os.path.basename(test_file), mstr(module)))
            return []

        if test_file in self.test_files:
            return []

        parent_module_name = '.'.join(module.__name__.split('.')[:-1])
        test_module_name,ext = os.path.splitext(os.path.basename(test_file))
        test_dir_module = parent_module_name+'.tests'
        test_module_name = test_dir_module+'.'+test_module_name

        if test_dir_module not in sys.modules:
            sys.modules[test_dir_module] = imp.new_module(test_dir_module)

        old_sys_path = sys.path[:]
        try:
            f = open(test_file,'r')
            test_module = imp.load_module(test_module_name, f,
                                          test_file, ('.py', 'r', 1))
            f.close()
        except:
            sys.path[:] = old_sys_path
            self.warn('FAILURE importing tests for %s' % (mstr(module)))
            output_exception(sys.stderr)
            return []
        sys.path[:] = old_sys_path

        self.test_files.append(test_file)

        return self._get_suite_list(test_module, level, module.__name__)

    def _get_suite_list(self, test_module, level, module_name='__main__',
                        verbosity=1):
        suite_list = []
        if hasattr(test_module, 'test_suite'):
            suite_list.extend(test_module.test_suite(level)._tests)
        for name in dir(test_module):
            obj = getattr(test_module, name)
            if type(obj) is not type(unittest.TestCase) \
               or not issubclass(obj, unittest.TestCase) \
               or not self.check_testcase_name(obj.__name__):
                continue
            for mthname in self._get_method_names(obj,level):
                suite = obj(mthname)
                if getattr(suite,'isrunnable',lambda mthname:1)(mthname):
                    suite_list.append(suite)
        matched_suite_list = [suite for suite in suite_list \
                              if self.testcase_match(suite.id()\
                                                     .replace('__main__.',''))]
        if verbosity>=0:
            self.info('  Found %s/%s tests for %s' \
                      % (len(matched_suite_list), len(suite_list), module_name))
        return matched_suite_list

    def _test_suite_from_modules(self, this_package, level, verbosity):
        package_name = this_package.__name__
        modules = []
        for name, module in sys.modules.items():
            if not name.startswith(package_name) or module is None:
                continue
            if not hasattr(module,'__file__'):
                continue
            if os.path.basename(os.path.dirname(module.__file__))=='tests':
                continue
            modules.append((name, module))

        modules.sort()
        modules = [m[1] for m in modules]

        self.test_files = []
        suites = []
        for module in modules:
            suites.extend(self._get_module_tests(module, abs(level), verbosity))

        suites.extend(self._get_suite_list(sys.modules[package_name],
                                           abs(level), verbosity=verbosity))
        return unittest.TestSuite(suites)

    def _test_suite_from_all_tests(self, this_package, level, verbosity):
        importall(this_package)
        package_name = this_package.__name__

        # Find all tests/ directories under the package
        test_dirs_names = {}
        for name, module in sys.modules.items():
            if not name.startswith(package_name) or module is None:
                continue
            if not hasattr(module, '__file__'):
                continue
            d = os.path.dirname(module.__file__)
            if os.path.basename(d)=='tests':
                continue
            d = os.path.join(d, 'tests')
            if not os.path.isdir(d):
                continue
            if d in test_dirs_names:
                continue
            test_dir_module = '.'.join(name.split('.')[:-1]+['tests'])
            test_dirs_names[d] = test_dir_module

        test_dirs = test_dirs_names.keys()
        test_dirs.sort()

        # For each file in each tests/ directory with a test case in it,
        # import the file, and add the test cases to our list
        suite_list = []
        testcase_match = re.compile(r'\s*class\s+\w+\s*\(.*TestCase').match
        for test_dir in test_dirs:
            test_dir_module = test_dirs_names[test_dir]

            if test_dir_module not in sys.modules:
                sys.modules[test_dir_module] = imp.new_module(test_dir_module)

            for fn in os.listdir(test_dir):
                base, ext = os.path.splitext(fn)
                if ext != '.py':
                    continue
                f = os.path.join(test_dir, fn)

                # check that file contains TestCase class definitions:
                fid = open(f, 'r')
                skip = True
                for line in fid:
                    if testcase_match(line):
                        skip = False
                        break
                fid.close()
                if skip:
                    continue

                # import the test file
                n = test_dir_module + '.' + base
                # in case test files import local modules
                sys.path.insert(0, test_dir)
                fo = None
                try:
                    try:
                        fo = open(f)
                        test_module = imp.load_module(n, fo, f,
                                                      ('.py', 'U', 1))
                    except Exception, msg:
                        print 'Failed importing %s: %s' % (f,msg)
                        continue
                finally:
                    if fo:
                        fo.close()
                    del sys.path[0]

                suites = self._get_suite_list(test_module, level,
                                              module_name=n,
                                              verbosity=verbosity)
                suite_list.extend(suites)

        all_tests = unittest.TestSuite(suite_list)
        return all_tests

    def test(self, level=1, verbosity=1, verbose=0, all=True, sys_argv=[],
             testcase_pattern='.*'):
        """Run Numpy module test suite with level and verbosity.

        level:
          None           --- do nothing, return None
          < 0            --- scan for tests of level=abs(level),
                             don't run them, return TestSuite-list
          > 0            --- scan for tests of level, run them,
                             return TestRunner
          > 10           --- run all tests (same as specifying all=True).
                             (backward compatibility).

        verbosity:
          >= 0           --- show information messages
          > 1            --- show warnings on missing tests

        all:
          True            --- run all test files (like self.testall())
          False (default) --- only run test files associated with a module

        sys_argv          --- replacement of sys.argv[1:] during running
                              tests.

        testcase_pattern  --- run only tests that match given pattern.

        It is assumed (when all=False) that package tests suite follows
        the following convention: for each package module, there exists
        file <packagepath>/tests/test_<modulename>.py that defines
        TestCase classes (with names having prefix 'test_') with methods
        (with names having prefixes 'check_' or 'bench_'); each of these
        methods are called when running unit tests.
        """
        # add verbose keyword and make it an alias for verbosity
        # so that buildbots using newer test framework work.
        verbosity = max(verbosity, verbose)

        if level is None: # Do nothing.
            return

        if isinstance(self.package, str):
            exec 'import %s as this_package' % (self.package)
        else:
            this_package = self.package

        self.testcase_match = re.compile(testcase_pattern).match

        if all:
            all_tests = self._test_suite_from_all_tests(this_package,
                                                        level, verbosity)
        else:
            all_tests = self._test_suite_from_modules(this_package,
                                                      level, verbosity)

        if level < 0:
            return all_tests

        runner = unittest.TextTestRunner(verbosity=verbosity)
        old_sys_argv = sys.argv[1:]
        sys.argv[1:] = sys_argv
        # Use the builtin displayhook. If the tests are being run
        # under IPython (for instance), any doctest test suites will
        # fail otherwise.
        old_displayhook = sys.displayhook
        sys.displayhook = sys.__displayhook__
        try:
            r = runner.run(all_tests)
        finally:
            sys.displayhook = old_displayhook
        sys.argv[1:] = old_sys_argv
        return r

    def testall(self, level=1,verbosity=1):
        """ Run Numpy module test suite with level and verbosity.

        level:
          None           --- do nothing, return None
          < 0            --- scan for tests of level=abs(level),
                             don't run them, return TestSuite-list
          > 0            --- scan for tests of level, run them,
                             return TestRunner

        verbosity:
          >= 0           --- show information messages
          > 1            --- show warnings on missing tests

        Different from .test(..) method, this method looks for
        TestCase classes from all files in <packagedir>/tests/
        directory and no assumptions are made for naming the
        TestCase classes or their methods.
        """
        return self.test(level=level, verbosity=verbosity, all=True)

    def run(self):
        """ Run Numpy module test suite with level and verbosity
        taken from sys.argv. Requires optparse module.
        """
        try:
            from optparse import OptionParser
        except ImportError:
            self.warn('Failed to import optparse module, ignoring.')
            return self.test()
        usage = r'usage: %prog [-v <verbosity>] [-l <level>]'\
                r' [-s "<replacement of sys.argv[1:]>"]'\
                r' [-t "<testcase pattern>"]'
        parser = OptionParser(usage)
        parser.add_option("-v", "--verbosity",
                          action="store",
                          dest="verbosity",
                          default=1,
                          type='int')
        parser.add_option("-l", "--level",
                          action="store",
                          dest="level",
                          default=1,
                          type='int')
        parser.add_option("-s", "--sys-argv",
                          action="store",
                          dest="sys_argv",
                          default='',
                          type='string')
        parser.add_option("-t", "--testcase-pattern",
                          action="store",
                          dest="testcase_pattern",
                          default=r'.*',
                          type='string')
        (options, args) = parser.parse_args()
        return self.test(options.level,options.verbosity,
                         sys_argv=shlex.split(options.sys_argv or ''),
                         testcase_pattern=options.testcase_pattern)

    def warn(self, message):
        from numpy.distutils.misc_util import yellow_text
        print>>sys.stderr,yellow_text('Warning: %s' % (message))
        sys.stderr.flush()
    def info(self, message):
        print>>sys.stdout, message
        sys.stdout.flush()

class ScipyTest(NumpyTest):
    def __init__(self, package=None):
        warnings.warn("ScipyTest is now called NumpyTest; please update your code",
                         DeprecationWarning, stacklevel=2)
        NumpyTest.__init__(self, package)


def importall(package):
    """
    Try recursively to import all subpackages under package.
    """
    if isinstance(package,str):
        package = __import__(package)

    package_name = package.__name__
    package_dir = os.path.dirname(package.__file__)
    for subpackage_name in os.listdir(package_dir):
        subdir = os.path.join(package_dir, subpackage_name)
        if not os.path.isdir(subdir):
            continue
        if not os.path.isfile(os.path.join(subdir,'__init__.py')):
            continue
        name = package_name+'.'+subpackage_name
        try:
            exec 'import %s as m' % (name)
        except Exception, msg:
            print 'Failed importing %s: %s' %(name, msg)
            continue
        importall(m)
    return
