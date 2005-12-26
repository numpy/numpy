
import os
import sys
import imp
import types
import unittest
import traceback

__all__ = ['set_package_path', 'set_local_path', 'restore_path',
           'IgnoreException', 'ScipyTestCase', 'ScipyTest']

DEBUG=0
get_frame = sys._getframe
from utils import jiffies


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
        print 'Inserting %r to sys.path' % (d1)
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
    local_path = os.path.join(os.path.dirname(os.path.abspath(testfile)),reldir)
    if DEBUG:
        print 'Inserting %r to sys.path' % (local_path)
    sys.path.insert(0,local_path)
    return


def restore_path():
    if DEBUG:
        print 'Removing %r from sys.path' % (sys.path[0])
    del sys.path[0]
    return


def output_exception():
    try:
        type, value, tb = sys.exc_info()
        info = traceback.extract_tb(tb)
        #this is more verbose
        #traceback.print_exc()
        filename, lineno, function, text = info[-1] # last line only
        print "%s:%d: %s: %s (in %s)" %\
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


class ScipyTestCase (unittest.TestCase):

    def measure(self,code_str,times=1):
        """ Return elapsed time for executing code_str in the
        namespace of the caller for given times.
        """
        frame = get_frame(1)
        locs,globs = frame.f_locals,frame.f_globals
        code = compile(code_str,
                       'ScipyTestCase runner for '+self.__class__.__name__,
                       'exec')
        i = 0
        elapsed = jiffies()
        while i<times:
            i += 1
            exec code in globs,locs
        elapsed = jiffies() - elapsed
        return 0.01*elapsed

    def __call__(self, result=None):
        if result is None:
            return unittest.TestCase.__call__(self, result)

        nof_errors = len(result.errors)
        save_stream = result.stream
        result.stream = _dummy_stream(save_stream)
        unittest.TestCase.__call__(self, result)
        if nof_errors != len(result.errors):
            test, errstr = result.errors[-1]
            if type(errstr) is type(()):
                errstr = str(errstr[0])
            else:
                errstr = errstr.split('\n')[-2]
            l = len(result.stream.data)
            if errstr.startswith('IgnoreException:'):
                if l==1:
                    assert result.stream.data[-1]=='E',`result.stream.data`
                    result.stream.data[-1] = 'i'
                else:
                    assert result.stream.data[-1]=='ERROR\n',`result.stream.data`
                    result.stream.data[-1] = 'ignoring\n'
                del result.errors[-1]
        map(save_stream.write, result.stream.data)
        result.stream = save_stream

def _get_all_method_names(cls):
    names = dir(cls)
    if sys.version[:3]<='2.1':
        for b in cls.__bases__:
            for n in dir(b)+_get_all_method_names(b):
                if n not in names:
                    names.append(n)
    return names


# for debug build--check for memory leaks during the test.
class _SciPyTextTestResult(unittest._TextTestResult):
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

class SciPyTextTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return _SciPyTextTestResult(self.stream, self.descriptions, self.verbosity)
    

class ScipyTest:
    """ Scipy tests site manager.

    Usage:
      >>> ScipyTest(<package>).test(level=1,verbosity=2)

    <package> is package name or its module object.

    Package is supposed to contain a directory tests/
    with test_*.py files where * refers to the names of submodules.

    test_*.py files are supposed to define a classes, derived
    from ScipyTestCase or unittest.TestCase, with methods having
    names starting with test or bench or check.

    And that is it! No need to implement test or test_suite functions
    in each .py file.

    Also old styled test_suite(level=1) hooks are supported but
    soon to be removed.
    """
    def __init__(self, package=None):
        if package is None:
            from scipy.distutils.misc_util import get_frame
            f = get_frame(1)
            package = f.f_locals['__name__']
        self.package = package

    def _module_str(self, module):
        filename = module.__file__[-30:]
        if filename!=module.__file__:
            filename = '...'+filename
        return '<module %s from %s>' % (`module.__name__`, `filename`)

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

    def _get_module_tests(self,module,level,verbosity):
        mstr = self._module_str
        d,f = os.path.split(module.__file__)

        short_module_name = os.path.splitext(os.path.basename(f))[0]
        if short_module_name=='__init__':
            short_module_name = module.__name__.split('.')[-1]

        test_dir = os.path.join(d,'tests')
        test_file = os.path.join(test_dir,'test_'+short_module_name+'.py')

        local_test_dir = os.path.join(os.getcwd(),'tests')
        local_test_file = os.path.join(local_test_dir,
                                       'test_'+short_module_name+'.py')
        if os.path.basename(os.path.dirname(local_test_dir)) \
               == os.path.basename(os.path.dirname(test_dir)) \
           and os.path.isfile(local_test_file):
            test_file = local_test_file

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
                print test_file
                print '   !! No test file %r found for %s' \
                      % (os.path.basename(test_file), mstr(module))
            return []

        try:
            if sys.version[:3]=='2.1':
                # Workaround for Python 2.1 .pyc file generator bug
                import random
                pref = '-nopyc'+`random.randint(1,100)`
            else:
                pref = ''
            f = open(test_file,'r')
            test_module = imp.load_module(\
                module.__name__+'.test_'+short_module_name+pref,
                f, test_file+pref,('.py', 'r', 1))
            f.close()
            if sys.version[:3]=='2.1' and os.path.isfile(test_file+pref+'c'):
                os.remove(test_file+pref+'c')
        except:
            print '   !! FAILURE importing tests for ', mstr(module)
            print '   ',
            output_exception()
            return []
        return self._get_suite_list(test_module, level, module.__name__)

    def _get_suite_list(self, test_module, level, module_name='__main__'):
        mstr = self._module_str
        if hasattr(test_module,'test_suite'):
            # Using old styled test suite
            try:
                total_suite = test_module.test_suite(level)
                return total_suite._tests
            except:
                print '   !! FAILURE building tests for ', mstr(test_module)
                print '   ',
                output_exception()
                return []
        suite_list = []
        for name in dir(test_module):
            obj = getattr(test_module, name)
            if type(obj) is not type(unittest.TestCase) \
               or not issubclass(obj, unittest.TestCase) \
               or obj.__name__[:4] != 'test':
                continue
            for mthname in self._get_method_names(obj,level):
                suite = obj(mthname)
                if getattr(suite,'isrunnable',lambda mthname:1)(mthname):
                    suite_list.append(suite)
        print '  Found',len(suite_list),'tests for',module_name
        return suite_list

    def test(self,level=1,verbosity=1):
        """ Run Scipy module test suite with level and verbosity.
        """
        if type(self.package) is type(''):
            exec 'import %s as this_package' % (self.package)
        else:
            this_package = self.package

        package_name = this_package.__name__

        suites = []
        for name, module in sys.modules.items():
            if package_name != name[:len(package_name)] \
                   or module is None:
                continue
            if os.path.basename(os.path.dirname(module.__file__))=='tests':
                continue
            suites.extend(self._get_module_tests(module, level, verbosity))

        suites.extend(self._get_suite_list(sys.modules[package_name], level))

        all_tests = unittest.TestSuite(suites)
        #if hasattr(sys,'getobjects'):
        #    runner = SciPyTextTestRunner(verbosity=verbosity)
        #else:
        runner = unittest.TextTestRunner(verbosity=verbosity)
        # Use the builtin displayhook. If the tests are being run
        # under IPython (for instance), any doctest test suites will
        # fail otherwise.
        old_displayhook = sys.displayhook
        sys.displayhook = sys.__displayhook__
        try:
            runner.run(all_tests)
        finally:
            sys.displayhook = old_displayhook
        return runner

    def run(self):
        """ Run Scipy module test suite with level and verbosity
        taken from sys.argv. Requires optparse module.
        """
        try:
            from optparse import OptionParser
        except ImportError:
            print 'Failed to import optparse module, ignoring.'
            return self.test()
        usage = r'usage: %prog [-v <verbosity>] [-l <level>]'
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
        (options, args) = parser.parse_args()
        self.test(options.level,options.verbosity)
        return
