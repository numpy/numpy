
__all__ = []

import os,sys,time,glob,string,traceback,unittest
import types
import imp

try:
    # These are used by Numeric tests.
    # If Numeric and scipy_base  are not available, then some of the
    # functions below will not be available.
    from Numeric import alltrue,equal,shape,ravel,around,zeros,Float64,asarray,\
         less_equal,array2string,less
    import scipy_base.fastumath as math
except ImportError:
    pass

DEBUG=0

__all__.append('set_package_path')
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
    from scipy_distutils.misc_util import get_frame
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

__all__.append('set_local_path')
def set_local_path(reldir='', level=1):
    """ Prepend local directory to sys.path.

    The caller is responsible for removing this path by using

      restore_path()
    """
    from scipy_distutils.misc_util import get_frame
    f = get_frame(level)
    if f.f_locals['__name__']=='__main__':
        testfile = sys.argv[0]
    else:
        testfile = f.f_locals['__file__']
    local_path = os.path.join(os.path.dirname(os.path.abspath(testfile)),reldir)
    if DEBUG:
        print 'Inserting %r to sys.path' % (local_path)
    sys.path.insert(0,local_path)

__all__.append('restore_path')
def restore_path():
    if DEBUG:
        print 'Removing %r from sys.path' % (sys.path[0])
    del sys.path[0]

if sys.platform[:5]=='linux':
    def jiffies(_proc_pid_stat = '/proc/%s/stat'%(os.getpid()),
                _load_time=time.time()):
        """ Return number of jiffies (1/100ths of a second) that this
    process has been scheduled in user mode. See man 5 proc. """
        try:
            f=open(_proc_pid_stat,'r')
            l = f.readline().split(' ')
            f.close()
            return int(l[13])
        except:
            return int(100*(time.time()-_load_time))

    def memusage(_proc_pid_stat = '/proc/%s/stat'%(os.getpid())):
        """ Return virtual memory size in bytes of the running python.
        """
        try:
            f=open(_proc_pid_stat,'r')
            l = f.readline().split(' ')
            f.close()
            return int(l[22])
        except:
            return
else:
    # os.getpid is not in all platforms available.
    # Using time is safe but inaccurate, especially when process
    # was suspended or sleeping.
    def jiffies(_load_time=time.time()):
        """ Return number of jiffies (1/100ths of a second) that this
    process has been scheduled in user mode. [Emulation with time.time]. """
        return int(100*(time.time()-_load_time))

    def memusage():
        """ Return memory usage of running python. [Not implemented]"""
        return

__all__.append('ScipyTestCase')
class ScipyTestCase (unittest.TestCase):

    def measure(self,code_str,times=1):
        """ Return elapsed time for executing code_str in the
        namespace of the caller for given times.
        """
        frame = sys._getframe(1)
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
        result.stream = _dummy_stream()
        unittest.TestCase.__call__(self, result)
        if nof_errors != len(result.errors):
            test, errstr = result.errors[-1]
            if errstr.split('\n')[-2].startswith('IgnoreException:'):
                assert result.stream.data[-1]=='E',`result.stream.data`
                result.stream.data[-1] = 'i'
                del result.errors[-1]
        map(save_stream.write, result.stream.data)
        result.stream = save_stream

class _dummy_stream:
    def __init__(self):
        self.data = []
    def write(self,message):
        self.data.append(message)
    def writeln(self,message):
        self.data.append(message+'\n')

__all__.append('IgnoreException')
class IgnoreException(Exception):
    "Ignoring this exception due to disabled feature"

#------------

def _get_all_method_names(cls):
    names = dir(cls)
    if sys.version[:3]<='2.1':
        for b in cls.__bases__:
            for n in dir(b)+_get_all_method_names(b):
                if n not in names:
                    names.append(n)
    return names

__all__.append('ScipyTest')
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
    def __init__(self, package='__main__'):
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

    def _get_module_tests(self,module,level):
        mstr = self._module_str
        d,f = os.path.split(module.__file__)

        short_module_name = os.path.splitext(os.path.basename(f))[0]
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
            suite_list.extend(map(obj,self._get_method_names(obj,level)))
        print '  Found',len(suite_list),'tests for',module_name
        return suite_list

    def _touch_ppimported(self, module):
        from scipy_base.ppimport import _ModuleLoader
        if os.path.isdir(os.path.join(os.path.dirname(module.__file__),'tests')):
            # only touching those modules that have tests/ directory
            try: module._pliuh_plauh
            except AttributeError: pass
            for name in dir(module):
                obj = getattr(module,name)
                if isinstance(obj,_ModuleLoader) \
                   and not hasattr(obj,'_ppimport_module') \
                   and not hasattr(obj,'_ppimport_exc_info'):
                    self._touch_ppimported(obj)

    def test(self,level=1,verbosity=1):
        """ Run Scipy module test suite with level and verbosity.
        """
        if type(self.package) is type(''):
            exec 'import %s as this_package' % (self.package)
        else:
            this_package = self.package

        self._touch_ppimported(this_package)

        package_name = this_package.__name__

        suites = []
        for name, module in sys.modules.items():
            if package_name != name[:len(package_name)] \
                   or module is None \
                   or os.path.basename(os.path.dirname(module.__file__))=='tests':
                continue
            suites.extend(self._get_module_tests(module, level))

        suites.extend(self._get_suite_list(sys.modules[package_name], level))

        all_tests = unittest.TestSuite(suites)
        runner = unittest.TextTestRunner(verbosity=verbosity)
        runner.run(all_tests)
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
        usage = r'usage: %prog [<options>]'
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

#------------
        
def remove_ignored_patterns(files,pattern):
    from fnmatch import fnmatch
    good_files = []
    for file in files:
        if not fnmatch(file,pattern):
            good_files.append(file)
    return good_files

def remove_ignored_files(original,ignored_files,cur_dir):
    """ This is actually expanded to do pattern matching.

    """
    if not ignored_files: ignored_files = []
    ignored_modules = map(lambda x: x+'.py',ignored_files)
    ignored_packages = ignored_files[:]
    # always ignore setup.py and __init__.py files
    ignored_files = ['setup.py','setup_*.py','__init__.py']
    ignored_files += ignored_modules + ignored_packages
    ignored_files = map(lambda x,cur_dir=cur_dir: os.path.join(cur_dir,x),
                        ignored_files)
    #print 'ignored:', ignored_files
    #good_files = filter(lambda x,ignored = ignored_files: x not in ignored,
    #                    original)
    good_files = original
    for pattern in ignored_files:
        good_files = remove_ignored_patterns(good_files,pattern)

    return good_files

__all__.append('harvest_modules')
def harvest_modules(package,ignore=None):
    """* Retreive a list of all modules that live within a package.

         Only retreive files that are immediate children of the
         package -- do not recurse through child packages or
         directories.  The returned list contains actual modules, not
         just their names.
    *"""
    d,f = os.path.split(package.__file__)

    # go through the directory and import every py file there.
    common_dir = os.path.join(d,'*.py')
    py_files = glob.glob(common_dir)
    #py_files.remove(os.path.join(d,'__init__.py'))
    #py_files.remove(os.path.join(d,'setup.py'))

    py_files = remove_ignored_files(py_files,ignore,d)
    #print 'py_files:', py_files
    try:
        prefix = package.__name__
    except:
        prefix = ''

    all_modules = []
    for file in py_files:
        d,f = os.path.split(file)
        base,ext =  os.path.splitext(f)
        mod = prefix + '.' + base
        #print 'module: import ' + mod
        try:
            exec ('import ' + mod)
            all_modules.append(eval(mod))
        except:
            print 'FAILURE to import ' + mod
            output_exception()

    return all_modules

__all__.append('harvest_packages')
def harvest_packages(package,ignore = None):
    """ Retreive a list of all sub-packages that live within a package.

         Only retreive packages that are immediate children of this
         package -- do not recurse through child packages or
         directories.  The returned list contains actual package objects, not
         just their names.
    """
    join = os.path.join

    d,f = os.path.split(package.__file__)

    common_dir = os.path.abspath(d)
    all_files = os.listdir(d)

    all_files = remove_ignored_files(all_files,ignore,'')
    #print 'all_files:', all_files
    try:
        prefix = package.__name__
    except:
        prefix = ''
    all_packages = []
    for directory in all_files:
        path = join(common_dir,directory)
        if os.path.isdir(path) and \
           os.path.exists(join(path,'__init__.py')):
            sub_package = prefix + '.' + directory
            #print 'sub-package import ' + sub_package
            try:
                exec ('import ' + sub_package)
                all_packages.append(eval(sub_package))
            except:
                print 'FAILURE to import ' + sub_package
                output_exception()
    return all_packages

__all__.append('harvest_modules_and_packages')
def harvest_modules_and_packages(package,ignore=None):
    """ Retreive list of all packages and modules that live within a package.

         See harvest_packages() and harvest_modules()
    """
    all = harvest_modules(package,ignore) + harvest_packages(package,ignore)
    return all

__all__.append('harvest_test_suites')
def harvest_test_suites(package,ignore = None,level=10):
    """
        package -- the module to test.  This is an actual module object
                   (not a string)
        ignore  -- a list of module names to omit from the tests
        level   -- a value between 1 and 10.  1 will run the minimum number
                   of tests.  This is a fast "smoke test".  Tests that take
                   longer to run should have higher numbers ranging up to 10.
    """
    suites=[]
    test_modules = harvest_modules_and_packages(package,ignore)
    #for i in test_modules:
    #    print i.__name__
    for module in test_modules:
        if hasattr(module,'test_suite'):
            try:
                suite = module.test_suite(level=level)
                if suite:
                    suites.append(suite)
                else:
                    print "    !! FAILURE without error - shouldn't happen",
                    print module.__name__
            except:
                print '   !! FAILURE building test for ', module.__name__
                print '   ',
                output_exception()
        else:
            try:
                print 'No test suite found for ', module.__name__
            except AttributeError:
                # __version__.py getting replaced by a string throws a kink
                # in checking for modules, so we think is a module has
                # actually been overwritten
                print 'No test suite found for ', str(module)
    total_suite = unittest.TestSuite(suites)
    return total_suite

__all__.append('module_test')
def module_test(mod_name,mod_file,level=10):
    """*

    *"""
    #print 'testing', mod_name
    d,f = os.path.split(mod_file)

    # insert the tests directory to the python path
    test_dir = os.path.join(d,'tests')
    sys.path.insert(0,test_dir)

    # call the "test_xxx.test()" function for the appropriate
    # module.

    # This should deal with package naming issues correctly
    short_mod_name = string.split(mod_name,'.')[-1]
    test_module = 'test_' + short_mod_name
    test_string = 'import %s;reload(%s);%s.test(%d)' % \
                  ((test_module,)*3 + (level,))

    # This would be better cause it forces a reload of the orginal
    # module.  It doesn't behave with packages however.
    #test_string = 'reload(%s);import %s;reload(%s);%s.test(%d)' % \
    #              ((mod_name,) + (test_module,)*3)
    exec(test_string)

    # remove test directory from python path.
    sys.path = sys.path[1:]

__all__.append('module_test_suite')
def module_test_suite(mod_name,mod_file,level=10):
    #try:
        print ' creating test suite for:', mod_name
        d,f = os.path.split(mod_file)

        # insert the tests directory to the python path
        test_dir = os.path.join(d,'tests')
        sys.path.insert(0,test_dir)

        # call the "test_xxx.test()" function for the appropriate
        # module.

        # This should deal with package naming issues correctly
        short_mod_name = string.split(mod_name,'.')[-1]
        test_module = 'test_' + short_mod_name
        test_string = 'import %s;reload(%s);suite = %s.test_suite(%d)' % \
                      ((test_module,)*3+(level,))
        #print test_string
        exec(test_string)

        # remove test directory from python path.
        sys.path = sys.path[1:]
        return suite
    #except:
    #    print '    !! FAILURE loading test suite from', test_module, ':'
    #    print '   ',
    #    output_exception()


# Utility function to facilitate testing.

__all__.append('assert_equal')
def assert_equal(actual,desired,err_msg='',verbose=1):
    """ Raise an assertion if two items are not
        equal.  I think this should be part of unittest.py
    """
    msg = '\nItems are not equal:\n' + err_msg
    try:
        if ( verbose and len(repr(desired)) < 100 and len(repr(actual)) ):
            msg =  msg \
                 + 'DESIRED: ' + repr(desired) \
                 + '\nACTUAL: ' + repr(actual)
    except:
        msg =  msg \
             + 'DESIRED: ' + repr(desired) \
             + '\nACTUAL: ' + repr(actual)
    assert desired == actual, msg

__all__.append('assert_almost_equal')
def assert_almost_equal(actual,desired,decimal=7,err_msg='',verbose=1):
    """ Raise an assertion if two items are not
        equal.  I think this should be part of unittest.py
    """
    msg = '\nItems are not equal:\n' + err_msg
    try:
        if ( verbose and len(repr(desired)) < 100 and len(repr(actual)) ):
            msg =  msg \
                 + 'DESIRED: ' + repr(desired) \
                 + '\nACTUAL: ' + repr(actual)
    except:
        msg =  msg \
             + 'DESIRED: ' + repr(desired) \
             + '\nACTUAL: ' + repr(actual)
    assert round(abs(desired - actual),decimal) == 0, msg

__all__.append('assert_approx_equal')
def assert_approx_equal(actual,desired,significant=7,err_msg='',verbose=1):
    """ Raise an assertion if two items are not
        equal.  I think this should be part of unittest.py
        Approximately equal is defined as the number of significant digits
        correct
    """
    msg = '\nItems are not equal to %d significant digits:\n' % significant
    msg += err_msg
    actual, desired = map(float, (actual, desired))
    # Normalized the numbers to be in range (-10.0,10.0)
    scale = pow(10,math.floor(math.log10(0.5*(abs(desired)+abs(actual)))))
    try:
        sc_desired = desired/scale
    except ZeroDivisionError:
        sc_desired = 0.0
    try:
        sc_actual = actual/scale
    except ZeroDivisionError:
        sc_actual = 0.0
    try:
        if ( verbose and len(repr(desired)) < 100 and len(repr(actual)) ):
            msg =  msg \
                 + 'DESIRED: ' + repr(desired) \
                 + '\nACTUAL: ' + repr(actual)
    except:
        msg =  msg \
             + 'DESIRED: ' + repr(desired) \
             + '\nACTUAL: ' + repr(actual)
    assert math.fabs(sc_desired - sc_actual) < pow(10.,-1*significant), msg


__all__.append('assert_array_equal')
def assert_array_equal(x,y,err_msg=''):
    x,y = asarray(x), asarray(y)
    msg = '\nArrays are not equal'
    try:
        assert 0 in [len(shape(x)),len(shape(y))] \
               or (len(shape(x))==len(shape(y)) and \
                   alltrue(equal(shape(x),shape(y)))),\
                   msg + ' (shapes %s, %s mismatch):\n\t' \
                   % (shape(x),shape(y)) + err_msg
        reduced = ravel(equal(x,y))
        cond = alltrue(reduced)
        if not cond:
            s1 = array2string(x,precision=16)
            s2 = array2string(y,precision=16)
            if len(s1)>120: s1 = s1[:120] + '...'
            if len(s2)>120: s2 = s2[:120] + '...'
            match = 100-100.0*reduced.tolist().count(1)/len(reduced)
            msg = msg + ' (mismatch %s%%):\n\tArray 1: %s\n\tArray 2: %s' % (match,s1,s2)
        assert cond,\
               msg + '\n\t' + err_msg
    except ValueError:
        raise ValueError, msg

__all__.append('assert_array_almost_equal')
def assert_array_almost_equal(x,y,decimal=6,err_msg=''):
    x = asarray(x)
    y = asarray(y)
    msg = '\nArrays are not almost equal'
    try:
        cond = alltrue(equal(shape(x),shape(y)))
        if not cond:
            msg = msg + ' (shapes mismatch):\n\t'\
                  'Shape of array 1: %s\n\tShape of array 2: %s' % (shape(x),shape(y))
        assert cond, msg + '\n\t' + err_msg
        reduced = ravel(equal(less_equal(around(abs(x-y),decimal),10.0**(-decimal)),1))
        cond = alltrue(reduced)
        if not cond:
            s1 = array2string(x,precision=decimal+1)
            s2 = array2string(y,precision=decimal+1)
            if len(s1)>120: s1 = s1[:120] + '...'
            if len(s2)>120: s2 = s2[:120] + '...'
            match = 100-100.0*reduced.tolist().count(1)/len(reduced)
            msg = msg + ' (mismatch %s%%):\n\tArray 1: %s\n\tArray 2: %s' % (match,s1,s2)
        assert cond,\
               msg + '\n\t' + err_msg
    except ValueError:
        print sys.exc_value
        print shape(x),shape(y)
        print x, y
        raise ValueError, 'arrays are not almost equal'

__all__.append('assert_array_less')
def assert_array_less(x,y,err_msg=''):
    x,y = asarray(x), asarray(y)
    msg = '\nArrays are not less-ordered'
    try:
        assert alltrue(equal(shape(x),shape(y))),\
               msg + ' (shapes mismatch):\n\t' + err_msg
        reduced = ravel(less(x,y))
        cond = alltrue(reduced)
        if not cond:
            s1 = array2string(x,precision=16)
            s2 = array2string(y,precision=16)
            if len(s1)>120: s1 = s1[:120] + '...'
            if len(s2)>120: s2 = s2[:120] + '...'
            match = 100-100.0*reduced.tolist().count(1)/len(reduced)
            msg = msg + ' (mismatch %s%%):\n\tArray 1: %s\n\tArray 2: %s' % (match,s1,s2)
        assert cond,\
               msg + '\n\t' + err_msg
    except ValueError:
        print shape(x),shape(y)
        raise ValueError, 'arrays are not less-ordered'

__all__.append('rand')
def rand(*args):
    """ Returns an array of random numbers with the given shape.
    used for testing
    """
    import random
    results = zeros(args,Float64)
    f = results.flat
    for i in range(len(f)):
        f[i] = random.random()
    return results

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
