import unittest
import time
import os,sys

from scipy_distutils.misc_util import add_grandparent_to_path, restore_path

add_grandparent_to_path(__name__)
import ext_tools
from catalog import unique_file
from build_tools import msvc_exists, gcc_exists
import scalar_spec
restore_path()

def unique_mod(d,file_name):
    f = os.path.basename(unique_file(d,file_name))
    m = os.path.splitext(f)[0]
    return m
    
def remove_whitespace(in_str):
    import string
    out = string.replace(in_str," ","")
    out = string.replace(out,"\t","")
    out = string.replace(out,"\n","")
    return out
   
def print_assert_equal(test_string,actual,desired):
    """this should probably be in scipy_base.testing
    """
    import pprint
    try:
        assert(actual == desired)
    except AssertionError:
        import cStringIO
        msg = cStringIO.StringIO()
        msg.write(test_string)
        msg.write(' failed\nACTUAL: \n')
        pprint.pprint(actual,msg)
        msg.write('DESIRED: \n')
        pprint.pprint(desired,msg)
        raise AssertionError, msg.getvalue()

class test_int_converter(unittest.TestCase):
    compiler = ''    
    def check_type_match_string(self):
        s = scalar_spec.int_converter()
        assert( not s.type_match('string') )
    def check_type_match_int(self):
        s = scalar_spec.int_converter()        
        assert(s.type_match(5))
    def check_type_match_float(self):
        s = scalar_spec.int_converter()        
        assert(not s.type_match(5.))
    def check_type_match_complex(self):
        s = scalar_spec.int_converter()        
        assert(not s.type_match(5.+1j))
    def check_var_in(self):
        test_dir = setup_test_location()
        mod_name = 'int_var_in' + self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = 1
        code = "a=2;"
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import test'
        b=1
        test(b)
        try:
            b = 1.
            test(b)
        except TypeError:
            pass
        try:
            b = 'abc'
            test(b)
        except TypeError:
            pass
        teardown_test_location()
                    
    def check_int_var_local(self):
        test_dir = setup_test_location()
        mod_name = sys._getframe().f_code.co_name + self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = 1
        code = "a=2;"
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler= self.compiler)
        exec 'from ' + mod_name + ' import test'
        b=1
        q={}
        test(b,q)
        teardown_test_location()
        assert(q['a'] == 2)
    def check_int_return(self):
        test_dir = setup_test_location()        
        mod_name = sys._getframe().f_code.co_name + self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = 1
        code = """
               a=a+2;
               return_val = Py::new_reference_to(Py::Int(a));
               """
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import test'
        b=1
        c = test(b)
        teardown_test_location()
        assert( c == 3)

class test_float_converter(unittest.TestCase):    
    compiler = ''
    def check_type_match_string(self):
        s = scalar_spec.float_converter()
        assert( not s.type_match('string') )
    def check_type_match_int(self):
        s = scalar_spec.float_converter()        
        assert(not s.type_match(5))
    def check_type_match_float(self):
        s = scalar_spec.float_converter()        
        assert(s.type_match(5.))
    def check_type_match_complex(self):
        s = scalar_spec.float_converter()        
        assert(not s.type_match(5.+1j))
    def check_float_var_in(self):
        test_dir = setup_test_location()                
        mod_name = sys._getframe().f_code.co_name + self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)

        a = 1.
        code = "a=2.;"
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import test'
        b=1.
        test(b)
        try:
            b = 1.
            test(b)
        except TypeError:
            pass
        try:
            b = 'abc'
            test(b)
        except TypeError:
            pass
        teardown_test_location()
    def check_float_var_local(self):
        test_dir = setup_test_location()                
        mod_name = sys._getframe().f_code.co_name + self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = 1.
        code = "a=2.;"
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import test'
        b=1.
        q={}
        test(b,q)
        teardown_test_location()
        assert(q['a'] == 2.)
    def check_float_return(self):
        test_dir = setup_test_location()        
        mod_name = sys._getframe().f_code.co_name + self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = 1.
        code = """
               a=a+2.;
               return_val = Py::new_reference_to(Py::Float(a));
               """
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import test'
        b=1.
        c = test(b)
        teardown_test_location()
        assert( c == 3.)
        
class test_complex_converter(unittest.TestCase):    
    compiler = ''
    def check_type_match_string(self):
        s = scalar_spec.complex_converter()
        assert( not s.type_match('string') )
    def check_type_match_int(self):
        s = scalar_spec.complex_converter()        
        assert(not s.type_match(5))
    def check_type_match_float(self):
        s = scalar_spec.complex_converter()        
        assert(not s.type_match(5.))
    def check_type_match_complex(self):
        s = scalar_spec.complex_converter()        
        assert(s.type_match(5.+1j))
    def check_complex_var_in(self):
        test_dir = setup_test_location()        
        mod_name = sys._getframe().f_code.co_name + self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = 1.+1j
        code = "a=std::complex<double>(2.,2.);"
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import test'
        b=1.+1j
        test(b)
        try:
            b = 1.
            test(b)
        except TypeError:
            pass
        try:
            b = 'abc'
            test(b)
        except TypeError:
            pass
        teardown_test_location()
    def check_complex_var_local(self):
        test_dir = setup_test_location()        
        mod_name = sys._getframe().f_code.co_name + self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = 1.+1j
        code = "a= a + std::complex<double>(2.,2.);"
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import test'
        b=1.+1j
        q={}
        test(b,q)
        teardown_test_location()        
        assert(q['a'] == 3.+3j)
    def check_complex_return(self):
        test_dir = setup_test_location()        
        mod_name = sys._getframe().f_code.co_name + self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = 1.+1j
        code = """
               a= a + std::complex<double>(2.,2.);
               return_val = Py::new_reference_to(Py::Complex(a.real(),a.imag()));
               """
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import test'
        b=1.+1j
        c = test(b)
        teardown_test_location()        
        assert( c == 3.+3j)

class test_msvc_int_converter(test_int_converter):    
    compiler = 'msvc'
class test_msvc_float_converter(test_float_converter):    
    compiler = 'msvc'
class test_msvc_complex_converter(test_complex_converter):    
    compiler = 'msvc'

class test_unix_int_converter(test_int_converter):    
    compiler = ''
class test_unix_float_converter(test_float_converter):    
    compiler = ''
class test_unix_complex_converter(test_complex_converter):    
    compiler = ''

class test_gcc_int_converter(test_int_converter):    
    compiler = 'gcc'
class test_gcc_float_converter(test_float_converter):    
    compiler = 'gcc'
class test_gcc_complex_converter(test_complex_converter):    
    compiler = 'gcc'
    

def setup_test_location():
    import tempfile
    test_dir = os.path.join(tempfile.gettempdir(),'test_files')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    sys.path.insert(0,test_dir)    
    return test_dir

def teardown_test_location():
    import tempfile
    test_dir = os.path.join(tempfile.gettempdir(),'test_files')
    if sys.path[0] == test_dir:
        sys.path = sys.path[1:]
    return test_dir

def remove_file(name):
    test_dir = os.path.abspath(name)
    
def test_suite(level=1):
    suites = []    
    if level >= 5:
        if msvc_exists():
            suites.append( unittest.makeSuite(test_msvc_int_converter,
                           'check_'))
            suites.append( unittest.makeSuite(test_msvc_float_converter,
                           'check_'))    
            suites.append( unittest.makeSuite(test_msvc_complex_converter,
                           'check_'))
            pass
        else: # unix
            suites.append( unittest.makeSuite(test_unix_int_converter,
                           'check_'))
            suites.append( unittest.makeSuite(test_unix_float_converter,
                           'check_'))    
            suites.append( unittest.makeSuite(test_unix_complex_converter,
                           'check_'))
        
        if gcc_exists():        
            suites.append( unittest.makeSuite(test_gcc_int_converter,
                           'check_'))
            suites.append( unittest.makeSuite(test_gcc_float_converter,
                           'check_'))
            suites.append( unittest.makeSuite(test_gcc_complex_converter,
                           'check_'))

    total_suite = unittest.TestSuite(suites)
    return total_suite

def test(level=10):
    all_tests = test_suite(level)
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
