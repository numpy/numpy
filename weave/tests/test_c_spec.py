import unittest
import time
import os,sys

# Note: test_dir is global to this file.  
#       It is made by setup_test_location()

#globals
global test_dir 
test_dir = ''

from scipy_test.testing import *
set_package_path()
from weave import inline_tools,ext_tools,c_spec
from weave.build_tools import msvc_exists, gcc_exists
from weave.catalog import unique_file
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
    """this should probably be in scipy_test.testing
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

#----------------------------------------------------------------------------
# Scalar conversion test classes
#   int, float, complex
#----------------------------------------------------------------------------
class test_int_converter(unittest.TestCase):
    compiler = ''    
    def check_type_match_string(self,level=5):
        s = c_spec.int_converter()
        assert( not s.type_match('string') )
    def check_type_match_int(self,level=5):
        s = c_spec.int_converter()        
        assert(s.type_match(5))
    def check_type_match_float(self,level=5):
        s = c_spec.int_converter()        
        assert(not s.type_match(5.))
    def check_type_match_complex(self,level=5):
        s = c_spec.int_converter()        
        assert(not s.type_match(5.+1j))
    def check_var_in(self,level=5):
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
                    
    def check_int_return(self,level=5):
        mod_name = sys._getframe().f_code.co_name + self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = 1
        code = """
               a=a+2;
               return_val = PyInt_FromLong(a);
               """
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import test'
        b=1
        c = test(b)

        assert( c == 3)

class test_float_converter(unittest.TestCase):    
    compiler = ''
    def check_type_match_string(self,level=5):
        s = c_spec.float_converter()
        assert( not s.type_match('string') )
    def check_type_match_int(self,level=5):
        s = c_spec.float_converter()        
        assert(not s.type_match(5))
    def check_type_match_float(self,level=5):
        s = c_spec.float_converter()        
        assert(s.type_match(5.))
    def check_type_match_complex(self,level=5):
        s = c_spec.float_converter()        
        assert(not s.type_match(5.+1j))
    def check_float_var_in(self,level=5):
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


    def check_float_return(self,level=5):   
        mod_name = sys._getframe().f_code.co_name + self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = 1.
        code = """
               a=a+2.;
               return_val = PyFloat_FromDouble(a);
               """
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import test'
        b=1.
        c = test(b)
        assert( c == 3.)
        
class test_complex_converter(unittest.TestCase):    
    compiler = ''
    def check_type_match_string(self,level=5):
        s = c_spec.complex_converter()
        assert( not s.type_match('string') )
    def check_type_match_int(self,level=5):
        s = c_spec.complex_converter()        
        assert(not s.type_match(5))
    def check_type_match_float(self,level=5):
        s = c_spec.complex_converter()        
        assert(not s.type_match(5.))
    def check_type_match_complex(self,level=5):
        s = c_spec.complex_converter()        
        assert(s.type_match(5.+1j))
    def check_complex_var_in(self,level=5):
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

    def check_complex_return(self,level=5):
        mod_name = sys._getframe().f_code.co_name + self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = 1.+1j
        code = """
               a= a + std::complex<double>(2.,2.);
               return_val = PyComplex_FromDoubles(a.real(),a.imag());
               """
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import test'
        b=1.+1j
        c = test(b)
        assert( c == 3.+3j)

#----------------------------------------------------------------------------
# File conversion tests
#----------------------------------------------------------------------------

class test_file_converter(unittest.TestCase):    
    compiler = ''
    def check_py_to_file(self,level=5):
        import tempfile
        file_name = tempfile.mktemp()        
        file = open(file_name,'w')
        code = """
               fprintf(file,"hello bob");
               """
        inline_tools.inline(code,['file'],compiler=self.compiler,force=1) 
        file.close()
        file = open(file_name,'r')
        assert(file.read() == "hello bob")
    def check_file_to_py(self,level=5):
        import tempfile
        file_name = tempfile.mktemp()        
        # not sure I like Py::String as default -- might move to std::sting
        # or just plain char*
        code = """
               char* _file_name = (char*) file_name.c_str();
               FILE* file = fopen(_file_name,"w");
               return_val = file_to_py(file,_file_name,"w");
               """
        file = inline_tools.inline(code,['file_name'], compiler=self.compiler,
                                   force=1)
        file.write("hello fred")        
        file.close()
        file = open(file_name,'r')
        assert(file.read() == "hello fred")

#----------------------------------------------------------------------------
# Instance conversion tests
#----------------------------------------------------------------------------

class test_instance_converter(unittest.TestCase):    
    pass

#----------------------------------------------------------------------------
# Callable object conversion tests
#----------------------------------------------------------------------------
    
class test_callable_converter(unittest.TestCase):        
    compiler=''
    def check_call_function(self,level=5):
        import string
        func = string.find
        search_str = "hello world hello"
        sub_str = "world"
        # * Not sure about ref counts on search_str and sub_str.
        # * Is the Py::String necessary? (it works anyways...)
        code = """
               py::tuple args(2);
               args[0] = search_str;
               args[1] = sub_str;
               return_val = func.call(args);
               """
        actual = inline_tools.inline(code,['func','search_str','sub_str'],
                                     compiler=self.compiler,force=1)
        desired = func(search_str,sub_str)        
        assert(desired == actual)

class test_sequence_converter(unittest.TestCase):    
    compiler = ''
    def check_convert_to_dict(self,level=5):
        d = {}
        inline_tools.inline("",['d'],compiler=self.compiler,force=1) 
    def check_convert_to_list(self,level=5):        
        l = []
        inline_tools.inline("",['l'],compiler=self.compiler,force=1)
    def check_convert_to_string(self,level=5):        
        s = 'hello'
        inline_tools.inline("",['s'],compiler=self.compiler,force=1)
    def check_convert_to_tuple(self,level=5):        
        t = ()
        inline_tools.inline("",['t'],compiler=self.compiler,force=1)

class test_string_converter(unittest.TestCase):    
    compiler = ''
    def check_type_match_string(self,level=5):
        s = c_spec.string_converter()
        assert( s.type_match('string') )
    def check_type_match_int(self,level=5):
        s = c_spec.string_converter()        
        assert(not s.type_match(5))
    def check_type_match_float(self,level=5):
        s = c_spec.string_converter()        
        assert(not s.type_match(5.))
    def check_type_match_complex(self,level=5):
        s = c_spec.string_converter()        
        assert(not s.type_match(5.+1j))
    def check_var_in(self,level=5):
        mod_name = 'string_var_in'+self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = 'string'
        code = 'a=std::string("hello");'
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)

        exec 'from ' + mod_name + ' import test'
        b='bub'
        test(b)
        try:
            b = 1.
            test(b)
        except TypeError:
            pass
        try:
            b = 1
            test(b)
        except TypeError:
            pass
            
    def check_return(self,level=5):
        mod_name = 'string_return'+self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = 'string'
        code = """
               a= std::string("hello");
               return_val = PyString_FromString(a.c_str());
               """
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import test'
        b='bub'
        c = test(b)
        assert( c == 'hello')

class test_list_converter(unittest.TestCase):    
    compiler = ''
    def check_type_match_bad(self,level=5):
        s = c_spec.list_converter()
        objs = [{},(),'',1,1.,1+1j]
        for i in objs:
            assert( not s.type_match(i) )
    def check_type_match_good(self,level=5):
        s = c_spec.list_converter()        
        assert(s.type_match([]))
    def check_var_in(self,level=5):
        mod_name = 'list_var_in'+self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = [1]
        code = 'a=py::list();'
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import test'
        b=[1,2]
        test(b)
        try:
            b = 1.
            test(b)
        except TypeError:
            pass
        try:
            b = 'string'
            test(b)
        except TypeError:
            pass
            
    def check_return(self,level=5):
        mod_name = 'list_return'+self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = [1]
        code = """
               a=py::list();
               a.append("hello");
               return_val = a;
               """
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import test'
        b=[1,2]
        c = test(b)
        assert( c == ['hello'])
        
    def check_speed(self,level=5):
        mod_name = 'list_speed'+self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = range(1000000);
        code = """
               int v, sum = 0;            
               for(int i = 0; i < a.len(); i++)
               {
                   v = a[i];
                   if (v % 2)
                    sum += v;
                   else
                    sum -= v; 
               }
               return_val = sum;
               """
        with_cxx = ext_tools.ext_function('with_cxx',code,['a'])
        mod.add_function(with_cxx)
        code = """
               int vv, sum = 0;
               PyObject *v;               
               for(int i = 0; i < a.len(); i++)
               {
                   v = PyList_GetItem(py_a,i);
                   //didn't set error here -- just speed test
                   vv = py_to_int(v,"list item");
                   if (vv % 2)
                    sum += vv;
                   else
                    sum -= vv; 
               }
               return_val = sum;
               """
        no_checking = ext_tools.ext_function('no_checking',code,['a'])
        mod.add_function(no_checking)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import with_cxx, no_checking'
        import time
        t1 = time.time()
        sum1 = with_cxx(a)
        t2 = time.time()
        print 'speed test for list access'
        print 'compiler:',  self.compiler
        print 'scxx:',  t2 - t1
        t1 = time.time()
        sum2 = no_checking(a)
        t2 = time.time()
        print 'C, no checking:',  t2 - t1
        sum3 = 0
        t1 = time.time()
        for i in a:
            if i % 2:
                sum3 += i
            else:
                sum3 -= i
        t2 = time.time()
        print 'python:', t2 - t1        
        assert( sum1 == sum2 and sum1 == sum3)

class test_tuple_converter(unittest.TestCase):    
    compiler = ''
    def check_type_match_bad(self,level=5):
        s = c_spec.tuple_converter()
        objs = [{},[],'',1,1.,1+1j]
        for i in objs:
            assert( not s.type_match(i) )
    def check_type_match_good(self,level=5):
        s = c_spec.tuple_converter()        
        assert(s.type_match((1,)))
    def check_var_in(self,level=5):
        mod_name = 'tuple_var_in'+self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = (1,)
        code = 'a=py::tuple();'
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import test'
        b=(1,2)
        test(b)
        try:
            b = 1.
            test(b)
        except TypeError:
            pass
        try:
            b = 'string'
            test(b)
        except TypeError:
            pass
            
    def check_return(self,level=5):
        mod_name = 'tuple_return'+self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = (1,)
        code = """
               a=py::tuple(2);
               a[0] = "hello";
               a.set_item(1,py::None);
               return_val = a;
               """
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import test'
        b=(1,2)
        c = test(b)
        assert( c == ('hello',None))


class test_dict_converter(unittest.TestCase):    
    def check_type_match_bad(self,level=5):
        s = c_spec.dict_converter()
        objs = [[],(),'',1,1.,1+1j]
        for i in objs:
            assert( not s.type_match(i) )
    def check_type_match_good(self,level=5):
        s = c_spec.dict_converter()        
        assert(s.type_match({}))
    def check_var_in(self,level=5):
        mod_name = 'dict_var_in'+self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = {'z':1}
        code = 'a=py::dict();' # This just checks to make sure the type is correct
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import test'
        b={'y':2}
        test(b)
        try:
            b = 1.
            test(b)
        except TypeError:
            pass
        try:
            b = 'string'
            test(b)
        except TypeError:
            pass
            
    def check_return(self,level=5):
        mod_name = 'dict_return'+self.compiler
        mod_name = unique_mod(test_dir,mod_name)
        mod = ext_tools.ext_module(mod_name)
        a = {'z':1}
        code = """
               a=py::dict();
               a["hello"] = 5;
               return_val = a;
               """
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = test_dir, compiler = self.compiler)
        exec 'from ' + mod_name + ' import test'
        b = {'z':2}
        c = test(b)
        assert( c['hello'] == 5)

class test_msvc_int_converter(test_int_converter):    
    compiler = 'msvc'
class test_unix_int_converter(test_int_converter):    
    compiler = ''
class test_gcc_int_converter(test_int_converter):    
    compiler = 'gcc'

class test_msvc_float_converter(test_float_converter):    
    compiler = 'msvc'

class test_msvc_float_converter(test_float_converter):    
    compiler = 'msvc'
class test_unix_float_converter(test_float_converter):    
    compiler = ''
class test_gcc_float_converter(test_float_converter):    
    compiler = 'gcc'

class test_msvc_complex_converter(test_complex_converter):    
    compiler = 'msvc'
class test_unix_complex_converter(test_complex_converter):    
    compiler = ''
class test_gcc_complex_converter(test_complex_converter):    
    compiler = 'gcc'

class test_msvc_file_converter(test_file_converter):    
    compiler = 'msvc'
class test_unix_file_converter(test_file_converter):    
    compiler = ''
class test_gcc_file_converter(test_file_converter):    
    compiler = 'gcc'

class test_msvc_callable_converter(test_callable_converter):    
    compiler = 'msvc'
class test_unix_callable_converter(test_callable_converter):    
    compiler = ''
class test_gcc_callable_converter(test_callable_converter):    
    compiler = 'gcc'

class test_msvc_sequence_converter(test_sequence_converter):    
    compiler = 'msvc'
class test_unix_sequence_converter(test_sequence_converter):    
    compiler = ''
class test_gcc_sequence_converter(test_sequence_converter):    
    compiler = 'gcc'

class test_msvc_string_converter(test_string_converter):    
    compiler = 'msvc'
class test_unix_string_converter(test_string_converter):    
    compiler = ''
class test_gcc_string_converter(test_string_converter):    
    compiler = 'gcc'

class test_msvc_list_converter(test_list_converter):    
    compiler = 'msvc'
class test_unix_list_converter(test_list_converter):    
    compiler = ''
class test_gcc_list_converter(test_list_converter):    
    compiler = 'gcc'

class test_msvc_tuple_converter(test_tuple_converter):    
    compiler = 'msvc'
class test_unix_tuple_converter(test_tuple_converter):    
    compiler = ''
class test_gcc_tuple_converter(test_tuple_converter):    
    compiler = 'gcc'

class test_msvc_dict_converter(test_dict_converter):    
    compiler = 'msvc'
class test_unix_dict_converter(test_dict_converter):    
    compiler = ''
class test_gcc_dict_converter(test_dict_converter):    
    compiler = 'gcc'

class test_msvc_instance_converter(test_instance_converter):    
    compiler = 'msvc'
class test_unix_instance_converter(test_instance_converter):    
    compiler = ''
class test_gcc_instance_converter(test_instance_converter):    
    compiler = 'gcc'
    
def setup_test_location():
    import tempfile
    #test_dir = os.path.join(tempfile.gettempdir(),'test_files')
    test_dir = tempfile.mktemp()
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    sys.path.insert(0,test_dir)    
    return test_dir

test_dir = setup_test_location()

def teardown_test_location():
    import tempfile
    test_dir = os.path.join(tempfile.gettempdir(),'test_files')
    if sys.path[0] == test_dir:
        sys.path = sys.path[1:]
    return test_dir

def remove_file(name):
    test_dir = os.path.abspath(name)

if not msvc_exists():
    for _n in dir():
        if _n[:10]=='test_msvc_': exec 'del '+_n
else:
    for _n in dir():
        if _n[:10]=='test_unix_': exec 'del '+_n

if not (gcc_exists() and msvc_exists() and sys.platform == 'win32'):
    for _n in dir():
        if _n[:9]=='test_gcc_': exec 'del '+_n

if __name__ == "__main__":
    ScipyTest('weave.c_spec').run()
