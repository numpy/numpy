import unittest
import time
import os,sys

from scipy_distutils.misc_util import add_grandparent_to_path, restore_path

add_grandparent_to_path(__name__)
import inline_tools
import ext_tools
from catalog import unique_file
from build_tools import msvc_exists, gcc_exists
import c_spec
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
    def check_type_match_string(self):
        s = c_spec.int_converter()
        assert( not s.type_match('string') )
    def check_type_match_int(self):
        s = c_spec.int_converter()        
        assert(s.type_match(5))
    def check_type_match_float(self):
        s = c_spec.int_converter()        
        assert(not s.type_match(5.))
    def check_type_match_complex(self):
        s = c_spec.int_converter()        
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
                    
    def check_int_return(self):
        test_dir = setup_test_location()        
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
        teardown_test_location()
        assert( c == 3)

class test_float_converter(unittest.TestCase):    
    compiler = ''
    def check_type_match_string(self):
        s = c_spec.float_converter()
        assert( not s.type_match('string') )
    def check_type_match_int(self):
        s = c_spec.float_converter()        
        assert(not s.type_match(5))
    def check_type_match_float(self):
        s = c_spec.float_converter()        
        assert(s.type_match(5.))
    def check_type_match_complex(self):
        s = c_spec.float_converter()        
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

    def check_float_return(self):
        test_dir = setup_test_location()        
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
        teardown_test_location()
        assert( c == 3.)
        
class test_complex_converter(unittest.TestCase):    
    compiler = ''
    def check_type_match_string(self):
        s = c_spec.complex_converter()
        assert( not s.type_match('string') )
    def check_type_match_int(self):
        s = c_spec.complex_converter()        
        assert(not s.type_match(5))
    def check_type_match_float(self):
        s = c_spec.complex_converter()        
        assert(not s.type_match(5.))
    def check_type_match_complex(self):
        s = c_spec.complex_converter()        
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

    def check_complex_return(self):
        test_dir = setup_test_location()        
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

#----------------------------------------------------------------------------
# File conversion tests
#----------------------------------------------------------------------------

class test_file_converter(unittest.TestCase):    
    def check_py_to_file(self):
        import tempfile
        file_name = tempfile.mktemp()        
        file = open(file_name,'w')
        code = """
               fprintf(file,"hello bob");
               """
        inline_tools.inline(code,['file']) 
        file.close()
        file = open(file_name,'r')
        assert(file.read() == "hello bob")
    def check_file_to_py(self):
        import tempfile
        file_name = tempfile.mktemp()        
        # not sure I like Py::String as default -- might move to std::sting
        # or just plain char*
        code = """
               char* _file_name = (char*) file_name.c_str();
               FILE* file = fopen(_file_name,"w");
               return_val = file_to_py(file,_file_name,"w");
               Py_XINCREF(return_val);
               """
        file = inline_tools.inline(code,['file_name'])
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
    def check_call_function(self):
        import string
        func = string.find
        search_str = "hello world hello"
        sub_str = "world"
        # * Not sure about ref counts on search_str and sub_str.
        # * Is the Py::String necessary? (it works anyways...)
        code = """
               PWOTuple args(2);
               args.setItem(0,PWOString(search_str.c_str()));
               args.setItem(1,PWOString(sub_str.c_str()));
               return_val = PyObject_CallObject(func,args);
               """
        actual = inline_tools.inline(code,['func','search_str','sub_str'])
        desired = func(search_str,sub_str)        
        assert(desired == actual)


class test_sequence_converter(unittest.TestCase):    
    def check_convert_to_dict(self):
        d = {}
        inline_tools.inline("",['d']) 
    def check_convert_to_list(self):        
        l = []
        inline_tools.inline("",['l']) 
    def check_convert_to_string(self):        
        s = 'hello'
        inline_tools.inline("",['s']) 
    def check_convert_to_tuple(self):        
        t = ()
        inline_tools.inline("",['t']) 

class test_string_converter(unittest.TestCase):    
    def check_type_match_string(self):
        s = c_spec.string_converter()
        assert( s.type_match('string') )
    def check_type_match_int(self):
        s = c_spec.string_converter()        
        assert(not s.type_match(5))
    def check_type_match_float(self):
        s = c_spec.string_converter()        
        assert(not s.type_match(5.))
    def check_type_match_complex(self):
        s = c_spec.string_converter()        
        assert(not s.type_match(5.+1j))
    def check_var_in(self):
        mod = ext_tools.ext_module('string_var_in')
        a = 'string'
        code = 'a=std::string("hello");'
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile()
        import string_var_in
        b='bub'
        string_var_in.test(b)
        try:
            b = 1.
            string_var_in.test(b)
        except TypeError:
            pass
        try:
            b = 1
            string_var_in.test(b)
        except TypeError:
            pass
            
    def check_return(self):
        mod = ext_tools.ext_module('string_return')
        a = 'string'
        code = """
               a= std::string("hello");
               return_val = PyString_FromString(a.c_str());
               """
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile()
        import string_return
        b='bub'
        c = string_return.test(b)
        assert( c == 'hello')

class test_list_converter(unittest.TestCase):    
    def check_type_match_bad(self):
        s = c_spec.list_converter()
        objs = [{},(),'',1,1.,1+1j]
        for i in objs:
            assert( not s.type_match(i) )
    def check_type_match_good(self):
        s = c_spec.list_converter()        
        assert(s.type_match([]))
    def check_var_in(self):
        mod = ext_tools.ext_module('list_var_in')
        a = [1]
        code = 'a=PWOList();'
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile()
        import list_var_in
        b=[1,2]
        list_var_in.test(b)
        try:
            b = 1.
            list_var_in.test(b)
        except TypeError:
            pass
        try:
            b = 'string'
            list_var_in.test(b)
        except TypeError:
            pass
            
    def check_return(self):
        mod = ext_tools.ext_module('list_return')
        a = [1]
        code = """
               a=PWOList();
               a.append(PWOString("hello"));
               return_val = a.disOwn();
               """
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile()
        import list_return
        b=[1,2]
        c = list_return.test(b)
        assert( c == ['hello'])
        
    def check_speed(self):
        mod = ext_tools.ext_module('list_speed')
        a = range(1e6);
        code = """
               PWONumber v = PWONumber();
               int vv, sum = 0;            
               for(int i = 0; i < a.len(); i++)
               {
                   v = a[i];
                   vv = (int)v;
                   if (vv % 2)
                    sum += vv;
                   else
                    sum -= vv; 
               }
               return_val = PyInt_FromLong(sum);
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
               return_val = PyInt_FromLong(sum);
               """
        no_checking = ext_tools.ext_function('no_checking',code,['a'])
        mod.add_function(no_checking)
        mod.compile()
        import list_speed
        import time
        t1 = time.time()
        sum1 = list_speed.with_cxx(a)
        t2 = time.time()
        print 'scxx:',  t2 - t1
        t1 = time.time()
        sum2 = list_speed.no_checking(a)
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
    def check_type_match_bad(self):
        s = c_spec.tuple_converter()
        objs = [{},[],'',1,1.,1+1j]
        for i in objs:
            assert( not s.type_match(i) )
    def check_type_match_good(self):
        s = c_spec.tuple_converter()        
        assert(s.type_match((1,)))
    def check_var_in(self):
        mod = ext_tools.ext_module('tuple_var_in')
        a = (1,)
        code = 'a=PWOTuple();'
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile()
        import tuple_var_in
        b=(1,2)
        tuple_var_in.test(b)
        try:
            b = 1.
            tuple_var_in.test(b)
        except TypeError:
            pass
        try:
            b = 'string'
            tuple_var_in.test(b)
        except TypeError:
            pass
            
    def check_return(self):
        mod = ext_tools.ext_module('tuple_return')
        a = (1,)
        code = """
               a=PWOTuple(2);
               a.setItem(0,PWOString("hello"));
               a.setItem(1,PWOBase(Py_None));
               return_val = a.disOwn();
               """
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile()
        import tuple_return
        b=(1,2)
        c = tuple_return.test(b)
        assert( c == ('hello',None))


class test_dict_converter(unittest.TestCase):    
    def check_type_match_bad(self):
        s = c_spec.dict_converter()
        objs = [[],(),'',1,1.,1+1j]
        for i in objs:
            assert( not s.type_match(i) )
    def check_type_match_good(self):
        s = c_spec.dict_converter()        
        assert(s.type_match({}))
    def check_var_in(self):
        mod = ext_tools.ext_module('dict_var_in')
        a = {'z':1}
        code = 'a=PWODict();' # This just checks to make sure the type is correct
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile()
        import dict_var_in
        b={'y':2}
        dict_var_in.test(b)
        try:
            b = 1.
            dict_var_in.test(b)
        except TypeError:
            pass
        try:
            b = 'string'
            dict_var_in.test(b)
        except TypeError:
            pass
            
    def check_return(self):
        mod = ext_tools.ext_module('dict_return')
        a = {'z':1}
        code = """
               a=PWODict();
               a["hello"] = PWONumber(5);
               return_val = a.disOwn();
               """
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile()
        import dict_return
        b = {'z':2}
        c = dict_return.test(b)
        assert( c['hello'] == 5)
                  
def test_suite(level=1):
    suites = []    
    if level >= 5:
        """
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

        # file, instance, callable object tests
        suites.append( unittest.makeSuite(test_file_converter,'check_'))
        suites.append( unittest.makeSuite(test_instance_converter,'check_'))
        suites.append( unittest.makeSuite(test_callable_converter,'check_'))
        """
        # sequenc conversion tests
        suites.append( unittest.makeSuite(test_sequence_converter,'check_'))
        suites.append( unittest.makeSuite(test_string_converter,'check_'))
        suites.append( unittest.makeSuite(test_list_converter,'check_'))
        suites.append( unittest.makeSuite(test_tuple_converter,'check_'))
        suites.append( unittest.makeSuite(test_dict_converter,'check_'))
        
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test(level=10):
    all_tests = test_suite(level)
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
