import unittest

from scipy_distutils.misc_util import add_grandparent_to_path, restore_path

add_grandparent_to_path(__name__)
import inline_tools
restore_path()
    
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
               char* _file_name = PyString_AsString( file_name.ptr() );
               FILE* file = fopen(_file_name,"w");
               Py::Object file_obj(file_to_py(file,_file_name,"w"));
               return_val = Py::new_reference_to(file_obj);
               """
        file = inline_tools.inline(code,['file_name'])
        file.write("hello fred")        
        file.close()
        file = open(file_name,'r')
        assert(file.read() == "hello fred")

class test_instance_converter(unittest.TestCase):    
    pass
    
class test_callable_converter(unittest.TestCase):        
    def check_call_function(self):
        import string
        func = string.find
        search_str = "hello world hello"
        sub_str = "world"
        # * Not sure about ref counts on search_str and sub_str.
        # * Is the Py::String necessary? (it works anyways...)
        code = """
               Py::Tuple args(2);
               args[0] = Py::String(search_str);
               args[1] = Py::String(sub_str);
               PyObject* result = PyObject_CallObject(func,args.ptr());
               return_val = Py::new_reference_to(Py::Int(result));
               """
        actual = inline_tools.inline(code,['func','search_str','sub_str'])
        desired = func(search_str,sub_str)        
        assert(desired == actual)

def test_suite(level=1):
    suites = []
    if level >= 5:   
        suites.append( unittest.makeSuite(test_file_converter,'check_'))
        suites.append( unittest.makeSuite(test_instance_converter,'check_'))
        suites.append( unittest.makeSuite(test_callable_converter,'check_'))
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test(level=10):
    all_tests = test_suite(level)
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
