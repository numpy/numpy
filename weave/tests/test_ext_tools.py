import unittest
import time

from scipy_distutils.misc_util import add_grandparent_to_path, restore_path
from scipy_distutils.misc_util import add_local_to_path

add_grandparent_to_path(__name__)
import ext_tools
try:
    from standard_array_spec import array_specification
except ImportError:
    pass # requires Numeric    
restore_path()

add_local_to_path(__name__)
from compiler_test_utils import *
restore_path()

build_dir = empty_temp_dir()
print 'building extensions here:', build_dir    

class test_ext_module(unittest.TestCase):
    #should really do some testing of where modules end up
    def check_simple(self):
        """ Simplest possible module """
        mod = ext_tools.ext_module('simple_ext_module')
        mod.compile(location = build_dir)
        import simple_ext_module
    def check_multi_functions(self):
        mod = ext_tools.ext_module('module_multi_function')
        var_specs = []
        code = ""
        test = ext_tools.ext_function_from_specs('test',code,var_specs)
        mod.add_function(test)
        test2 = ext_tools.ext_function_from_specs('test2',code,var_specs)
        mod.add_function(test2)
        mod.compile(location = build_dir)
        import module_multi_function
        module_multi_function.test()
        module_multi_function.test2()
    def check_with_include(self):
        # decalaring variables
        a = 2.;
    
        # declare module
        mod = ext_tools.ext_module('ext_module_with_include')
        mod.customize.add_header('<iostream>')
    
        # function 2 --> a little more complex expression
        var_specs = ext_tools.assign_variable_types(['a'],locals(),globals())
        code = """
               std::cout << "a:" << a << std::endl;
               """
        test = ext_tools.ext_function_from_specs('test',code,var_specs)
        mod.add_function(test)
        # build module
        mod.compile(location = build_dir)
        import ext_module_with_include
        ext_module_with_include.test(a)

    def check_string_and_int(self):        
        # decalaring variables
        a = 2;b = 'string'    
        # declare module
        mod = ext_tools.ext_module('ext_string_and_int')
        code = """
               a=b.length();
               return_val = Py::new_reference_to(Py::Int(a));
               """
        test = ext_tools.ext_function('test',code,['a','b'])
        mod.add_function(test)
        mod.compile(location = build_dir)
        import ext_string_and_int
        c = ext_string_and_int.test(a,b)
        assert(c == len(b))
        
    def check_return_tuple(self):        
        # decalaring variables
        a = 2    
        # declare module
        mod = ext_tools.ext_module('ext_return_tuple')
        var_specs = ext_tools.assign_variable_types(['a'],locals())
        code = """
               int b;
               b = a + 1;
               Py::Tuple returned(2);
               returned[0] = Py::Int(a);
               returned[1] = Py::Int(b);
               return_val = Py::new_reference_to(returned);
               """
        test = ext_tools.ext_function('test',code,['a'])
        mod.add_function(test)
        mod.compile(location = build_dir)
        import ext_return_tuple
        c,d = ext_return_tuple.test(a)
        assert(c==a and d == a+1)
           
class test_ext_function(unittest.TestCase):
    #should really do some testing of where modules end up
    def check_simple(self):
        """ Simplest possible function """
        mod = ext_tools.ext_module('simple_ext_function')
        var_specs = []
        code = ""
        test = ext_tools.ext_function_from_specs('test',code,var_specs)
        mod.add_function(test)
        mod.compile(location = build_dir)
        import simple_ext_function
        simple_ext_function.test()
      
class test_assign_variable_types(unittest.TestCase):            
    def check_assign_variable_types(self):
        try:
            from Numeric import *
        except:
            # skip this test if Numeric not installed
            return
            
        import types
        a = arange(10,typecode = Float32)
        b = arange(5,typecode = Float64)
        c = 5
        arg_list = ['a','b','c']
        actual = ext_tools.assign_variable_types(arg_list,locals())        
        #desired = {'a':(Float32,1),'b':(Float32,1),'i':(Int32,0)}
        
        ad = array_specification()
        ad.name, ad.numeric_type, ad.dims = 'a', Float32, 1
        bd = array_specification()
        bd.name, bd.numeric_type, bd.dims = 'b', Float64, 1
        import scalar_spec
        cd = scalar_spec.int_specification()
        cd.name, cd.numeric_type = 'c', types.IntType        
        desired = [ad,bd,cd]
        expr = ""
        print_assert_equal(expr,actual,desired)


def test_suite():
    suites = []
    suites.append( unittest.makeSuite(test_assign_variable_types,'check_') )
    suites.append( unittest.makeSuite(test_ext_module,'check_'))
    suites.append( unittest.makeSuite(test_ext_function,'check_'))      
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test():
    all_tests = test_suite()
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
