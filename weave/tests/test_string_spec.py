"""
check_var_in -- tests whether a variable is passed in correctly
                and also if the passed in variable can be reassigned
check_var_local -- tests wheter a variable is passed in , modified,
                   and returned correctly in the local_dict dictionary
                   argument
check_return -- test whether a variable is passed in, modified, and
                then returned as a function return value correctly                                 
"""
import unittest
import time

import sys
sys.path.append('..')
import ext_tools

class test_string_converter(unittest.TestCase):    
    def check_type_match_string(self):
        s = ext_tools.string_converter()
        assert( s.type_match('string') )
    def check_type_match_int(self):
        s = ext_tools.string_converter()        
        assert(not s.type_match(5))
    def check_type_match_float(self):
        s = ext_tools.string_converter()        
        assert(not s.type_match(5.))
    def check_type_match_complex(self):
        s = ext_tools.string_converter()        
        assert(not s.type_match(5.+1j))
    def check_var_in(self):
        mod = ext_tools.ext_module('string_var_in')
        a = 'string'
        var_specs = ext_tools.assign_variable_types(['a'],locals())
        code = 'a=Py::String("hello");'
        test = ext_tools.ext_function('test',var_specs,code)
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
            
    def check_var_local(self):
        mod = ext_tools.ext_module('string_var_local')
        a = 'string'
        var_specs = ext_tools.assign_variable_types(['a'],locals())
        code = 'a=Py::String("hello");'
        test = ext_tools.ext_function('test',var_specs,code)
        mod.add_function(test)
        mod.compile()
        import string_var_local
        b='bub'
        q={}
        string_var_local.test(b,q)
        assert(q['a'] == 'hello')
    def check_return(self):
        mod = ext_tools.ext_module('string_return')
        a = 'string'
        var_specs = ext_tools.assign_variable_types(['a'],locals())
        code = """
               a= Py::String("hello");
               return_val = Py::new_reference_to(a);
               """
        test = ext_tools.ext_function('test',var_specs,code)
        mod.add_function(test)
        mod.compile()
        import string_return
        b='bub'
        c = string_return.test(b)
        assert( c == 'hello')

class test_list_converter(unittest.TestCase):    
    def check_type_match_bad(self):
        s = ext_tools.list_converter()
        objs = [{},(),'',1,1.,1+1j]
        for i in objs:
            assert( not s.type_match(i) )
    def check_type_match_good(self):
        s = ext_tools.list_converter()        
        assert(s.type_match([]))
    def check_var_in(self):
        mod = ext_tools.ext_module('list_var_in')
        a = [1]
        var_specs = ext_tools.assign_variable_types(['a'],locals())
        code = 'a=Py::List();'
        test = ext_tools.ext_function('test',var_specs,code)
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
            
    def check_var_local(self):
        mod = ext_tools.ext_module('list_var_local')
        a = []
        var_specs = ext_tools.assign_variable_types(['a'],locals())
        code = """
               a=Py::List();
               a.append(Py::String("hello"));
               """
        test = ext_tools.ext_function('test',var_specs,code)
        mod.add_function(test)
        mod.compile()
        import list_var_local
        a=[1,2]
        q={}
        list_var_local.test(a,q)
        assert(q['a'] == ['hello'])
    def check_return(self):
        mod = ext_tools.ext_module('list_return')
        a = [1]
        var_specs = ext_tools.assign_variable_types(['a'],locals())
        code = """
               a=Py::List();
               a.append(Py::String("hello"));
               return_val = Py::new_reference_to(a);
               """
        test = ext_tools.ext_function('test',var_specs,code)
        mod.add_function(test)
        mod.compile()
        import list_return
        b=[1,2]
        c = list_return.test(b)
        assert( c == ['hello'])
        
    def check_speed(self):
        mod = ext_tools.ext_module('list_speed')
        a = range(1e6);b=1 # b to force availability of py_to_scalar<int>
        var_specs = ext_tools.assign_variable_types(['a','b'],locals())
        code = """
               Py::Int v = Py::Int();
               int vv, sum = 0;            
               for(int i = 0; i < a.length(); i++)
               {
                   v = a[i];
                   vv = (int)v;
                   if (vv % 2)
                    sum += vv;
                   else
                    sum -= vv; 
               }
               return_val = Py::new_reference_to(Py::Int(sum));
               """
        with_cxx = ext_tools.ext_function('with_cxx',var_specs,code)
        mod.add_function(with_cxx)
        code = """
               int vv, sum = 0;
               PyObject *a_ptr = a.ptr(), *v;               
               for(int i = 0; i < a.length(); i++)
               {
                   v = PyList_GetItem(a_ptr,i);
                   //didn't set error here -- just speed test
                   vv = py_to_scalar<int>(v,"list item");
                   if (vv % 2)
                    sum += vv;
                   else
                    sum -= vv; 
               }
               return_val = Py::new_reference_to(Py::Int(sum));
               """
        no_checking = ext_tools.ext_function('no_checking',var_specs,code)
        mod.add_function(no_checking)
        mod.compile()
        import list_speed
        import time
        t1 = time.time()
        sum1 = list_speed.with_cxx(a,b)
        t2 = time.time()
        print 'cxx:',  t2 - t1
        t1 = time.time()
        sum2 = list_speed.no_checking(a,b)
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
        s = ext_tools.tuple_converter()
        objs = [{},[],'',1,1.,1+1j]
        for i in objs:
            assert( not s.type_match(i) )
    def check_type_match_good(self):
        s = ext_tools.tuple_converter()        
        assert(s.type_match((1,)))
    def check_var_in(self):
        mod = ext_tools.ext_module('tuple_var_in')
        a = (1,)
        var_specs = ext_tools.assign_variable_types(['a'],locals())
        code = 'a=Py::Tuple();'
        test = ext_tools.ext_function('test',var_specs,code)
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
            
    def check_var_local(self):
        mod = ext_tools.ext_module('tuple_var_local')
        a = (1,)
        var_specs = ext_tools.assign_variable_types(['a'],locals())
        code = """
               a=Py::Tuple(2);
               a[0] = Py::String("hello");
               """
        test = ext_tools.ext_function('test',var_specs,code)
        mod.add_function(test)
        mod.compile()
        import tuple_var_local
        a=(1,2)
        q={}
        tuple_var_local.test(a,q)
        assert(q['a'] == ('hello',None))
    def check_return(self):
        mod = ext_tools.ext_module('tuple_return')
        a = (1,)
        var_specs = ext_tools.assign_variable_types(['a'],locals())
        code = """
               a=Py::Tuple(2);
               a[0] = Py::String("hello");
               return_val = Py::new_reference_to(a);
               """
        test = ext_tools.ext_function('test',var_specs,code)
        mod.add_function(test)
        mod.compile()
        import tuple_return
        b=(1,2)
        c = tuple_return.test(b)
        assert( c == ('hello',None))


class test_dict_converter(unittest.TestCase):    
    def check_type_match_bad(self):
        s = ext_tools.dict_converter()
        objs = [[],(),'',1,1.,1+1j]
        for i in objs:
            assert( not s.type_match(i) )
    def check_type_match_good(self):
        s = ext_tools.dict_converter()        
        assert(s.type_match({}))
    def check_var_in(self):
        mod = ext_tools.ext_module('dict_var_in')
        a = {'z':1}
        var_specs = ext_tools.assign_variable_types(['a'],locals())
        code = 'a=Py::Dict();' # This just checks to make sure the type is correct
        test = ext_tools.ext_function('test',var_specs,code)
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
            
    def check_var_local(self):
        mod = ext_tools.ext_module('dict_var_local')
        a = {'z':1}
        var_specs = ext_tools.assign_variable_types(['a'],locals())
        code = """
               a=Py::Dict();
               a[Py::String("hello")] = Py::Int(5);
               """
        test = ext_tools.ext_function('test',var_specs,code)
        mod.add_function(test)
        mod.compile()
        import dict_var_local
        a = {'z':2}
        q={}
        dict_var_local.test(a,q)
        assert(q['a']['hello'] == 5)
    def check_return(self):
        mod = ext_tools.ext_module('dict_return')
        a = {'z':1}
        var_specs = ext_tools.assign_variable_types(['a'],locals())
        code = """
               a=Py::Dict();
               a[Py::String("hello")] = Py::Int(5);
               return_val = Py::new_reference_to(a);
               """
        test = ext_tools.ext_function('test',var_specs,code)
        mod.add_function(test)
        mod.compile()
        import dict_return
        b = {'z':2}
        c = dict_return.test(b)
        assert( c['hello'] == 5)

def test_suite(level=1):
    suites = []
    if level >= 5:    
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
