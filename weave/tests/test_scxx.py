""" Test refcounting and behavior of SCXX.
"""
import unittest
import time
import os,sys
from scipy_distutils.misc_util import add_grandparent_to_path, restore_path

add_grandparent_to_path(__name__)
import inline_tools
restore_path()

# Test:
#     append            DONE
#     insert            DONE
#     in                DONE
#     count             DONE
#     setItem           DONE
#     operator[] (get)
#     operator[] (set)  DONE

class test_list(unittest.TestCase):
    def check_conversion(self):
        a = []
        before = sys.getrefcount(a)
        import weave
        weave.inline("",['a'])
        print 'first:',before
        # first call is goofing up refcount.
        before = sys.getrefcount(a)        
        weave.inline("",['a'])
        after = sys.getrefcount(a)        
        print '2nd,3rd:', before, after
        assert(after == before)

    def check_append_passed_item(self):
        a = []
        item = 1
        
        # temporary refcount fix until I understand why it incs by one.
        inline_tools.inline("a.append(item);",['a','item'])
        del a[0]                
        
        before1 = sys.getrefcount(a)
        before2 = sys.getrefcount(item)
        inline_tools.inline("a.append(item);",['a','item'])
        assert a[0] is item
        del a[0]                
        after1 = sys.getrefcount(a)
        after2 = sys.getrefcount(item)
        assert after1 == before1
        assert after2 == before2

    
    def check_append(self):
        a = []

        # temporary refcount fix until I understand why it incs by one.
        inline_tools.inline("a.append(1);",['a'])
        del a[0]                
        
        before1 = sys.getrefcount(a)
        
        # check overloaded append(int val) method
        inline_tools.inline("a.append(1234);",['a'])        
        assert sys.getrefcount(a[0]) == 2                
        assert a[0] == 1234
        del a[0]                

        # check overloaded append(double val) method
        inline_tools.inline("a.append(123.0);",['a'])
        assert sys.getrefcount(a[0]) == 2       
        assert a[0] == 123.0
        del a[0]                
        
        # check overloaded append(char* val) method        
        inline_tools.inline('a.append("bubba");',['a'])
        assert sys.getrefcount(a[0]) == 2       
        assert a[0] == 'bubba'
        del a[0]                
        
        # check overloaded append(std::string val) method
        inline_tools.inline('a.append(std::string("sissy"));',['a'])
        assert sys.getrefcount(a[0]) == 2       
        assert a[0] == 'sissy'
        del a[0]                
                
        after1 = sys.getrefcount(a)
        assert after1 == before1

    def check_insert(self):
        a = [1,2,3]
    
        a.insert(1,234)
        del a[1]
        
        # temporary refcount fix until I understand why it incs by one.
        inline_tools.inline("a.insert(1,1234);",['a'])
        del a[1]                
        
        before1 = sys.getrefcount(a)
        
        # check overloaded insert(int ndx, int val) method
        inline_tools.inline("a.insert(1,1234);",['a'])        
        assert sys.getrefcount(a[1]) == 2                
        assert a[1] == 1234
        del a[1]                

        # check overloaded insert(int ndx, double val) method
        inline_tools.inline("a.insert(1,123.0);",['a'])
        assert sys.getrefcount(a[1]) == 2       
        assert a[1] == 123.0
        del a[1]                
        
        # check overloaded insert(int ndx, char* val) method        
        inline_tools.inline('a.insert(1,"bubba");',['a'])
        assert sys.getrefcount(a[1]) == 2       
        assert a[1] == 'bubba'
        del a[1]                
        
        # check overloaded insert(int ndx, std::string val) method
        inline_tools.inline('a.insert(1,std::string("sissy"));',['a'])
        assert sys.getrefcount(a[1]) == 2       
        assert a[1] == 'sissy'
        del a[0]                
                
        after1 = sys.getrefcount(a)
        assert after1 == before1

    def check_set_item(self):
        a = [1,2,3]
            
        # temporary refcount fix until I understand why it incs by one.
        inline_tools.inline("a.set_item(1,1234);",['a'])
        
        before1 = sys.getrefcount(a)
        
        # check overloaded insert(int ndx, int val) method
        inline_tools.inline("a.set_item(1,1234);",['a'])        
        assert sys.getrefcount(a[1]) == 2                
        assert a[1] == 1234

        # check overloaded insert(int ndx, double val) method
        inline_tools.inline("a.set_item(1,123.0);",['a'])
        assert sys.getrefcount(a[1]) == 2       
        assert a[1] == 123.0
        
        # check overloaded insert(int ndx, char* val) method        
        inline_tools.inline('a.set_item(1,"bubba");',['a'])
        assert sys.getrefcount(a[1]) == 2       
        assert a[1] == 'bubba'
        
        # check overloaded insert(int ndx, std::string val) method
        inline_tools.inline('a.set_item(1,std::string("sissy"));',['a'])
        assert sys.getrefcount(a[1]) == 2       
        assert a[1] == 'sissy'
                
        after1 = sys.getrefcount(a)
        assert after1 == before1

    def check_set_item_operator_equal(self):
        a = [1,2,3]
            
        # temporary refcount fix until I understand why it incs by one.
        inline_tools.inline("a[1] = 1234;",['a'])
        
        before1 = sys.getrefcount(a)
        
        # check overloaded insert(int ndx, int val) method
        inline_tools.inline("a[1] = 1234;",['a'])        
        assert sys.getrefcount(a[1]) == 2                
        assert a[1] == 1234

        # check overloaded insert(int ndx, double val) method
        inline_tools.inline("a[1] = 123.0;",['a'])
        assert sys.getrefcount(a[1]) == 2       
        assert a[1] == 123.0
        
        # check overloaded insert(int ndx, char* val) method        
        inline_tools.inline('a[1] = "bubba";',['a'])
        assert sys.getrefcount(a[1]) == 2       
        assert a[1] == 'bubba'
        
        # check overloaded insert(int ndx, std::string val) method
        inline_tools.inline('a[1] = std::string("sissy");',['a'])
        assert sys.getrefcount(a[1]) == 2       
        assert a[1] == 'sissy'
                
        after1 = sys.getrefcount(a)
        assert after1 == before1

    def check_in(self):
        """ Test the "in" method for lists.  We'll assume
            it works for sequences if it works here.
        """
        a = [1,2,'alpha',3.1416]

        item = 1
        code = "return_val = PyInt_FromLong(a.in(item));"
        res = inline_tools.inline(code,['a','item'])
        assert res == 1
        item = 0
        res = inline_tools.inline(code,['a','item'])
        assert res == 0
        
        # check overloaded in(int val) method
        code = "return_val = PyInt_FromLong(a.in(1));"
        res = inline_tools.inline(code,['a'])
        assert res == 1
        code = "return_val = PyInt_FromLong(a.in(0));"
        res = inline_tools.inline(code,['a'])
        assert res == 0
        
        # check overloaded in(double val) method
        code = "return_val = PyInt_FromLong(a.in(3.1416));"
        res = inline_tools.inline(code,['a'])
        assert res == 1
        code = "return_val = PyInt_FromLong(a.in(3.1417));"
        res = inline_tools.inline(code,['a'])
        assert res == 0
        
        # check overloaded in(char* val) method        
        code = 'return_val = PyInt_FromLong(a.in("alpha"));'
        res = inline_tools.inline(code,['a'])
        assert res == 1
        code = 'return_val = PyInt_FromLong(a.in("beta"));'
        res = inline_tools.inline(code,['a'])
        assert res == 0
        
        # check overloaded in(std::string val) method
        code = 'return_val = PyInt_FromLong(a.in(std::string("alpha")));'
        res = inline_tools.inline(code,['a'])
        assert res == 1
        code = 'return_val = PyInt_FromLong(a.in(std::string("beta")));'
        res = inline_tools.inline(code,['a'])
        assert res == 0

    def check_count(self):
        """ Test the "count" method for lists.  We'll assume
            it works for sequences if it works hre.
        """
        a = [1,2,'alpha',3.1416]

        item = 1
        code = "return_val = PyInt_FromLong(a.count(item));"
        res = inline_tools.inline(code,['a','item'])
        assert res == 1
        
        # check overloaded count(int val) method
        code = "return_val = PyInt_FromLong(a.count(1));"
        res = inline_tools.inline(code,['a'])
        assert res == 1
        
        # check overloaded count(double val) method
        code = "return_val = PyInt_FromLong(a.count(3.1416));"
        res = inline_tools.inline(code,['a'])
        assert res == 1
        
        # check overloaded count(char* val) method        
        code = 'return_val = PyInt_FromLong(a.count("alpha"));'
        res = inline_tools.inline(code,['a'])
        assert res == 1
        
        # check overloaded count(std::string val) method
        code = 'return_val = PyInt_FromLong(a.count(std::string("alpha")));'
        res = inline_tools.inline(code,['a'])
        assert res == 1

    def check_access_speed(self):
        N = 1000000
        print 'list access -- val = a[i] for N =', N
        a = [0] * N
        val = 0
        t1 = time.time()
        for i in xrange(N):
            val = a[i]
        t2 = time.time()
        print 'python1:', t2 - t1
        t1 = time.time()
        for i in a:
            val = i
        t2 = time.time()
        print 'python2:', t2 - t1
        
        code = """
               const int N = a.length();
               py::object val;
               for(int i=0; i < N; i++)
                   val = a[i];
               """
        # compile not included in timing       
        inline_tools.inline(code,['a'])           
        t1 = time.time()
        inline_tools.inline(code,['a'])           
        t2 = time.time()
        print 'weave:', t2 - t1

    def check_access_set_speed(self):
        N = 1000000
        print 'list access/set -- b[i] = a[i] for N =', N        
        a = [0] * N
        b = [1] * N
        t1 = time.time()
        for i in xrange(N):
            b[i] = a[i]
        t2 = time.time()
        print 'python:', t2 - t1
        
        a = [0] * N
        b = [1] * N     
        code = """
               const int N = a.length();
               for(int i=0; i < N; i++)
                   b[i] = a[i];       
               """
        # compile not included in timing
        inline_tools.inline(code,['a','b'])           
        t1 = time.time()
        inline_tools.inline(code,['a','b'])           
        t2 = time.time()
        print 'weave:', t2 - t1
        assert b == a   

    def check_string_add_speed(self):
        N = 1000000
        print 'string add -- b[i] = a[i] + "blah" for N =', N        
        a = ["blah"] * N
        desired = [1] * N
        t1 = time.time()
        for i in xrange(N):
            desired[i] = a[i] + 'blah'
        t2 = time.time()
        print 'python:', t2 - t1
        
        a = ["blah"] * N
        b = [1] * N     
        code = """
               const int N = a.length();
               std::string blah = std::string("blah");
               for(int i=0; i < N; i++)
                   b[i] = (std::string)a[i] + blah;       
               """
        # compile not included in timing
        inline_tools.inline(code,['a','b'])           
        t1 = time.time()
        inline_tools.inline(code,['a','b'])           
        t2 = time.time()
        print 'weave:', t2 - t1
        assert b == desired   

    def check_int_add_speed(self):
        N = 1000000
        print 'int add -- b[i] = a[i] + 1 for N =', N        
        a = [0] * N
        desired = [1] * N
        t1 = time.time()
        for i in xrange(N):
            desired[i] = a[i] + 1
        t2 = time.time()
        print 'python:', t2 - t1
        
        a = [0] * N
        b = [0] * N     
        code = """
               const int N = a.length();
               for(int i=0; i < N; i++)
                   b[i] = (int)a[i] + 1;       
               """
        # compile not included in timing
        inline_tools.inline(code,['a','b'])           
        t1 = time.time()
        inline_tools.inline(code,['a','b'])           
        t2 = time.time()
        print 'weave:', t2 - t1
        assert b == desired   

class test_object_construct(unittest.TestCase):
    #------------------------------------------------------------------------
    # Check that construction from basic types is allowed and have correct
    # reference counts
    #------------------------------------------------------------------------
    def check_int(self):
        # strange int value used to try and make sure refcount is 2.
        code = """
               py::object val = 1001;
               return_val = val;
               """
        res = inline_tools.inline(code)
        assert sys.getrefcount(res) == 2
        assert res == 1001
    def check_float(self):
        code = """
               py::object val = (float)1.0;
               return_val = val;
               """
        res = inline_tools.inline(code)
        assert sys.getrefcount(res) == 2
        assert res == 1.0
    def check_double(self):
        code = """
               py::object val = 1.0;
               return_val = val;
               """
        res = inline_tools.inline(code)
        assert sys.getrefcount(res) == 2
        assert res == 1.0
    def check_complex(self):
        code = """
               std::complex<double> num = std::complex<double>(1.0,1.0);
               py::object val = num;
               return_val = val;
               """
        res = inline_tools.inline(code)
        assert sys.getrefcount(res) == 2
        assert res == 1.0+1.0j
    def check_string(self):
        code = """
               py::object val = "hello";
               return_val = val;
               """
        res = inline_tools.inline(code)
        assert sys.getrefcount(res) == 2
        assert res == "hello"

    def check_std_string(self):
        code = """
               std::string s = std::string("hello");
               py::object val = s;
               return_val = val;
               """
        res = inline_tools.inline(code)
        assert sys.getrefcount(res) == 2
        assert res == "hello"
            
            
class test_object_cast(unittest.TestCase):
    def check_int_cast(self):
        code = """
               py::object val = 1;
               int raw_val = val;
               """
        inline_tools.inline(code)
    def check_double_cast(self):
        code = """
               py::object val = 1.0;
               double raw_val = val;
               """
        inline_tools.inline(code)
    def check_float_cast(self):
        code = """
               py::object val = 1.0;
               float raw_val = val;
               """
        inline_tools.inline(code)
    def check_complex_cast(self):
        code = """
               std::complex<double> num = std::complex<double>(1.0,1.0);
               py::object val = num;
               std::complex<double> raw_val = val;
               """
        inline_tools.inline(code)
    def check_string_cast(self):
        code = """
               py::object val = "hello";
               std::string raw_val = val;
               """
        inline_tools.inline(code)
                    
# test class used for testing python class access from C++.
class foo:
    def bar(self):
        return "bar results"
    def bar2(self,val1,val2):
        return val1, val2
    def bar3(self,val1,val2,val3=1):
        return val1, val2, val3

class str_obj:
            def __str__(self):
                return "b"

class test_object_hasattr(unittest.TestCase):
    def check_string(self):
        a = foo()
        a.b = 12345
        code = """
               return_val = a.hasattr("b");               
               """
        res = inline_tools.inline(code,['a'])
        assert res
    def check_std_string(self):
        a = foo()
        a.b = 12345
        attr_name = "b"
        code = """
               return_val = a.hasattr(attr_name);               
               """
        res = inline_tools.inline(code,['a','attr_name'])
        assert res        
    def check_string_fail(self):
        a = foo()
        a.b = 12345
        code = """
               return_val = a.hasattr("c");               
               """
        res = inline_tools.inline(code,['a'])
        assert not res
    def check_inline(self):
        """ THIS NEEDS TO MOVE TO THE INLINE TEST SUITE
        """
        a = foo()
        a.b = 12345
        code = """
               throw_error(PyExc_AttributeError,"bummer");               
               """
        try:
            before = sys.getrefcount(a)
            res = inline_tools.inline(code,['a'])
        except AttributeError:
            after = sys.getrefcount(a)
            try: 
                res = inline_tools.inline(code,['a'])
            except:
                after2 = sys.getrefcount(a)
            print "after and after2 should be equal in the following"        
            print 'before, after, after2:', before, after, after2
            pass    

    def check_func(self):
        a = foo()
        a.b = 12345
        code = """
               return_val = a.hasattr("bar");               
               """
        res = inline_tools.inline(code,['a'])
        assert res

class test_object_attr(unittest.TestCase):

    def generic_attr(self,code,args=['a']):
        a = foo()
        a.b = 12345
                
        before = sys.getrefcount(a.b)
        res = inline_tools.inline(code,args)
        assert res == a.b
        del res
        after = sys.getrefcount(a.b)
        assert after == before

    def check_char(self):
        self.generic_attr('return_val = a.attr("b");')

    def check_string(self):
        self.generic_attr('return_val = a.attr(std::string("b"));')

    def check_obj(self):
        code = """
               py::str name = py::str("b");
               return_val = a.attr(name);
               """ 
        self.generic_attr(code,['a'])
    def check_attr_call(self):
        a = foo()
        res = inline_tools.inline('return_val = a.attr("bar").call();',['a'])
        first = sys.getrefcount(res)
        del res
        res = inline_tools.inline('return_val = a.attr("bar").call();',['a'])
        second = sys.getrefcount(res)
        assert res == "bar results"
        assert first == second

class test_object_mcall(unittest.TestCase):
    def check_noargs(self):
        a = foo()
        res = inline_tools.inline('return_val = a.mcall("bar");',['a'])
        assert res == "bar results"
        first = sys.getrefcount(res)
        del res
        res = inline_tools.inline('return_val = a.mcall("bar");',['a'])
        assert res == "bar results"
        second = sys.getrefcount(res)
        assert first == second
    def check_args(self):
        a = foo()
        code = """
               py::tuple args(2);
               args[0] = 1;
               args[1] = "hello";
               return_val = a.mcall("bar2",args);
               """
        res = inline_tools.inline(code,['a'])
        assert res == (1,"hello")
        assert sys.getrefcount(res) == 2
    def check_args_kw(self):
        a = foo()
        code = """
               py::tuple args(2);
               args[0] = 1;
               args[1] = "hello";
               py::dict kw;
               kw["val3"] = 3;
               return_val = a.mcall("bar3",args,kw);
               """
        res = inline_tools.inline(code,['a'])
        assert res == (1,"hello",3)
        assert sys.getrefcount(res) == 2
    def check_std_noargs(self):
        a = foo()
        method = "bar"
        res = inline_tools.inline('return_val = a.mcall(method);',['a','method'])
        assert res == "bar results"
        first = sys.getrefcount(res)
        del res
        res = inline_tools.inline('return_val = a.mcall(method);',['a','method'])
        assert res == "bar results"
        second = sys.getrefcount(res)
        assert first == second
    def check_std_args(self):
        a = foo()
        method = "bar2"
        code = """
               py::tuple args(2);
               args[0] = 1;
               args[1] = "hello";
               return_val = a.mcall(method,args);
               """
        res = inline_tools.inline(code,['a','method'])
        assert res == (1,"hello")
        assert sys.getrefcount(res) == 2
    def check_std_args_kw(self):
        a = foo()
        method = "bar3"
        code = """
               py::tuple args(2);
               args[0] = 1;
               args[1] = "hello";
               py::dict kw;
               kw["val3"] = 3;
               return_val = a.mcall(method,args,kw);
               """
        res = inline_tools.inline(code,['a','method'])
        assert res == (1,"hello",3)
        assert sys.getrefcount(res) == 2
    def check_noargs_with_args(self):
        # calling a function that does take args with args 
        # should fail.
        a = foo()
        code = """
               py::tuple args(2);
               args[0] = 1;
               args[1] = "hello";
               return_val = a.mcall("bar",args);
               """
        try:
            first = sys.getrefcount(a)
            res = inline_tools.inline(code,['a'])
        except TypeError:
            second = sys.getrefcount(a) 
            try:
                res = inline_tools.inline(code,['a'])
            except TypeError:
                third = sys.getrefcount(a)    
        # first should == second, but the weird refcount error        
        assert second == third

class test_object_call(unittest.TestCase):
    def check_noargs(self):
        def foo():
            return (1,2,3)
        res = inline_tools.inline('return_val = foo.call();',['foo'])
        assert res == (1,2,3)
        assert sys.getrefcount(res) == 2
    def check_args(self):
        def foo(val1,val2):
            return (val1,val2)
        code = """
               py::tuple args(2);
               args[0] = 1;
               args[1] = "hello";
               return_val = foo.call(args);
               """
        res = inline_tools.inline(code,['foo'])
        assert res == (1,"hello")
        assert sys.getrefcount(res) == 2
    def check_args_kw(self):
        def foo(val1,val2,val3=1):
            return (val1,val2,val3)
        code = """
               py::tuple args(2);
               args[0] = 1;
               args[1] = "hello";
               py::dict kw;
               kw["val3"] = 3;               
               return_val = foo.call(args,kw);
               """
        res = inline_tools.inline(code,['foo'])
        assert res == (1,"hello",3)
        assert sys.getrefcount(res) == 2
    def check_noargs_with_args(self):
        # calling a function that does take args with args 
        # should fail.
        def foo():
            return "blah"
        code = """
               py::tuple args(2);
               args[0] = 1;
               args[1] = "hello";
               return_val = foo.call(args);
               """
        try:
            first = sys.getrefcount(foo)
            res = inline_tools.inline(code,['foo'])
        except TypeError:
            second = sys.getrefcount(foo) 
            try:
                res = inline_tools.inline(code,['foo'])
            except TypeError:
                third = sys.getrefcount(foo)    
        # first should == second, but the weird refcount error        
        assert second == third
                
def test_suite(level=1):
    from unittest import makeSuite
    suites = []    
    if level >= 5:
        #suites.append( makeSuite(test_list,'check_'))
        
        #suites.append( makeSuite(test_object_construct,'check_'))
        #suites.append( makeSuite(test_object_cast,'check_'))
        #suites.append( makeSuite(test_object_hasattr,'check_'))        
        #suites.append( makeSuite(test_object_attr,'check_'))
        suites.append( makeSuite(test_object_mcall,'check_'))
        suites.append( makeSuite(test_object_call,'check_'))
        

    total_suite = unittest.TestSuite(suites)
    return total_suite

def test(level=10):
    all_tests = test_suite(level)
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
