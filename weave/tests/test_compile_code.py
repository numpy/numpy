import unittest
from Numeric import *
from fastumath import *
import RandomArray
import time

#from scipy.compiler.ast_tools import *
#import scipy.compiler.compile_code
#compile_code = scipy.compiler.compile_code
from scipy.compiler.ast_tools import *
import scipy.compiler.compile_code as compile_code
import scipy.compiler

def remove_whitespace(in_str):
    import string
    out = string.replace(in_str," ","")
    out = string.replace(out,"\t","")
    out = string.replace(out,"\n","")
    return out
    
def print_assert_equal(test_string,actual,desired):
    """this should probably be in scipy.scipy_test
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

class test_ast_to_blitz_expr(unittest.TestCase):

    def generic_test(self,expr,desired):
        import parser
        ast = parser.suite(expr)
        ast_list = ast.tolist()
        actual = compile_code.ast_to_blitz_expr(ast_list)
        actual = remove_whitespace(actual)
        desired = remove_whitespace(desired)
        print_assert_equal(expr,actual,desired)

    def check_simple_expr(self):
        """convert simple expr to blitz
           
           a[:1:2] = b[:1+i+2:]
        """
        expr = "a[:1:2] = b[:1+i+2:]"        
        desired = "a(blitz::Range(_beg,1-1,2))="\
                  "b(blitz::Range(_beg,1+i+2-1));"
        self.generic_test(expr,desired)

    def check_fdtd_expr(self):
        """ convert fdtd equation to blitz.
             ex[:,1:,1:] =   ca_x[:,1:,1:] * ex[:,1:,1:] 
                           + cb_y_x[:,1:,1:] * (hz[:,1:,1:] - hz[:,:-1,:])
                           - cb_z_x[:,1:,1:] * (hy[:,1:,1:] - hy[:,1:,:-1]);
             Note:  This really should have "\" at the end of each line
                    to indicate continuation.  
        """
        expr = "ex[:,1:,1:] =   ca_x[:,1:,1:] * ex[:,1:,1:]" \
                             "+ cb_y_x[:,1:,1:] * (hz[:,1:,1:] - hz[:,:-1,:])"\
                             "- cb_z_x[:,1:,1:] * (hy[:,1:,1:] - hy[:,1:,:-1])"        
        desired = 'ex(_all,blitz::Range(1,_end),blitz::Range(1,_end))='\
                  '  ca_x(_all,blitz::Range(1,_end),blitz::Range(1,_end))'\
                  ' *ex(_all,blitz::Range(1,_end),blitz::Range(1,_end))'\
                  '+cb_y_x(_all,blitz::Range(1,_end),blitz::Range(1,_end))'\
                  '*(hz(_all,blitz::Range(1,_end),blitz::Range(1,_end))'\
                  '  -hz(_all,blitz::Range(_beg,_Nhz(1)-1-1),_all))'\
                  ' -cb_z_x(_all,blitz::Range(1,_end),blitz::Range(1,_end))'\
                  '*(hy(_all,blitz::Range(1,_end),blitz::Range(1,_end))'\
                  '-hy(_all,blitz::Range(1,_end),blitz::Range(_beg,_Nhy(2)-1-1)));'
        self.generic_test(expr,desired)

class test_harvest_variables(unittest.TestCase):
    def generic_test(self,expr,desired):
        import parser
        ast_list = parser.suite(expr).tolist()
        actual = compile_code.harvest_variables(ast_list)
        print_assert_equal(expr,actual,desired)

    def check_simple_expr(self):
        """convert simple expr to blitz
           
           a[:1:2] = b[:1+i+2:]
        """
        expr = "a[:1:2] = b[:1+i+2:]"        
        desired = ['a','b','i']        
        self.generic_test(expr,desired)

class test_assign_variable_types(unittest.TestCase):            
    def check_assign_variable_types(self):
        a = arange(10,typecode = Float32)
        b = arange(5,typecode = Float32)
        i = 5
        expr = "a[:1:2] = b[:1+i+2:]"     
        ast_list = parser.suite(expr).tolist()
        actual = compile_code.assign_variable_types(ast_list,locals())        
        desired = {'a':(Float32,1),'b':(Float32,1),'i':(Int32,0)}
        print_assert_equal(expr,actual,desired)

class test_blitz_array_declaration(unittest.TestCase):            
    def generic_test(self,comment,vals,desired):
        var = vals['var']; specs = vals['specs']; arg = vals['arg']
        actual=compile_code.blitz_array_declaration(var,specs,arg)
        actual = remove_whitespace(actual)
        desired = remove_whitespace(desired)
        print_assert_equal(comment,actual,desired)
        
    def check_float1_case(self):
        "convert 2nd arg -- a = (Float32,1)"
        comment = "convert 2nd arg a = (Float32,1)"
        settings = {'var':'a','specs':(Float32,1), 'arg':1}
        desired = '//compile_code.blitz_array_declaration'\
                  'blitz::Array<float,1> a = py_to_blitz<float,1>(py_a,"a");'\
                  'blitz::TinyVector<int,1>_Na=a.shape();'\
                  'clean_up[1]=py_a;'
        self.generic_test(comment,settings,desired)

    def check_float2_case(self):
        "convert 3rd arg -- a = (float64,2)"
        comment = "convert 3rd arg -- a = (float64,2)"
        settings = {'var':'bob','specs':(Float64,2), 'arg':2}
        desired = '//compile_code.blitz_array_declaration'\
                  'blitz::Array<double,2> bob = py_to_blitz<double,2>(py_bob,"bob");'\
                  'blitz::TinyVector<int,2>_Nbob=bob.shape();'\
                  'clean_up[2] = py_bob;'
        self.generic_test(comment,settings,desired)

    def check_complex5_case(self):
        "convert 1st arg -- a = (Complex64,5)"
        comment = "convert 1st arg -- a = (Complex64,5)"
        settings = {'var':'a','specs':(Complex64,5), 'arg':0}
        desired = '//compile_code.blitz_array_declaration'\
                  'blitz::Array<std::complex<double>,5> a = '\
                  'py_to_blitz<std::complex<double>,5>(py_a,"a");'\
                  'blitz::TinyVector<int,5>_Na=a.shape();'\
                  'clean_up[0] = py_a;'
        self.generic_test(comment,settings,desired)

class test_blitz_scalar_declaration(unittest.TestCase):            
    def generic_test(self,comment,vals,desired):
        var = vals['var']; specs = vals['specs']; arg = vals['arg']
        actual=compile_code.blitz_scalar_declaration(var,specs,arg)
        actual = remove_whitespace(actual)
        desired = remove_whitespace(desired)
        print_assert_equal(comment,actual,desired)
        
    def check_float_case(self):
        "convert 'a' to scalar float"
        comment = "convert 'a' to scalar float"
        settings = {'var':'a','specs':(Float32,0), 'arg':1}
        desired = 'float a = py_to_scalar<float>(py_a,"a");'
        self.generic_test(comment,settings,desired)

    def check_complex_case(self):
        "convert 'a' to scalar complex<double>"
        comment = "convert 'a' to scalar complex<double>"
        settings = {'var':'a','specs':(Complex64,0), 'arg':1}
        desired = 'inta=py_to_scalar<int>(py_a,"a");'
        desired = 'std::complex<double>a=py_to_scalar<std::complex<double>>(py_a,"a");'
        self.generic_test(comment,settings,desired)

    def check_int_case(self):
        "convert 'a' to scalar int"
        comment = "convert 'a' to scalar int"
        settings = {'var':'a','specs':(Int32,0), 'arg':1}
        desired = 'int a = py_to_scalar<int>(py_a,"a");'
        self.generic_test(comment,settings,desired)

class test_parse_tuple_block(unittest.TestCase):

    def generic_test(self,name,expr,desired):
        import parser
        ast = parser.suite(expr)
        ast_list = ast.tolist()                
        variables = compile_code.harvest_variables(ast_list)
        actual = compile_code.parse_tuple_block(name,variables)
        actual = remove_whitespace(actual)
        desired = remove_whitespace(desired)
        print_assert_equal(expr,actual,desired)

    def check_simple_expr(self):
        """convert simple expr to blitz
           
           a[:1:2] = b[:1+i+2:]
        """
        name = "test_function"
        expr = "a[:1:2] = b[:1+i+2:]"        
        desired = 'static char* kwlist[] = {"a","b","i",NULL};'\
                  'PyObject *py_a, *py_b, *py_i;'\
                  'py_a = py_b = py_i = NULL;'\
                  'blitz::TinyVector<PyObject*,3> clean_up(0);'\
                  'if(!PyArg_ParseTupleAndKeywords(args,kywds,'\
                  '"OOO:test_function",kwlist,&py_a,&py_b,&py_i))returnNULL;'
        self.generic_test(name, expr,desired)

class test_compiled_exec(unittest.TestCase):
    """* These are long running tests...
    
         I'd like to benchmark these things somehow.
    *"""
    def generic_test(self,expr,arg_dict,type,size):
        clean_result = array(arg_dict['result'],copy=1)
        t1 = time.time()
        exec expr in globals(),arg_dict
        t2 = time.time()
        standard = t2 - t1
        desired = arg_dict['result']
        arg_dict['result'] = clean_result
        t1 = time.time()
        scipy.compiler.compile_code.compiled_exec(expr,arg_dict,module_location=None)
        t2 = time.time()
        compiled = t2 - t1
        actual = arg_dict['result']
        # this really should give more info...
        try:
            # this isn't very stringent.  Need to tighten this up and
            # learn where failures are occuring.
            assert(allclose(abs(actual.flat),abs(desired.flat),1e-4,1e-6))
        except:
            diff = actual-desired
            print diff[:4,:4]
            print diff[:4,-4:]
            print diff[-4:,:4]
            print diff[-4:,-4:]
            print sum(abs(diff.flat))            
            raise AssertionError
        return standard,compiled
    def generic_2d(self,expr):
        """ The complex testing is pretty lame...
        """
        import parser
        ast = parser.suite(expr)
        arg_list = harvest_variables(ast.tolist())
        #print arg_list
        all_types = [Float32,Float64,Complex32,Complex64]
        all_sizes = [(10,10), (50,50), (100,100), (500,500), (1000,1000)]
        print '\nExpression:', expr
        for typ in all_types:
            for size in all_sizes:
                result = zeros(size,typ)
                arg_dict = {}
                for arg in arg_list:
                    arg_dict[arg] = RandomArray.normal(0,1,size).astype(typ)
                    arg_dict[arg].savespace(1)
                    # set imag part of complex values to non-zero value
                    try:     arg_dict[arg].imag = arg_dict[arg].real
                    except:  pass  
                print 'Run:', size,typ
                standard,compiled = self.generic_test(expr,arg_dict,type,size)
                try:
                    speed_up = standard/compiled
                except:
                    speed_up = -1.
                print "1st run(Numeric,compiled,speed up):  %3.4f, %3.4f, " \
                      "%3.4f" % (standard,compiled,speed_up)    
                standard,compiled = self.generic_test(expr,arg_dict,type,size)
                try:
                    speed_up = standard/compiled
                except:
                    speed_up = -1.                    
                print "2nd run(Numeric,compiled,speed up):  %3.4f, %3.4f, " \
                      "%3.4f" % (standard,compiled,speed_up)    
    #def check_simple_2d(self):
    #    """ result = a + b""" 
    #    expr = "result = a + b"
    #    self.generic_2d(expr)
    def check_5point_avg_2d(self):
        """ result[1:-1,1:-1] = (b[1:-1,1:-1] + b[2:,1:-1] + b[:-2,1:-1]
                               + b[1:-1,2:] + b[1:-1,:-2]) / 5.
        """                                  
        expr = "result[1:-1,1:-1] = (b[1:-1,1:-1] + b[2:,1:-1] + b[:-2,1:-1]" \
                                  "+ b[1:-1,2:] + b[1:-1,:-2]) / 5."
        self.generic_2d(expr)
    
    def setUp(self):
        # try and get rid of any shared libraries that currently exist in 
        # test directory.  If some other program is using them though,
        # (another process is running exact same tests, this will to 
        # fail clean-up stuff on NT)        
        #remove_test_files()
        pass
    def tearDown(self):
        #print '\n\n\ntearing down\n\n\n'
        #remove_test_files()
        pass
        # Get rid of any files created by the test such as function catalogs
        # and compiled modules.
        # We'll assume any .pyd, .so files, .cpp, .def or .o 
        # in the test directory is a test file.  To make sure we
        # don't abliterate something desireable, we'll move it
        # to a file called 'test_trash'

def remove_test_files():
    import os,glob
    test_dir = scipy.compiler.compile_code.home_dir(__file__)
    trash = os.path.join(test_dir,'test_trash')
    files = glob.glob(os.path.join(test_dir,'*.so'))
    files += glob.glob(os.path.join(test_dir,'*.o'))
    files += glob.glob(os.path.join(test_dir,'*.a'))
    files += glob.glob(os.path.join(test_dir,'*.cpp'))
    files += glob.glob(os.path.join(test_dir,'*.pyd'))
    files += glob.glob(os.path.join(test_dir,'*.def'))
    files += glob.glob(os.path.join(test_dir,'*compiled_catalog*'))
    for i in files:
        try:
            #print i
            os.remove(i)
        except:    
            pass        
        #all this was to handle "saving files in trash, but doesn't fly on NT
        #d,f=os.path.split(i)
        #trash_file = os.path.join(trash,f)
        #print 'tf:',trash_file
        #if os.path.exists(trash_file):
        #    os.remove(trash_file)
        #    print trash_file
        #os.renames(i,trash_file)

def test_suite():
    suites = []
    suites.append( unittest.makeSuite(test_ast_to_blitz_expr,'check_') )
    suites.append( unittest.makeSuite(test_harvest_variables,'check_') )
    suites.append( unittest.makeSuite(test_assign_variable_types,'check_') )
    suites.append( unittest.makeSuite(test_blitz_array_declaration,'check_') )
    suites.append( unittest.makeSuite(test_blitz_scalar_declaration,'check_') )
    suites.append( unittest.makeSuite(test_parse_tuple_block,'check_') )
    suites.append( unittest.makeSuite(test_compiled_exec,'check_') )    
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test():
    all_tests = test_suite()
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner


