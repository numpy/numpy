import unittest
from Numeric import *
# The following try/except so that non-SciPy users can still use blitz
try:
    from scipy_base.fastumath import *
except:
    pass # scipy_base.fastumath not available    
import RandomArray
import os
import time

from scipy_distutils.misc_util import add_grandparent_to_path,restore_path
from scipy_distutils.misc_util import add_local_to_path

add_grandparent_to_path(__name__)
import blitz_tools
from ast_tools import *
from weave_test_utils import *
restore_path()

add_local_to_path(__name__)
import test_scalar_spec
restore_path()

class test_ast_to_blitz_expr(unittest.TestCase):

    def generic_test(self,expr,desired):
        import parser
        ast = parser.suite(expr)
        ast_list = ast.tolist()
        actual = blitz_tools.ast_to_blitz_expr(ast_list)
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

class test_blitz(unittest.TestCase):
    """* These are long running tests...
    
         I'd like to benchmark these things somehow.
    *"""
    def generic_test(self,expr,arg_dict,type,size,mod_location):
        clean_result = array(arg_dict['result'],copy=1)
        t1 = time.time()
        exec expr in globals(),arg_dict
        t2 = time.time()
        standard = t2 - t1
        desired = arg_dict['result']
        arg_dict['result'] = clean_result
        t1 = time.time()
        old_env = os.environ.get('PYTHONCOMPILED','')
        os.environ['PYTHONCOMPILED'] = mod_location
        blitz_tools.blitz(expr,arg_dict,{},verbose=0) #,
                          #extra_compile_args = ['-O3','-malign-double','-funroll-loops'])
        os.environ['PYTHONCOMPILED'] = old_env
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
        mod_location = empty_temp_dir()
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
                standard,compiled = self.generic_test(expr,arg_dict,type,size,
                                                      mod_location)
                try:
                    speed_up = standard/compiled
                except:
                    speed_up = -1.
                print "1st run(Numeric,compiled,speed up):  %3.4f, %3.4f, " \
                      "%3.4f" % (standard,compiled,speed_up)    
                standard,compiled = self.generic_test(expr,arg_dict,type,size,
                                                      mod_location)
                try:
                    speed_up = standard/compiled
                except:
                    speed_up = -1.                    
                print "2nd run(Numeric,compiled,speed up):  %3.4f, %3.4f, " \
                      "%3.4f" % (standard,compiled,speed_up)
        cleanup_temp_dir(mod_location)                      
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
    
def test_suite(level=1):
    suites = []
    if level > 0:
        suites.append( unittest.makeSuite(test_ast_to_blitz_expr,'check_') )
    if level >= 10:
        suites.append( unittest.makeSuite(test_blitz,'check_') )    
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test(level=10):
    all_tests = test_suite(level)
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
