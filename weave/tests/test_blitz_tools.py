import unittest
from Numeric import *
from fastumath import *
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
    def generic_test(self,expr,arg_dict,type,size):
        mod_location = setup_test_location()
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
        blitz_tools.blitz(expr,arg_dict,{},verbose=0)
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
            teardown_test_location()
            raise AssertionError        
        teardown_test_location()                
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
        # Get rid of any files created by the test such as function catalogs
        # and compiled modules.
        # We'll assume any .pyd, .so files, .cpp, .def or .o 
        # in the test directory is a test file.  To make sure we
        # don't abliterate something desireable, we'll move it
        # to a file called 'test_trash'
        teardown_test_location()
        
def remove_test_files():
    import os,glob
    test_dir = compiler.compile_code.home_dir(__file__)
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

def setup_test_location():
    import tempfile
    pth = os.path.join(tempfile.tempdir,'test_files')
    if not os.path.exists(pth):
        os.mkdir(pth)
    #sys.path.insert(0,pth)    
    return pth

def teardown_test_location():
    pass
    #import test_scalar_spec    
    pth = os.path.join(tempfile.tempdir,'test_files')
    #if sys.path[0] == pth:
    #    sys.path = sys.path[1:]
    #return pth

def test_suite():
    suites = []
    suites.append( unittest.makeSuite(test_ast_to_blitz_expr,'check_') )
    suites.append( unittest.makeSuite(test_blitz,'check_') )    
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test():
    all_tests = test_suite()
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
