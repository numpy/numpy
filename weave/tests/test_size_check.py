import unittest, os
from Numeric import *
# The following try/except so that non-SciPy users can still use blitz
try:
    from scipy_base.fastumath import *
except:
    pass # scipy_base.fastumath not available    

from scipy_test.testing import *
set_package_path()
from weave import size_check
from weave.ast_tools import *
restore_path()

empty = array(())
 
def array_assert_equal(test_string,actual,desired):
    """this should probably be in scipy_test.testing
    """
    import pprint        
    try:
        assert(alltrue(equal(actual,desired)))
    except AssertionError:
        try:
            # kluge for bug in Numeric
            assert (len(actual[0]) == len(actual[1]) == 
                    len(desired[0]) == len(desired[1]) == 0)
        except:    
            import cStringIO
            msg = cStringIO.StringIO()
            msg.write(test_string)
            msg.write(' failed\nACTUAL: \n')
            pprint.pprint(actual,msg)
            msg.write('DESIRED: \n')
            pprint.pprint(desired,msg)
            raise AssertionError, msg.getvalue()

class test_make_same_length(unittest.TestCase):

    def generic_test(self,x,y,desired):
        actual = size_check.make_same_length(x,y)
        desired = desired
        array_assert_equal('',actual,desired)

    def check_scalar(self):
        x,y = (),()
        desired = empty,empty        
        self.generic_test(x,y,desired)
    def check_x_scalar(self):
        x,y = (),(1,2)
        desired = array((1,1)),array((1,2))
        self.generic_test(x,y,desired)
    def check_y_scalar(self):
        x,y = (1,2),()
        desired = array((1,2)),array((1,1))
        self.generic_test(x,y,desired)
    def check_x_short(self):
        x,y = (1,2),(1,2,3)
        desired = array((1,1,2)),array((1,2,3))
        self.generic_test(x,y,desired)
    def check_y_short(self):
        x,y = (1,2,3),(1,2)
        desired = array((1,2,3)),array((1,1,2))
        self.generic_test(x,y,desired)

class test_binary_op_size(unittest.TestCase):
    def generic_test(self,x,y,desired):
        actual = size_check.binary_op_size(x,y)
        desired = desired
        array_assert_equal('',actual,desired)
    def generic_error_test(self,x,y):
        try:
            actual = size_check.binary_op_size(x,y)
            #print actual
            raise AttributeError, "Should have raised ValueError"
        except ValueError:
            pass    
    def desired_type(self,val):
        return array(val)            
    def check_scalar(self):
        x,y = (),()
        desired = self.desired_type(())
        self.generic_test(x,y,desired)
    def check_x1(self):
        x,y = (1,),()
        desired = self.desired_type((1,))
        self.generic_test(x,y,desired)
    def check_y1(self):
        x,y = (),(1,)
        desired = self.desired_type((1,))
        self.generic_test(x,y,desired)
    def check_x_y(self):
        x,y = (5,),(5,)
        desired = self.desired_type((5,))
        self.generic_test(x,y,desired)
    def check_x_y2(self):
        x,y = (5,10),(5,10)
        desired = self.desired_type((5,10))
        self.generic_test(x,y,desired)
    def check_x_y3(self):
        x,y = (5,10),(1,10)
        desired = self.desired_type((5,10))
        self.generic_test(x,y,desired)
    def check_x_y4(self):
        x,y = (1,10),(5,10)
        desired = self.desired_type((5,10))
        self.generic_test(x,y,desired)
    def check_x_y5(self):
        x,y = (5,1),(1,10)
        desired = self.desired_type((5,10))
        self.generic_test(x,y,desired)
    def check_x_y6(self):
        x,y = (1,10),(5,1)
        desired = self.desired_type((5,10))
        self.generic_test(x,y,desired)
    def check_x_y7(self):
        x,y = (5,4,3,2,1),(3,2,1)
        desired = self.desired_type((5,4,3,2,1))
        self.generic_test(x,y,desired)
        
    def check_error1(self):
        x,y = (5,),(4,)
        self.generic_error_test(x,y)
    def check_error2(self):
        x,y = (5,5),(4,5)
        self.generic_error_test(x,y)

class test_dummy_array(test_binary_op_size):
    def array_assert_equal(self,test_string,actual,desired):
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
    def generic_test(self,x,y,desired):
        if type(x) is type(()):
            x = ones(x)
        if type(y) is type(()):
            y = ones(y)
        xx = size_check.dummy_array(x)
        yy = size_check.dummy_array(y)
        ops = ['+', '-', '/', '*', '<<', '>>']
        for op in ops:
            actual = eval('xx' + op + 'yy')
            desired = desired
            self.array_assert_equal('',actual,desired)
    def generic_error_test(self,x,y):
        try:
            self.generic_test('',x,y)
            raise AttributeError, "Should have raised ValueError"
        except ValueError:
            pass    
    def desired_type(self,val):
        return size_check.dummy_array(array(val),1)

class test_dummy_array_indexing(unittest.TestCase):
    def array_assert_equal(self,test_string,actual,desired):
        """this should probably be in scipy_test.testing
        """
        import pprint        
        try:
            assert(alltrue(equal(actual,desired)))            
        except AssertionError:
            import cStringIO
            msg = cStringIO.StringIO()
            msg.write(test_string)
            msg.write(' failed\nACTUAL: \n')
            pprint.pprint(actual,msg)
            msg.write('DESIRED: \n')
            pprint.pprint(desired,msg)
            raise AssertionError, msg.getvalue()
    def generic_test(self,ary,expr,desired):
        a = size_check.dummy_array(ary)
        actual = eval(expr).shape        
        #print desired, actual
        self.array_assert_equal(expr,actual,desired)
    def generic_wrap(self,a,expr):
        #print expr ,eval(expr)
        desired = array(eval(expr).shape)
        try:
            self.generic_test(a,expr,desired)
        except IndexError:
            if 0 not in desired:
                msg = '%s raised IndexError in dummy_array, but forms\n' \
                      'valid array shape -> %s' % (expr, str(desired))
                raise AttributeError, msg        
    def generic_1d(self,expr):
        a = arange(10)
        self.generic_wrap(a,expr)
    def generic_2d(self,expr):
        a = ones((10,20))
        self.generic_wrap(a,expr)
    def generic_3d(self,expr):
        a = ones((10,20,1))
        self.generic_wrap(a,expr)
        
    def generic_1d_index(self,expr):
        a = arange(10)
        #print expr ,eval(expr)
        desired = array(())
        self.generic_test(a,expr,desired)
    def check_1d_index_0(self):
        self.generic_1d_index('a[0]')
    def check_1d_index_1(self):
        self.generic_1d_index('a[4]')
    def check_1d_index_2(self):
        self.generic_1d_index('a[-4]')
    def check_1d_index_3(self):
        try: self.generic_1d('a[12]')
        except IndexError: pass            
    def check_1d_index_calculated(self):
        self.generic_1d_index('a[0+1]')
    def check_1d_0(self):
        self.generic_1d('a[:]')
    def check_1d_1(self):            
        self.generic_1d('a[1:]')
    def check_1d_2(self):            
        self.generic_1d('a[-1:]')
    def check_1d_3(self):
        # dummy_array is "bug for bug" equiv to Numeric.array
        # on wrapping of indices.
        self.generic_1d('a[-11:]')
    def check_1d_4(self):            
        self.generic_1d('a[:1]')
    def check_1d_5(self):            
        self.generic_1d('a[:-1]')
    def check_1d_6(self):            
        self.generic_1d('a[:-11]')
    def check_1d_7(self):            
        self.generic_1d('a[1:5]')
    def check_1d_8(self):            
        self.generic_1d('a[1:-5]')
    def check_1d_9(self):
        # don't support zero length slicing at the moment.
        try: self.generic_1d('a[-1:-5]')
        except IndexError: pass            
    def check_1d_10(self):            
        self.generic_1d('a[-5:-1]')
        
    def check_1d_stride_0(self):            
        self.generic_1d('a[::1]')        
    def check_1d_stride_1(self):            
        self.generic_1d('a[::-1]')        
    def check_1d_stride_2(self):            
        self.generic_1d('a[1::1]')        
    def check_1d_stride_3(self):            
        self.generic_1d('a[1::-1]')        
    def check_1d_stride_4(self):            
        # don't support zero length slicing at the moment.
        try: self.generic_1d('a[1:5:-1]')        
        except IndexError: pass            
    def check_1d_stride_5(self):            
        self.generic_1d('a[5:1:-1]')        
    def check_1d_stride_6(self):            
        self.generic_1d('a[:4:1]')        
    def check_1d_stride_7(self):            
        self.generic_1d('a[:4:-1]')        
    def check_1d_stride_8(self):            
        self.generic_1d('a[:-4:1]')        
    def check_1d_stride_9(self):            
        self.generic_1d('a[:-4:-1]')        
    def check_1d_stride_10(self):            
        self.generic_1d('a[:-3:2]')        
    def check_1d_stride_11(self):            
        self.generic_1d('a[:-3:-2]')        
    def check_1d_stride_12(self):            
        self.generic_1d('a[:-3:-7]')        
    def check_1d_random(self):
        """ through a bunch of different indexes at it for good measure.
        """
        import random
        choices = map(lambda x: `x`,range(50)) + range(50) + ['']*50
        for i in range(100):
            try:
                beg = random.choice(choices)
                end = random.choice(choices)
                step = random.choice(choices)                
                self.generic_1d('a[%s:%s:%s]' %(beg,end,step))        
            except IndexError:
                pass

    def check_2d_0(self):
        self.generic_2d('a[:]')
    def check_2d_1(self):
        self.generic_2d('a[:2]')
    def check_2d_2(self):
        self.generic_2d('a[:,:]')
    def check_2d_random(self):
        """ through a bunch of different indexes at it for good measure.
        """
        import random
        choices = map(lambda x: `x`,range(50)) + range(50) + ['']*50        
        for i in range(100):
            try:
                beg = random.choice(choices)
                end = random.choice(choices)
                step = random.choice(choices)                
                beg2 = random.choice(choices)
                end2 = random.choice(choices)
                step2 = random.choice(choices)                
                expr = 'a[%s:%s:%s,%s:%s:%s]' %(beg,end,step,beg2,end2,step2)
                self.generic_2d(expr)        
            except IndexError:
                pass
    def check_3d_random(self):
        """ through a bunch of different indexes at it for good measure.
        """
        import random
        choices = map(lambda x: `x`,range(50)) + range(50) + ['']*50        
        for i in range(100):
            try:
                idx = []
                for i in range(9):
                    idx.append(random.choice(choices))
                expr = 'a[%s:%s:%s,%s:%s:%s,%s:%s:%s]' % tuple(idx)
                self.generic_3d(expr)        
            except IndexError:
                pass

class test_reduction(unittest.TestCase):
    def check_1d_0(self):
        a = ones((5,))
        actual = size_check.reduction(a,0)
        desired = size_check.dummy_array((),1)
        array_assert_equal('',actual.shape,desired.shape)        
    def check_2d_0(self):
        a = ones((5,10))
        actual = size_check.reduction(a,0)
        desired = size_check.dummy_array((10,),1)
        array_assert_equal('',actual.shape,desired.shape)        
    def check_2d_1(self):
        a = ones((5,10))
        actual = size_check.reduction(a,1)
        desired = size_check.dummy_array((5,),1)
        array_assert_equal('',actual.shape,desired.shape)        
    def check_3d_0(self):
        a = ones((5,6,7))
        actual = size_check.reduction(a,1)
        desired = size_check.dummy_array((5,7),1)
        array_assert_equal('',actual.shape,desired.shape)        
    def check_error0(self):
        a = ones((5,))
        try:
            actual = size_check.reduction(a,-2)
        except ValueError:
            pass            
    def check_error1(self):
        a = ones((5,))
        try:
            actual = size_check.reduction(a,1)
        except ValueError:
            pass            

class test_expressions(unittest.TestCase):        
    def array_assert_equal(self,test_string,actual,desired):
        """this should probably be in scipy_test.testing
        """
        import pprint        
        try:
            assert(alltrue(equal(actual,desired)))            
        except AssertionError:
            import cStringIO
            msg = cStringIO.StringIO()
            msg.write(test_string)
            msg.write(' failed\nACTUAL: \n')
            pprint.pprint(actual,msg)
            msg.write('DESIRED: \n')
            pprint.pprint(desired,msg)
            raise AssertionError, msg.getvalue()
    def generic_test(self,expr,desired,**kw):
        import parser
        ast_list = parser.expr(expr).tolist()
        args = harvest_variables(ast_list)
        loc = locals().update(kw)
        for var in args:
            s='%s = size_check.dummy_array(%s)'% (var,var)
            exec(s,loc)
        try:    
            actual = eval(expr,locals()).shape        
        except:
            actual = 'failed'    
        if actual is 'failed' and  desired is 'failed':
            return
        try:            
            self.array_assert_equal(expr,actual,desired)
        except:
            print 'EXPR:',expr
            print 'ACTUAL:',actual
            print 'DEISRED:',desired
    def generic_wrap(self,expr,**kw):
        try:
            x = array(eval(expr,kw))
            try:
                desired = x.shape
            except:
                desired = zeros(())
        except:
            desired = 'failed'
        self.generic_test(expr,desired,**kw)
    def check_generic_1d(self):
        a = arange(10)    
        expr = 'a[:]'    
        self.generic_wrap(expr,a=a)
        expr = 'a[:] + a'    
        self.generic_wrap(expr,a=a)
        bad_expr = 'a[4:] + a'    
        self.generic_wrap(bad_expr,a=a)
        a = arange(10)    
        b = ones((1,10))
        expr = 'a + b'    
        self.generic_wrap(expr,a=a,b=b)
        bad_expr = 'a[:5] + b'    
        self.generic_wrap(bad_expr,a=a,b=b)
    def check_single_index(self):    
        a = arange(10)    
        expr = 'a[5] + a[3]'    
        self.generic_wrap(expr,a=a)
        
    def check_calculated_index(self):    
        a = arange(10)    
        nx = 0
        expr = 'a[5] + a[nx+3]'    
        size_check.check_expr(expr,locals())
    def check_calculated_index2(self):    
        a = arange(10)    
        nx = 0
        expr = 'a[1:5] + a[nx+1:5+nx]'    
        size_check.check_expr(expr,locals())
    def generic_2d(self,expr):
        a = ones((10,20))
        self.generic_wrap(a,expr)
    def generic_3d(self,expr):
        a = ones((10,20,1))
        self.generic_wrap(a,expr)
    

if __name__ == "__main__":
    ScipyTest('weave.size_check').run()
