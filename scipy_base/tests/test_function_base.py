
import unittest

import sys
from scipy_test.testing import *
set_package_path()
import scipy_base;reload(scipy_base)
from scipy_base import *
del sys.path[0]

class test_any(unittest.TestCase):
    def check_basic(self):
        y1 = [0,0,1,0]
        y2 = [0,0,0,0]
        y3 = [1,0,1,0]
        assert(any(y1))
        assert(any(y3))
        assert(not any(y2))

    def check_nd(self):
        y1 = [[0,0,0],[0,1,0],[1,1,0]]
        assert(any(y1))
        assert_array_equal(sometrue(y1),[1,1,0])
        assert_array_equal(sometrue(y1,axis=1),[0,1,1])
        
class test_all(unittest.TestCase):
    def check_basic(self):
        y1 = [0,1,1,0]
        y2 = [0,0,0,0]
        y3 = [1,1,1,1]
        assert(not all(y1))
        assert(all(y3))
        assert(not all(y2))
        assert(all(~array(y2)))

    def check_nd(self):
        y1 = [[0,0,1],[0,1,1],[1,1,1]]
        assert(not all(y1))
        assert_array_equal(alltrue(y1),[0,0,1])
        assert_array_equal(alltrue(y1,axis=1),[0,0,1])

class test_logspace(unittest.TestCase):
    def check_basic(self):
        y = logspace(0,6)
        assert(len(y)==50)
        y = logspace(0,6,num=100)
        assert(y[-1] == 10**6)
        y = logspace(0,6,endpoint=0)
        assert(y[-1] < 10**6)
        y = logspace(0,6,num=7)
        assert_array_equal(y,[1,10,100,1e3,1e4,1e5,1e6])

class test_linspace(unittest.TestCase):
    def check_basic(self):
        y = linspace(0,10)
        assert(len(y)==50)
        y = linspace(2,10,num=100)
        assert(y[-1] == 10)
        y = linspace(2,10,endpoint=0)
        assert(y[-1] < 10)
        y,st = linspace(2,10,retstep=1)
        assert_almost_equal(st,8/49.0)
        assert_array_almost_equal(y,mgrid[2:10:50j],13)

class test_amax(unittest.TestCase):
    def check_basic(self):
        a = [3,4,5,10,-3,-5,6.0]
        assert_equal(amax(a),10.0)
        b = [[3,6.0, 9.0],
             [4,10.0,5.0],
             [8,3.0,2.0]]
        assert_equal(amax(b),[8.0,10.0,9.0])
        assert_equal(amax(b,axis=1),[9.0,10.0,8.0])
        
class test_amin(unittest.TestCase):
    def check_basic(self):
        a = [3,4,5,10,-3,-5,6.0]
        assert_equal(amin(a),-5.0)
        b = [[3,6.0, 9.0],
             [4,10.0,5.0],
             [8,3.0,2.0]]
        assert_equal(amin(b),[3.0,3.0,2.0])
        assert_equal(amin(b,axis=1),[3.0,4.0,2.0])

class test_ptp(unittest.TestCase):
    def check_basic(self):
        a = [3,4,5,10,-3,-5,6.0]
        assert_equal(ptp(a),15.0)
        b = [[3,6.0, 9.0],
             [4,10.0,5.0],
             [8,3.0,2.0]]
        assert_equal(ptp(b,axis=0),[5.0,7.0,7.0])
        assert_equal(ptp(b),[6.0,6.0,6.0])

class test_cumsum(unittest.TestCase):
    def check_basic(self):
        ba = [1,2,10,11,6,5,4]
        ba2 = [[1,2,3,4],[5,6,7,9],[10,3,4,5]]
        for ctype in ['1','b','s','i','l','f','d','F','D']:
            a = array(ba,ctype)
            a2 = array(ba2,ctype)
            assert_array_equal(cumsum(a), array([1,3,13,24,30,35,39],ctype))
            assert_array_equal(cumsum(a2,axis=0), array([[1,2,3,4],[6,8,10,13],
                                                         [16,11,14,18]],ctype))
            assert_array_equal(cumsum(a2,axis=1),
                               array([[1,3,6,10],
                                      [5,11,18,27],
                                      [10,13,17,22]],ctype))

class test_prod(unittest.TestCase):
    def check_basic(self):
        ba = [1,2,10,11,6,5,4]
        ba2 = [[1,2,3,4],[5,6,7,9],[10,3,4,5]]
        for ctype in ['1','b','s','i','l','f','d','F','D']:
            a = array(ba,ctype)
            a2 = array(ba2,ctype)
            if ctype in ['1', 'b']:
                self.failUnlessRaises(ArithmeticError, prod, a)
                self.failUnlessRaises(ArithmeticError, prod, a2, 1)
                self.failUnlessRaises(ArithmeticError, prod, a)
            else:                
                assert_equal(prod(a),26400)
                assert_array_equal(prod(a2,axis=0), 
                                   array([50,36,84,180],ctype))
                assert_array_equal(prod(a2),array([24, 1890, 600],ctype))

class test_cumprod(unittest.TestCase):
    def check_basic(self):
        ba = [1,2,10,11,6,5,4]
        ba2 = [[1,2,3,4],[5,6,7,9],[10,3,4,5]]
        for ctype in ['1','b','s','i','l','f','d','F','D']:
            a = array(ba,ctype)
            a2 = array(ba2,ctype)
            if ctype in ['1', 'b']:
                self.failUnlessRaises(ArithmeticError, cumprod, a)
                self.failUnlessRaises(ArithmeticError, cumprod, a2, 1)
                self.failUnlessRaises(ArithmeticError, cumprod, a)
            else:                
                assert_array_equal(cumprod(a),
                                   array([1, 2, 20, 220,
                                          1320, 6600, 26400],ctype))
                assert_array_equal(cumprod(a2,axis=0),
                                   array([[ 1,  2,  3,   4],
                                          [ 5, 12, 21,  36],
                                          [50, 36, 84, 180]],ctype))
                assert_array_equal(cumprod(a2),
                                   array([[ 1,  2,   6,   24],
                                          [ 5, 30, 210, 1890],
                                          [10, 30, 120,  600]],ctype))

class test_diff(unittest.TestCase):
    def check_basic(self):
        x = [1,4,6,7,12]
        out = array([3,2,1,5])
        out2 = array([-1,-1,4])
        out3 = array([0,5])
        assert_array_equal(diff(x),out)
        assert_array_equal(diff(x,n=2),out2)
        assert_array_equal(diff(x,n=3),out3)

    def check_nd(self):
        x = 20*rand(10,20,30)
        out1 = x[:,:,1:] - x[:,:,:-1]
        out2 = out1[:,:,1:] - out1[:,:,:-1]
        out3 = x[1:,:,:] - x[:-1,:,:]
        out4 = out3[1:,:,:] - out3[:-1,:,:]
        assert_array_equal(diff(x),out1)
        assert_array_equal(diff(x,n=2),out2)
        assert_array_equal(diff(x,axis=0),out3)
        assert_array_equal(diff(x,n=2,axis=0),out4)

class test_angle(unittest.TestCase):
    def check_basic(self):
        x = [1+3j,sqrt(2)/2.0+1j*sqrt(2)/2,1,1j,-1,-1j,1-3j,-1+3j]
        y = angle(x)
        yo = [arctan(3.0/1.0),arctan(1.0),0,pi/2,pi,-pi/2.0,
              -arctan(3.0/1.0),pi-arctan(3.0/1.0)]
        z = angle(x,deg=1)
        zo = array(yo)*180/pi
        assert_array_almost_equal(y,yo,11)
        assert_array_almost_equal(z,zo,11)

class test_trim_zeros(unittest.TestCase):
    """ only testing for integer splits.
    """
    def check_basic(self):
        a= array([0,0,1,2,3,4,0])
        res = trim_zeros(a)
        assert_array_equal(res,array([1,2,3,4]))
    def check_leading_skip(self):
        a= array([0,0,1,0,2,3,4,0])
        res = trim_zeros(a)
        assert_array_equal(res,array([1,0,2,3,4]))
    def check_trailing_skip(self):
        a= array([0,0,1,0,2,3,0,4,0])
        res = trim_zeros(a)
        assert_array_equal(res,array([1,0,2,3,0,4]))


class test_extins(unittest.TestCase):
    def check_basic(self):
        a = array([1,3,2,1,2,3,3])
        b = extract(a>1,a)
        assert_array_equal(b,[3,2,2,3,3])
    def check_insert(self):
        a = array([1,4,3,2,5,8,7])
        insert(a,[0,1,0,1,0,1,0],[2,4,6])
        assert_array_equal(a,[1,2,3,4,5,6,7])
    def check_both(self):
        a = rand(10)
        mask = a > 0.5
        ac = a.copy()
        c = extract(mask, a)
        insert(a,mask,0)
        insert(a,mask,c)
        assert_array_equal(a,ac)
                
class test_vectorize(unittest.TestCase):
    def check_simple(self):
        def addsubtract(a,b):
            if a > b:
                return a - b
            else:
                return a + b
        f = vectorize(addsubtract)
        r = f([0,3,6,9],[1,3,5,7])
        assert_array_equal(r,[1,6,1,2])
    def check_scalar(self):
        def addsubtract(a,b):
            if a > b:
                return a - b
            else:
                return a + b
        f = vectorize(addsubtract)
        r = f([0,3,6,9],5)
        assert_array_equal(r,[5,8,1,4])
        

def compare_results(res,desired):
    for i in range(len(desired)):
        assert_array_equal(res[i],desired[i])

if __name__ == "__main__":
    ScipyTest('scipy_base.function_base').run()
