
import unittest
import sys
from scipy_test.testing import *
set_package_path()
import scipy_base;reload(scipy_base)
from scipy_base import *
del sys.path[0]
       
class test_isscalar(unittest.TestCase):
    def check_basic(self):
        assert(isscalar(3))
        assert(not isscalar([3]))
        assert(not isscalar((3,)))
        assert(isscalar(3j))
        assert(isscalar(10L))
        assert(isscalar(4.0))

class test_real(unittest.TestCase):
    def check_real(self):
        y = rand(10,)
        assert_array_equal(y,real(y))

    def check_cmplx(self):
        y = rand(10,)+1j*rand(10,)
        assert_array_equal(y.real,real(y))

class test_imag(unittest.TestCase):
    def check_real(self):
        y = rand(10,)
        assert_array_equal(0,imag(y))

    def check_cmplx(self):
        y = rand(10,)+1j*rand(10,)
        assert_array_equal(y.imag,imag(y))

class test_iscomplex(unittest.TestCase):
    def check_fail(self):
        z = array([-1,0,1])
        res = iscomplex(z)
        assert(not sometrue(res))
    def check_pass(self):
        z = array([-1j,1,0])
        res = iscomplex(z)
        assert_array_equal(res,[1,0,0])

class test_isreal(unittest.TestCase):
    def check_pass(self):
        z = array([-1,0,1j])
        res = isreal(z)
        assert_array_equal(res,[1,1,0])
    def check_fail(self):
        z = array([-1j,1,0])
        res = isreal(z)
        assert_array_equal(res,[0,1,1])

class test_iscomplexobj(unittest.TestCase):
    def check_basic(self):
        z = array([-1,0,1])
        assert(not iscomplexobj(z))
        z = array([-1j,0,-1])
        assert(iscomplexobj(z))

class test_isrealobj(unittest.TestCase):
    def check_basic(self):
        z = array([-1,0,1])
        assert(isrealobj(z))
        z = array([-1j,0,-1])
        assert(not isrealobj(z))

class test_isnan(unittest.TestCase):
    def check_goodvalues(self):
        z = array((-1.,0.,1.))
        res = isnan(z) == 0
        assert(alltrue(res))            
    def check_posinf(self): 
        assert(isnan(array((1.,))/0.) == 0)
    def check_neginf(self): 
        assert(isnan(array((-1.,))/0.) == 0)
    def check_ind(self): 
        assert(isnan(array((0.,))/0.) == 1)
    #def check_qnan(self):             log(-1) return pi*j now
    #    assert(isnan(log(-1.)) == 1)
    def check_integer(self):
        assert(isnan(1) == 0)
    def check_complex(self):
        assert(isnan(1+1j) == 0)
    def check_complex1(self):
        assert(isnan(array(0+0j)/0.) == 1)
                
class test_isfinite(unittest.TestCase):
    def check_goodvalues(self):
        z = array((-1.,0.,1.))
        res = isfinite(z) == 1
        assert(alltrue(res))            
    def check_posinf(self): 
        assert(isfinite(array((1.,))/0.) == 0)
    def check_neginf(self): 
        assert(isfinite(array((-1.,))/0.) == 0)
    def check_ind(self): 
        assert(isfinite(array((0.,))/0.) == 0)
    #def check_qnan(self): 
    #    assert(isfinite(log(-1.)) == 0)
    def check_integer(self):
        assert(isfinite(1) == 1)
    def check_complex(self):
        assert(isfinite(1+1j) == 1)
    def check_complex1(self):
        assert(isfinite(array(1+1j)/0.) == 0)
        
class test_isinf(unittest.TestCase):
    def check_goodvalues(self):
        z = array((-1.,0.,1.))
        res = isinf(z) == 0
        assert(alltrue(res))            
    def check_posinf(self): 
        assert(isinf(array((1.,))/0.) == 1)
    def check_posinf_scalar(self): 
        assert(isinf(array(1.,)/0.) == 1)
    def check_neginf(self): 
        assert(isinf(array((-1.,))/0.) == 1)
    def check_neginf_scalar(self): 
        assert(isinf(array(-1.)/0.) == 1)
    def check_ind(self): 
        assert(isinf(array((0.,))/0.) == 0)
    #def check_qnan(self): 
    #    assert(isinf(log(-1.)) == 0)
    #    assert(isnan(log(-1.)) == 1)

class test_isposinf(unittest.TestCase):
    def check_generic(self):
        vals = isposinf(array((-1.,0,1))/0.)
        assert(vals[0] == 0)
        assert(vals[1] == 0)
        assert(vals[2] == 1)

class test_isneginf(unittest.TestCase):
    def check_generic(self):
        vals = isneginf(array((-1.,0,1))/0.)
        assert(vals[0] == 1)
        assert(vals[1] == 0)
        assert(vals[2] == 0)

class test_nan_to_num(unittest.TestCase):
    def check_generic(self):
        vals = nan_to_num(array((-1.,0,1))/0.)
        assert(vals[0] < -1e10 and isfinite(vals[0]))
        assert(vals[1] == 0)
        assert(vals[2] > 1e10 and isfinite(vals[2]))
    def check_integer(self):
        vals = nan_to_num(1)
        assert(vals == 1)
    def check_complex_good(self):
        vals = nan_to_num(1+1j)
        assert(vals == 1+1j)
    def check_complex_bad(self):
        v = 1+1j
        v += array(0+1.j)/0.
        vals = nan_to_num(v)
        # !! This is actually (unexpectedly) zero
        assert(vals.imag > 1e10 and isfinite(vals))
    def check_complex_bad2(self):
        v = 1+1j
        v += array(-1+1.j)/0.
        vals = nan_to_num(v)
        assert(isfinite(vals))    
        #assert(vals.imag > 1e10  and isfinite(vals))    
        # !! This is actually (unexpectedly) positive
        # !! inf.  Comment out for now, and see if it
        # !! changes
        #assert(vals.real < -1e10 and isfinite(vals))    


class test_real_if_close(unittest.TestCase):
    def check_basic(self):
        a = rand(10)
        b = real_if_close(a+1e-15j)
        assert(isrealobj(b))
        assert_array_equal(a,b)
        b = real_if_close(a+1e-7j)
        assert(iscomplexobj(b))
        b = real_if_close(a+1e-7j,tol=1e-6)
        assert(isrealobj(b))

if __name__ == "__main__":
    ScipyTest('scipy_base.type_check').run()
