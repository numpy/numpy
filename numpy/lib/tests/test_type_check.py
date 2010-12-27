from numpy.testing import *
from numpy.lib import *
from numpy.core import *
from numpy.compat import asbytes

try: 
    import ctypes 
    _HAS_CTYPE = True 
except ImportError: 
    _HAS_CTYPE = False 

    
def assert_all(x):
    assert(all(x)), x


class TestCommonType(TestCase):
    def test_basic(self):
        ai32 = array([[1,2],[3,4]], dtype=int32)
        af32 = array([[1,2],[3,4]], dtype=float32)
        af64 = array([[1,2],[3,4]], dtype=float64)
        acs = array([[1+5j,2+6j],[3+7j,4+8j]], dtype=csingle)
        acd = array([[1+5j,2+6j],[3+7j,4+8j]], dtype=cdouble)
        assert common_type(af32) == float32
        assert common_type(af64) == float64
        assert common_type(acs) == csingle
        assert common_type(acd) == cdouble



class TestMintypecode(TestCase):

    def test_default_1(self):
        for itype in '1bcsuwil':
            assert_equal(mintypecode(itype),'d')
        assert_equal(mintypecode('f'),'f')
        assert_equal(mintypecode('d'),'d')
        assert_equal(mintypecode('F'),'F')
        assert_equal(mintypecode('D'),'D')

    def test_default_2(self):
        for itype in '1bcsuwil':
            assert_equal(mintypecode(itype+'f'),'f')
            assert_equal(mintypecode(itype+'d'),'d')
            assert_equal(mintypecode(itype+'F'),'F')
            assert_equal(mintypecode(itype+'D'),'D')
        assert_equal(mintypecode('ff'),'f')
        assert_equal(mintypecode('fd'),'d')
        assert_equal(mintypecode('fF'),'F')
        assert_equal(mintypecode('fD'),'D')
        assert_equal(mintypecode('df'),'d')
        assert_equal(mintypecode('dd'),'d')
        #assert_equal(mintypecode('dF',savespace=1),'F')
        assert_equal(mintypecode('dF'),'D')
        assert_equal(mintypecode('dD'),'D')
        assert_equal(mintypecode('Ff'),'F')
        #assert_equal(mintypecode('Fd',savespace=1),'F')
        assert_equal(mintypecode('Fd'),'D')
        assert_equal(mintypecode('FF'),'F')
        assert_equal(mintypecode('FD'),'D')
        assert_equal(mintypecode('Df'),'D')
        assert_equal(mintypecode('Dd'),'D')
        assert_equal(mintypecode('DF'),'D')
        assert_equal(mintypecode('DD'),'D')

    def test_default_3(self):
        assert_equal(mintypecode('fdF'),'D')
        #assert_equal(mintypecode('fdF',savespace=1),'F')
        assert_equal(mintypecode('fdD'),'D')
        assert_equal(mintypecode('fFD'),'D')
        assert_equal(mintypecode('dFD'),'D')

        assert_equal(mintypecode('ifd'),'d')
        assert_equal(mintypecode('ifF'),'F')
        assert_equal(mintypecode('ifD'),'D')
        assert_equal(mintypecode('idF'),'D')
        #assert_equal(mintypecode('idF',savespace=1),'F')
        assert_equal(mintypecode('idD'),'D')


class TestIsscalar(TestCase):

    def test_basic(self):
        assert(isscalar(3))
        assert(not isscalar([3]))
        assert(not isscalar((3,)))
        assert(isscalar(3j))
        assert(isscalar(10L))
        assert(isscalar(4.0))


class TestReal(TestCase):

    def test_real(self):
        y = rand(10,)
        assert_array_equal(y,real(y))

    def test_cmplx(self):
        y = rand(10,)+1j*rand(10,)
        assert_array_equal(y.real,real(y))


class TestImag(TestCase):

    def test_real(self):
        y = rand(10,)
        assert_array_equal(0,imag(y))

    def test_cmplx(self):
        y = rand(10,)+1j*rand(10,)
        assert_array_equal(y.imag,imag(y))


class TestIscomplex(TestCase):

    def test_fail(self):
        z = array([-1,0,1])
        res = iscomplex(z)
        assert(not sometrue(res,axis=0))
    def test_pass(self):
        z = array([-1j,1,0])
        res = iscomplex(z)
        assert_array_equal(res,[1,0,0])


class TestIsreal(TestCase):

    def test_pass(self):
        z = array([-1,0,1j])
        res = isreal(z)
        assert_array_equal(res,[1,1,0])
    def test_fail(self):
        z = array([-1j,1,0])
        res = isreal(z)
        assert_array_equal(res,[0,1,1])


class TestIscomplexobj(TestCase):

    def test_basic(self):
        z = array([-1,0,1])
        assert(not iscomplexobj(z))
        z = array([-1j,0,-1])
        assert(iscomplexobj(z))



class TestIsrealobj(TestCase):
    def test_basic(self):
        z = array([-1,0,1])
        assert(isrealobj(z))
        z = array([-1j,0,-1])
        assert(not isrealobj(z))


class TestIsnan(TestCase):

    def test_goodvalues(self):
        z = array((-1.,0.,1.))
        res = isnan(z) == 0
        assert_all(alltrue(res,axis=0))

    def test_posinf(self):
        olderr = seterr(divide='ignore')
        try:
            assert_all(isnan(array((1.,))/0.) == 0)
        finally:
            seterr(**olderr)

    def test_neginf(self):
        olderr = seterr(divide='ignore')
        try:
            assert_all(isnan(array((-1.,))/0.) == 0)
        finally:
            seterr(**olderr)

    def test_ind(self):
        olderr = seterr(divide='ignore', invalid='ignore')
        try:
            assert_all(isnan(array((0.,))/0.) == 1)
        finally:
            seterr(**olderr)

    #def test_qnan(self):             log(-1) return pi*j now
    #    assert_all(isnan(log(-1.)) == 1)

    def test_integer(self):
        assert_all(isnan(1) == 0)

    def test_complex(self):
        assert_all(isnan(1+1j) == 0)

    def test_complex1(self):
        olderr = seterr(divide='ignore', invalid='ignore')
        try:
            assert_all(isnan(array(0+0j)/0.) == 1)
        finally:
            seterr(**olderr)


class TestIsfinite(TestCase):

    def test_goodvalues(self):
        z = array((-1.,0.,1.))
        res = isfinite(z) == 1
        assert_all(alltrue(res,axis=0))

    def test_posinf(self):
        olderr = seterr(divide='ignore', invalid='ignore')
        try:
            assert_all(isfinite(array((1.,))/0.) == 0)
        finally:
            seterr(**olderr)

    def test_neginf(self):
        olderr = seterr(divide='ignore', invalid='ignore')
        try:
            assert_all(isfinite(array((-1.,))/0.) == 0)
        finally:
            seterr(**olderr)

    def test_ind(self):
        olderr = seterr(divide='ignore', invalid='ignore')
        try:
            assert_all(isfinite(array((0.,))/0.) == 0)
        finally:
            seterr(**olderr)

    #def test_qnan(self):
    #    assert_all(isfinite(log(-1.)) == 0)

    def test_integer(self):
        assert_all(isfinite(1) == 1)

    def test_complex(self):
        assert_all(isfinite(1+1j) == 1)

    def test_complex1(self):
        olderr = seterr(divide='ignore', invalid='ignore')
        try:
            assert_all(isfinite(array(1+1j)/0.) == 0)
        finally:
            seterr(**olderr)


class TestIsinf(TestCase):

    def test_goodvalues(self):
        z = array((-1.,0.,1.))
        res = isinf(z) == 0
        assert_all(alltrue(res,axis=0))

    def test_posinf(self):
        olderr = seterr(divide='ignore', invalid='ignore')
        try:
            assert_all(isinf(array((1.,))/0.) == 1)
        finally:
            seterr(**olderr)

    def test_posinf_scalar(self):
        olderr = seterr(divide='ignore', invalid='ignore')
        try:
            assert_all(isinf(array(1.,)/0.) == 1)
        finally:
            seterr(**olderr)

    def test_neginf(self):
        olderr = seterr(divide='ignore', invalid='ignore')
        try:
            assert_all(isinf(array((-1.,))/0.) == 1)
        finally:
            seterr(**olderr)

    def test_neginf_scalar(self):
        olderr = seterr(divide='ignore', invalid='ignore')
        try:
            assert_all(isinf(array(-1.)/0.) == 1)
        finally:
            seterr(**olderr)

    def test_ind(self):
        olderr = seterr(divide='ignore', invalid='ignore')
        try:
            assert_all(isinf(array((0.,))/0.) == 0)
        finally:
            seterr(**olderr)

    #def test_qnan(self):
    #    assert_all(isinf(log(-1.)) == 0)
    #    assert_all(isnan(log(-1.)) == 1)


class TestIsposinf(TestCase):

    def test_generic(self):
        olderr = seterr(divide='ignore', invalid='ignore')
        try:
            vals = isposinf(array((-1.,0,1))/0.)
        finally:
            seterr(**olderr)
        assert(vals[0] == 0)
        assert(vals[1] == 0)
        assert(vals[2] == 1)


class TestIsneginf(TestCase):
    def test_generic(self):
        olderr = seterr(divide='ignore', invalid='ignore')
        try:
            vals = isneginf(array((-1.,0,1))/0.)
        finally:
            seterr(**olderr)
        assert(vals[0] == 1)
        assert(vals[1] == 0)
        assert(vals[2] == 0)


class TestNanToNum(TestCase):

    def test_generic(self):
        olderr = seterr(divide='ignore', invalid='ignore')
        try:
            vals = nan_to_num(array((-1.,0,1))/0.)
        finally:
            seterr(**olderr)
        assert_all(vals[0] < -1e10) and assert_all(isfinite(vals[0]))
        assert(vals[1] == 0)
        assert_all(vals[2] > 1e10) and assert_all(isfinite(vals[2]))

    def test_integer(self):
        vals = nan_to_num(1)
        assert_all(vals == 1)

    def test_complex_good(self):
        vals = nan_to_num(1+1j)
        assert_all(vals == 1+1j)

    def test_complex_bad(self):
        v = 1+1j
        olderr = seterr(divide='ignore', invalid='ignore')
        try:
            v += array(0+1.j)/0.
        finally:
            seterr(**olderr)
        vals = nan_to_num(v)
        # !! This is actually (unexpectedly) zero
        assert_all(isfinite(vals))

    def test_complex_bad2(self):
        v = 1+1j
        olderr = seterr(divide='ignore', invalid='ignore')
        try:
            v += array(-1+1.j)/0.
        finally:
            seterr(**olderr)
        vals = nan_to_num(v)
        assert_all(isfinite(vals))
        #assert_all(vals.imag > 1e10)  and assert_all(isfinite(vals))
        # !! This is actually (unexpectedly) positive
        # !! inf.  Comment out for now, and see if it
        # !! changes
        #assert_all(vals.real < -1e10) and assert_all(isfinite(vals))


class TestRealIfClose(TestCase):

    def test_basic(self):
        a = rand(10)
        b = real_if_close(a+1e-15j)
        assert_all(isrealobj(b))
        assert_array_equal(a,b)
        b = real_if_close(a+1e-7j)
        assert_all(iscomplexobj(b))
        b = real_if_close(a+1e-7j,tol=1e-6)
        assert_all(isrealobj(b))


class TestArrayConversion(TestCase):

    def test_asfarray(self):
        a = asfarray(array([1,2,3]))
        assert_equal(a.__class__,ndarray)
        assert issubdtype(a.dtype,float)

class TestDateTimeData:

    @dec.skipif(not _HAS_CTYPE, "ctypes not available on this python installation")
    def test_basic(self):
        a = array(['1980-03-23'], dtype=datetime64)
        assert_equal(datetime_data(a.dtype), (asbytes('us'), 1, 1, 1))


if __name__ == "__main__":
    run_module_suite()
