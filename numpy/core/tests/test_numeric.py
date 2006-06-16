from numpy.core import *
from numpy.random import rand
from numpy.testing import *
from numpy.core.multiarray import dot as dot_

class test_dot(ScipyTestCase):
    def setUp(self):
        self.A = rand(10,8)
        self.b1 = rand(8,1)
        self.b2 = rand(8)
        self.b3 = rand(1,8)
        self.b4 = rand(10)
        self.N = 14

    def check_matmat(self):
        A = self.A
        c1 = dot(A.transpose(), A)
        c2 = dot_(A.transpose(), A)
        assert_almost_equal(c1, c2, decimal=self.N)

    def check_matvec(self):
        A, b1 = self.A, self.b1
        c1 = dot(A, b1)
        c2 = dot_(A, b1)
        assert_almost_equal(c1, c2, decimal=self.N)

    def check_matvec2(self):
        A, b2 = self.A, self.b2
        c1 = dot(A, b2)
        c2 = dot_(A, b2)
        assert_almost_equal(c1, c2, decimal=self.N)

    def check_vecmat(self):
        A, b4 = self.A, self.b4
        c1 = dot(b4, A)
        c2 = dot_(b4, A)
        assert_almost_equal(c1, c2, decimal=self.N)

    def check_vecmat2(self):
        b3, A = self.b3, self.A
        c1 = dot(b3, A.transpose())
        c2 = dot_(b3, A.transpose())
        assert_almost_equal(c1, c2, decimal=self.N)

    def check_vecmat3(self):
        A, b4 = self.A, self.b4
        c1 = dot(A.transpose(),b4)
        c2 = dot_(A.transpose(),b4)
        assert_almost_equal(c1, c2, decimal=self.N)

    def check_vecvecouter(self):
        b1, b3 = self.b1, self.b3
        c1 = dot(b1, b3)
        c2 = dot_(b1, b3)
        assert_almost_equal(c1, c2, decimal=self.N)

    def check_vecvecinner(self):
        b1, b3 = self.b1, self.b3
        c1 = dot(b3, b1)
        c2 = dot_(b3, b1)
        assert_almost_equal(c1, c2, decimal=self.N)

    def check_matscalar(self):
        b1 = matrix(ones((3,3),dtype=complex))
        assert_equal(b1*1.0, b1)

    def check_columnvect(self):
        b1 = ones((3,1))
        b2 = [5.3]
        c1 = dot(b1,b2)
        c2 = dot_(b1,b2)
        assert_almost_equal(c1, c2, decimal=self.N)

    def check_columnvect(self):
        b1 = ones((3,1)).transpose()
        b2 = [6.2]
        c1 = dot(b2,b1)
        c2 = dot_(b2,b1)
        assert_almost_equal(c1, c2, decimal=self.N)

    def check_vecscalar(self):
        b1 = rand(1,1)
        b2 = rand(1,8)
        c1 = dot(b1,b2)
        c2 = dot_(b1,b2)
        assert_almost_equal(c1, c2, decimal=self.N)

    def check_vecscalar2(self):
        b1 = rand(8,1)
        b2 = rand(1,1)
        c1 = dot(b1,b2)
        c2 = dot_(b1,b2)
        assert_almost_equal(c1, c2, decimal=self.N)        

    def check_all(self):
        dims = [(),(1,),(1,1)]
        for dim1 in dims:
            for dim2 in dims:
                arg1 = rand(*dim1)
                arg2 = rand(*dim2)
                c1 = dot(arg1, arg2)
                c2 = dot_(arg1, arg2)
                assert (c1.shape == c2.shape)
                assert_almost_equal(c1, c2, decimal=self.N)


class test_bool_scalar(ScipyTestCase):
    def test_logical(self):
        f = False_
        t = True_
        s = "xyz"
        self.failUnless((t and s) is s)
        self.failUnless((f and s) is f)

    def test_bitwise_or(self):
        f = False_
        t = True_
        self.failUnless((t | t) is t)
        self.failUnless((f | t) is t)
        self.failUnless((t | f) is t)
        self.failUnless((f | f) is f)

    def test_bitwise_and(self):
        f = False_
        t = True_
        self.failUnless((t & t) is t)
        self.failUnless((f & t) is f)
        self.failUnless((t & f) is f)
        self.failUnless((f & f) is f)

    def test_bitwise_xor(self):
        f = False_
        t = True_
        self.failUnless((t ^ t) is f)
        self.failUnless((f ^ t) is t)
        self.failUnless((t ^ f) is t)
        self.failUnless((f ^ f) is f)


class test_seterr(ScipyTestCase):
    def test_set(self):
        err = seterr()
        old = seterr(divide='warn')
        self.failUnless(err == old)
        new = seterr()
        self.failUnless(new['divide'] == 'warn')
        seterr(over='raise')
        self.failUnless(geterr()['over'] == 'raise')
        self.failUnless(new['divide'] == 'warn')
        seterr(**old)
        self.failUnless(geterr() == old)
    def test_divideerr(self):
        seterr(divide='raise')
        try:
            array([1.]) / array([0.])
        except FloatingPointError:
            pass
        else:
            self.fail()
        seterr(divide='ignore')
        array([1.]) / array([0.])
        
        
class test_fromiter(ScipyTestCase):
    
    def makegen(self):
        for x in xrange(24):
            yield x**2
            
    def test_types(self):
        ai32 = fromiter(self.makegen(), int32)
        ai64 = fromiter(self.makegen(), int64)
        af = fromiter(self.makegen(), float)
        self.failUnless(ai32.dtype == dtype(int32))
        self.failUnless(ai64.dtype == dtype(int64))
        self.failUnless(af.dtype == dtype(float))

    def test_lengths(self):
        expected = array(list(self.makegen()))
        a = fromiter(self.makegen(), int)
        a20 = fromiter(self.makegen(), int, 20)
        self.failUnless(len(a) == len(expected))
        self.failUnless(len(a20) == 20)
        try:
            fromiter(self.makegen(), int, len(expected) + 10)
        except ValueError:
            pass
        else:
            self.fail()

    def test_values(self):
        expected = array(list(self.makegen()))
        a = fromiter(self.makegen(), int)
        a20 = fromiter(self.makegen(), int, 20)
        self.failUnless(alltrue(a == expected))
        self.failUnless(alltrue(a20 == expected[:20]))

if __name__ == '__main__':
    NumpyTest().run()
