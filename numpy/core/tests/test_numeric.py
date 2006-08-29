from numpy.core import *
from numpy.random import rand, randint
from numpy.testing import *
from numpy.core.multiarray import dot as dot_

class Vec:
    def __init__(self,sequence=None):
        if sequence is None:
            sequence=[]
        self.array=array(sequence)    
    def __add__(self,other):
        out=Vec()
        out.array=self.array+other.array
        return out
    def __sub__(self,other):
        out=Vec()
        out.array=self.array-other.array
        return out
    def __mul__(self,other): # with scalar
        out=Vec(self.array.copy())
        out.array*=other
        return out
    def __rmul__(self,other):
        return self*other
    def __abs__(self):
        out=Vec()
        out.array=abs(self.array)
        return out
    def __repr__(self):
        return "Vec("+repr(self.array.tolist())+")"
    __str__=__repr__

class test_dot(NumpyTestCase):
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

    def check_vecobject(self):
        U_non_cont = transpose([[1.,1.],[1.,2.]])
        U_cont = ascontiguousarray(U_non_cont)
        x = array([Vec([1.,0.]),Vec([0.,1.])])
        zeros = array([Vec([0.,0.]),Vec([0.,0.])])
        zeros_test = dot(U_cont,x) - dot(U_non_cont,x)
        assert_equal(zeros[0].array, zeros_test[0].array)
        assert_equal(zeros[1].array, zeros_test[1].array)


class test_bool_scalar(NumpyTestCase):
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


class test_seterr(NumpyTestCase):
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
        
        
class test_fromiter(NumpyTestCase):
    
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
        self.failUnless(alltrue(a == expected,axis=0))
        self.failUnless(alltrue(a20 == expected[:20],axis=0))

class test_index(NumpyTestCase):
    def test_boolean(self):
        a = rand(3,5,8)
        V = rand(5,8)
        g1 = randint(0,5,size=15)
        g2 = randint(0,8,size=15)
        V[g1,g2] = -V[g1,g2]
        assert (array([a[0][V>0],a[1][V>0],a[2][V>0]]) == a[:,V>0]).all()

class test_binary_repr(NumpyTestCase):
    def test_zero(self):
        assert_equal(binary_repr(0),'0')

    def test_large(self):
        assert_equal(binary_repr(10736848),'101000111101010011010000')

if __name__ == '__main__':
    NumpyTest().run()
