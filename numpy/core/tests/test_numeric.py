import sys
from numpy.core import *
from numpy.random import rand, randint, randn
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


class TestDot(TestCase):
    def setUp(self):
        self.A = rand(10,8)
        self.b1 = rand(8,1)
        self.b2 = rand(8)
        self.b3 = rand(1,8)
        self.b4 = rand(10)
        self.N = 14

    def test_matmat(self):
        A = self.A
        c1 = dot(A.transpose(), A)
        c2 = dot_(A.transpose(), A)
        assert_almost_equal(c1, c2, decimal=self.N)

    def test_matvec(self):
        A, b1 = self.A, self.b1
        c1 = dot(A, b1)
        c2 = dot_(A, b1)
        assert_almost_equal(c1, c2, decimal=self.N)

    def test_matvec2(self):
        A, b2 = self.A, self.b2
        c1 = dot(A, b2)
        c2 = dot_(A, b2)
        assert_almost_equal(c1, c2, decimal=self.N)

    def test_vecmat(self):
        A, b4 = self.A, self.b4
        c1 = dot(b4, A)
        c2 = dot_(b4, A)
        assert_almost_equal(c1, c2, decimal=self.N)

    def test_vecmat2(self):
        b3, A = self.b3, self.A
        c1 = dot(b3, A.transpose())
        c2 = dot_(b3, A.transpose())
        assert_almost_equal(c1, c2, decimal=self.N)

    def test_vecmat3(self):
        A, b4 = self.A, self.b4
        c1 = dot(A.transpose(),b4)
        c2 = dot_(A.transpose(),b4)
        assert_almost_equal(c1, c2, decimal=self.N)

    def test_vecvecouter(self):
        b1, b3 = self.b1, self.b3
        c1 = dot(b1, b3)
        c2 = dot_(b1, b3)
        assert_almost_equal(c1, c2, decimal=self.N)

    def test_vecvecinner(self):
        b1, b3 = self.b1, self.b3
        c1 = dot(b3, b1)
        c2 = dot_(b3, b1)
        assert_almost_equal(c1, c2, decimal=self.N)

    def test_matscalar(self):
        b1 = matrix(ones((3,3),dtype=complex))
        assert_equal(b1*1.0, b1)

    def test_columnvect1(self):
        b1 = ones((3,1))
        b2 = [5.3]
        c1 = dot(b1,b2)
        c2 = dot_(b1,b2)
        assert_almost_equal(c1, c2, decimal=self.N)

    def test_columnvect2(self):
        b1 = ones((3,1)).transpose()
        b2 = [6.2]
        c1 = dot(b2,b1)
        c2 = dot_(b2,b1)
        assert_almost_equal(c1, c2, decimal=self.N)

    def test_vecscalar(self):
        b1 = rand(1,1)
        b2 = rand(1,8)
        c1 = dot(b1,b2)
        c2 = dot_(b1,b2)
        assert_almost_equal(c1, c2, decimal=self.N)

    def test_vecscalar2(self):
        b1 = rand(8,1)
        b2 = rand(1,1)
        c1 = dot(b1,b2)
        c2 = dot_(b1,b2)
        assert_almost_equal(c1, c2, decimal=self.N)

    def test_all(self):
        dims = [(),(1,),(1,1)]
        for dim1 in dims:
            for dim2 in dims:
                arg1 = rand(*dim1)
                arg2 = rand(*dim2)
                c1 = dot(arg1, arg2)
                c2 = dot_(arg1, arg2)
                assert (c1.shape == c2.shape)
                assert_almost_equal(c1, c2, decimal=self.N)

    def test_vecobject(self):
        U_non_cont = transpose([[1.,1.],[1.,2.]])
        U_cont = ascontiguousarray(U_non_cont)
        x = array([Vec([1.,0.]),Vec([0.,1.])])
        zeros = array([Vec([0.,0.]),Vec([0.,0.])])
        zeros_test = dot(U_cont,x) - dot(U_non_cont,x)
        assert_equal(zeros[0].array, zeros_test[0].array)
        assert_equal(zeros[1].array, zeros_test[1].array)


class TestBoolScalar(TestCase):
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


class TestSeterr(TestCase):
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

    def test_divide_err(self):
        seterr(divide='raise')
        try:
            array([1.]) / array([0.])
        except FloatingPointError:
            pass
        else:
            self.fail()
        seterr(divide='ignore')
        array([1.]) / array([0.])


class TestFromiter(TestCase):
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


class TestIndex(TestCase):
    def test_boolean(self):
        a = rand(3,5,8)
        V = rand(5,8)
        g1 = randint(0,5,size=15)
        g2 = randint(0,8,size=15)
        V[g1,g2] = -V[g1,g2]
        assert (array([a[0][V>0],a[1][V>0],a[2][V>0]]) == a[:,V>0]).all()


class TestBinaryRepr(TestCase):
    def test_zero(self):
        assert_equal(binary_repr(0),'0')

    def test_large(self):
        assert_equal(binary_repr(10736848),'101000111101010011010000')

    def test_negative(self):
        assert_equal(binary_repr(-1), '-1')
        assert_equal(binary_repr(-1, width=8), '11111111')


class TestArrayComparisons(TestCase):
    def test_array_equal(self):
        res = array_equal(array([1,2]), array([1,2]))
        assert res
        assert type(res) is bool
        res = array_equal(array([1,2]), array([1,2,3]))
        assert not res
        assert type(res) is bool
        res = array_equal(array([1,2]), array([3,4]))
        assert not res
        assert type(res) is bool
        res = array_equal(array([1,2]), array([1,3]))
        assert not res
        assert type(res) is bool

    def test_array_equiv(self):
        res = array_equiv(array([1,2]), array([1,2]))
        assert res
        assert type(res) is bool
        res = array_equiv(array([1,2]), array([1,2,3]))
        assert not res
        assert type(res) is bool
        res = array_equiv(array([1,2]), array([3,4]))
        assert not res
        assert type(res) is bool
        res = array_equiv(array([1,2]), array([1,3]))
        assert not res
        assert type(res) is bool

        res = array_equiv(array([1,1]), array([1]))
        assert res
        assert type(res) is bool
        res = array_equiv(array([1,1]), array([[1],[1]]))
        assert res
        assert type(res) is bool
        res = array_equiv(array([1,2]), array([2]))
        assert not res
        assert type(res) is bool
        res = array_equiv(array([1,2]), array([[1],[2]]))
        assert not res
        assert type(res) is bool
        res = array_equiv(array([1,2]), array([[1,2,3],[4,5,6],[7,8,9]]))
        assert not res
        assert type(res) is bool


def assert_array_strict_equal(x, y):
    assert_array_equal(x, y)
    # Check flags
    assert x.flags == y.flags
    # check endianness
    assert x.dtype.isnative == y.dtype.isnative


class TestClip(TestCase):
    def setUp(self):
        self.nr = 5
        self.nc = 3

    def fastclip(self, a, m, M, out=None):
        if out is None:
            return a.clip(m,M)
        else:
            return a.clip(m,M,out)

    def clip(self, a, m, M, out=None):
        # use slow-clip
        selector = less(a, m)+2*greater(a, M)
        return selector.choose((a, m, M), out=out)

    # Handy functions
    def _generate_data(self, n, m):
        return randn(n, m)

    def _generate_data_complex(self, n, m):
        return randn(n, m) + 1.j *rand(n, m)

    def _generate_flt_data(self, n, m):
        return (randn(n, m)).astype(float32)

    def _neg_byteorder(self, a):
        a = asarray(a)
        if sys.byteorder == 'little':
            a = a.astype(a.dtype.newbyteorder('>'))
        else:
            a = a.astype(a.dtype.newbyteorder('<'))
        return a

    def _generate_non_native_data(self, n, m):
        data = randn(n, m)
        data = self._neg_byteorder(data)
        assert not data.dtype.isnative
        return data

    def _generate_int_data(self, n, m):
        return (10 * rand(n, m)).astype(int64)

    def _generate_int32_data(self, n, m):
        return (10 * rand(n, m)).astype(int32)

    # Now the real test cases
    def test_simple_double(self):
        """Test native double input with scalar min/max."""
        a   = self._generate_data(self.nr, self.nc)
        m   = 0.1
        M   = 0.6
        ac  = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_simple_int(self):
        """Test native int input with scalar min/max."""
        a   = self._generate_int_data(self.nr, self.nc)
        a   = a.astype(int)
        m   = -2
        M   = 4
        ac  = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_array_double(self):
        """Test native double input with array min/max."""
        a   = self._generate_data(self.nr, self.nc)
        m   = zeros(a.shape)
        M   = m + 0.5
        ac  = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_simple_nonnative(self):
        """Test non native double input with scalar min/max.
        Test native double input with non native double scalar min/max."""
        a   = self._generate_non_native_data(self.nr, self.nc)
        m   = -0.5
        M   = 0.6
        ac  = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

        "Test native double input with non native double scalar min/max."
        a   = self._generate_data(self.nr, self.nc)
        m   = -0.5
        M   = self._neg_byteorder(0.6)
        assert not M.dtype.isnative
        ac  = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

    def test_simple_complex(self):
        """Test native complex input with native double scalar min/max.
        Test native input with complex double scalar min/max.
        """
        a   = 3 * self._generate_data_complex(self.nr, self.nc)
        m   = -0.5
        M   = 1.
        ac  = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

        "Test native input with complex double scalar min/max."
        a   = 3 * self._generate_data(self.nr, self.nc)
        m   = -0.5 + 1.j
        M   = 1. + 2.j
        ac  = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_clip_non_contig(self):
        """Test clip for non contiguous native input and native scalar min/max."""
        a   = self._generate_data(self.nr * 2, self.nc * 3)
        a   = a[::2, ::3]
        assert not a.flags['F_CONTIGUOUS']
        assert not a.flags['C_CONTIGUOUS']
        ac  = self.fastclip(a, -1.6, 1.7)
        act = self.clip(a, -1.6, 1.7)
        assert_array_strict_equal(ac, act)

    def test_simple_out(self):
        """Test native double input with scalar min/max."""
        a   = self._generate_data(self.nr, self.nc)
        m   = -0.5
        M   = 0.6
        ac  = zeros(a.shape)
        act = zeros(a.shape)
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_simple_int32_inout(self):
        """Test native int32 input with double min/max and int32 out."""
        a   = self._generate_int32_data(self.nr, self.nc)
        m   = float64(0)
        M   = float64(2)
        ac  = zeros(a.shape, dtype = int32)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_simple_int64_out(self):
        """Test native int32 input with int32 scalar min/max and int64 out."""
        a   = self._generate_int32_data(self.nr, self.nc)
        m   = int32(-1)
        M   = int32(1)
        ac  = zeros(a.shape, dtype = int64)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_simple_int64_inout(self):
        """Test native in32 input with double array min/max and int32 out."""
        a   = self._generate_int32_data(self.nr, self.nc)
        m   = zeros(a.shape, float64)
        M   = float64(1)
        ac  = zeros(a.shape, dtype = int32)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_simple_int32_out(self):
        """Test native double input with scalar min/max and int out."""
        a   = self._generate_data(self.nr, self.nc)
        m   = -1.0
        M   = 2.0
        ac  = zeros(a.shape, dtype = int32)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_simple_inplace_01(self):
        """Test native double input with array min/max in-place."""
        a   = self._generate_data(self.nr, self.nc)
        ac  = a.copy()
        m   = zeros(a.shape)
        M   = 1.0
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_strict_equal(a, ac)

    def test_simple_inplace_02(self):
        """Test native double input with scalar min/max in-place."""
        a   = self._generate_data(self.nr, self.nc)
        ac  = a.copy()
        m   = -0.5
        M   = 0.6
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_strict_equal(a, ac)

    def test_noncontig_inplace(self):
        """Test non contiguous double input with double scalar min/max in-place."""
        a   = self._generate_data(self.nr * 2, self.nc * 3)
        a   = a[::2, ::3]
        assert not a.flags['F_CONTIGUOUS']
        assert not a.flags['C_CONTIGUOUS']
        ac  = a.copy()
        m   = -0.5
        M   = 0.6
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_equal(a, ac)

    def test_type_cast_01(self):
        "Test native double input with scalar min/max."
        a   = self._generate_data(self.nr, self.nc)
        m   = -0.5
        M   = 0.6
        ac  = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_02(self):
        "Test native int32 input with int32 scalar min/max."
        a   = self._generate_int_data(self.nr, self.nc)
        a   = a.astype(int32)
        m   = -2
        M   = 4
        ac  = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_03(self):
        "Test native int32 input with float64 scalar min/max."
        a   = self._generate_int32_data(self.nr, self.nc)
        m   = -2
        M   = 4
        ac  = self.fastclip(a, float64(m), float64(M))
        act = self.clip(a, float64(m), float64(M))
        assert_array_strict_equal(ac, act)

    def test_type_cast_04(self):
        "Test native int32 input with float32 scalar min/max."
        a   = self._generate_int32_data(self.nr, self.nc)
        m   = float32(-2)
        M   = float32(4)
        act = self.fastclip(a,m,M)
        ac  = self.clip(a,m,M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_05(self):
        "Test native int32 with double arrays min/max."
        a   = self._generate_int_data(self.nr, self.nc)
        m   = -0.5
        M   = 1.
        ac  = self.fastclip(a, m * zeros(a.shape), M)
        act = self.clip(a, m * zeros(a.shape), M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_06(self):
        "Test native with NON native scalar min/max."
        a   = self._generate_data(self.nr, self.nc)
        m   = 0.5
        m_s = self._neg_byteorder(m)
        M   = 1.
        act = self.clip(a, m_s, M)
        ac  = self.fastclip(a, m_s, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_07(self):
        "Test NON native with native array min/max."
        a   = self._generate_data(self.nr, self.nc)
        m   = -0.5 * ones(a.shape)
        M   = 1.
        a_s = self._neg_byteorder(a)
        assert not a_s.dtype.isnative
        act = a_s.clip(m, M)
        ac  = self.fastclip(a_s, m, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_08(self):
        "Test NON native with native scalar min/max."
        a   = self._generate_data(self.nr, self.nc)
        m   = -0.5
        M   = 1.
        a_s = self._neg_byteorder(a)
        assert not a_s.dtype.isnative
        ac  = self.fastclip(a_s, m , M)
        act = a_s.clip(m, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_09(self):
        "Test native with NON native array min/max."
        a   = self._generate_data(self.nr, self.nc)
        m   = -0.5 * ones(a.shape)
        M   = 1.
        m_s = self._neg_byteorder(m)
        assert not m_s.dtype.isnative
        ac  = self.fastclip(a, m_s , M)
        act = self.clip(a, m_s, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_10(self):
        """Test native int32 with float min/max and float out for output argument."""
        a   = self._generate_int_data(self.nr, self.nc)
        b   = zeros(a.shape, dtype = float32)
        m   = float32(-0.5)
        M   = float32(1)
        act = self.clip(a, m, M, out = b)
        ac  = self.fastclip(a, m , M, out = b)
        assert_array_strict_equal(ac, act)

    def test_type_cast_11(self):
        "Test non native with native scalar, min/max, out non native"
        a   = self._generate_non_native_data(self.nr, self.nc)
        b   = a.copy()
        b   = b.astype(b.dtype.newbyteorder('>'))
        bt  = b.copy()
        m   = -0.5
        M   = 1.
        self.fastclip(a, m , M, out = b)
        self.clip(a, m, M, out = bt)
        assert_array_strict_equal(b, bt)

    def test_type_cast_12(self):
        "Test native int32 input and min/max and float out"
        a   = self._generate_int_data(self.nr, self.nc)
        b   = zeros(a.shape, dtype = float32)
        m   = int32(0)
        M   = int32(1)
        act = self.clip(a, m, M, out = b)
        ac  = self.fastclip(a, m , M, out = b)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_simple(self):
        "Test native double input with scalar min/max"
        a   = self._generate_data(self.nr, self.nc)
        m   = -0.5
        M   = 0.6
        ac  = zeros(a.shape)
        act = zeros(a.shape)
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_simple2(self):
        "Test native int32 input with double min/max and int32 out"
        a   = self._generate_int32_data(self.nr, self.nc)
        m   = float64(0)
        M   = float64(2)
        ac  = zeros(a.shape, dtype = int32)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_simple_int32(self):
        "Test native int32 input with int32 scalar min/max and int64 out"
        a   = self._generate_int32_data(self.nr, self.nc)
        m   = int32(-1)
        M   = int32(1)
        ac  = zeros(a.shape, dtype = int64)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_array_int32(self):
        "Test native int32 input with double array min/max and int32 out"
        a   = self._generate_int32_data(self.nr, self.nc)
        m   = zeros(a.shape, float64)
        M   = float64(1)
        ac  = zeros(a.shape, dtype = int32)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_array_outint32(self):
        "Test native double input with scalar min/max and int out"
        a   = self._generate_data(self.nr, self.nc)
        m   = -1.0
        M   = 2.0
        ac  = zeros(a.shape, dtype = int32)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_inplace_array(self):
        "Test native double input with array min/max"
        a   = self._generate_data(self.nr, self.nc)
        ac  = a.copy()
        m   = zeros(a.shape)
        M   = 1.0
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_strict_equal(a, ac)

    def test_clip_inplace_simple(self):
        "Test native double input with scalar min/max"
        a   = self._generate_data(self.nr, self.nc)
        ac  = a.copy()
        m   = -0.5
        M   = 0.6
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_strict_equal(a, ac)

    def test_clip_func_takes_out(self):
        """ Ensure that the clip() function takes an out= argument.
        """
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = -0.5
        M = 0.6
        a2 = clip(a, m, M, out=a)
        self.clip(a, m, M, ac)
        assert_array_strict_equal(a2, ac)
        self.assert_(a2 is a)


class test_allclose_inf(TestCase):
    rtol = 1e-5
    atol = 1e-8

    def tst_allclose(self,x,y):
        assert allclose(x,y), "%s and %s not close" % (x,y)

    def tst_not_allclose(self,x,y):
        assert not allclose(x,y), "%s and %s shouldn't be close" % (x,y)

    def test_ip_allclose(self):
        """Parametric test factory."""
        arr = array([100,1000])
        aran = arange(125).reshape((5,5,5))

        atol = self.atol
        rtol = self.rtol

        data = [([1,0], [1,0]),
                ([atol], [0]),
                ([1], [1+rtol+atol]),
                (arr, arr + arr*rtol),
                (arr, arr + arr*rtol + atol*2),
                (aran, aran + aran*rtol),]

        for (x,y) in data:
            yield (self.tst_allclose,x,y)

    def test_ip_not_allclose(self):
        """Parametric test factory."""
        aran = arange(125).reshape((5,5,5))

        atol = self.atol
        rtol = self.rtol

        data = [([inf,0], [1,inf]),
                ([inf,0], [1,0]),
                ([inf,inf], [1,inf]),
                ([inf,inf], [1,0]),
                ([-inf, 0], [inf, 0]),
                ([nan,0], [nan,0]),
                ([atol*2], [0]),
                ([1], [1+rtol+atol*2]),
                (aran, aran + aran*atol + atol*2),
                (array([inf,1]), array([0,inf]))]

        for (x,y) in data:
            yield (self.tst_not_allclose,x,y)

    def test_no_parameter_modification(self):
        x = array([inf,1])
        y = array([0,inf])
        allclose(x,y)
        assert_array_equal(x,array([inf,1]))
        assert_array_equal(y,array([0,inf]))


class TestStdVar(TestCase):
    def setUp(self):
        self.A = array([1,-1,1,-1])
        self.real_var = 1

    def test_basic(self):
        assert_almost_equal(var(self.A),self.real_var)
        assert_almost_equal(std(self.A)**2,self.real_var)

    def test_ddof1(self):
        assert_almost_equal(var(self.A,ddof=1),
                            self.real_var*len(self.A)/float(len(self.A)-1))
        assert_almost_equal(std(self.A,ddof=1)**2,
                            self.real_var*len(self.A)/float(len(self.A)-1))

    def test_ddof2(self):
        assert_almost_equal(var(self.A,ddof=2),
                            self.real_var*len(self.A)/float(len(self.A)-2))
        assert_almost_equal(std(self.A,ddof=2)**2,
                            self.real_var*len(self.A)/float(len(self.A)-2))


class TestStdVarComplex(TestCase):
    def test_basic(self):
        A = array([1,1.j,-1,-1.j])
        real_var = 1
        assert_almost_equal(var(A),real_var)
        assert_almost_equal(std(A)**2,real_var)


if __name__ == "__main__":
    run_module_suite()

