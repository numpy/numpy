import sys
from decimal import Decimal

import numpy as np
from numpy.core import *
from numpy.random import rand, randint, randn
from numpy.testing import *
from numpy.testing.utils import WarningManager
from numpy.core.multiarray import dot as dot_
import warnings

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


class TestResize(TestCase):
    def test_copies(self):
        A = array([[1,2],[3,4]])
        Ar1 = array([[1,2,3,4],[1,2,3,4]])
        assert_equal(resize(A, (2,4)), Ar1)

        Ar2 = array([[1,2],[3,4],[1,2],[3,4]])
        assert_equal(resize(A, (4,2)), Ar2)

        Ar3 = array([[1,2,3],[4,1,2],[3,4,1],[2,3,4]])
        assert_equal(resize(A, (4,3)), Ar3)

    def test_zeroresize(self):
        A = array([[1,2],[3,4]])
        Ar = resize(A, (0,))
        assert_equal(Ar, array([]))

class TestEinSum(TestCase):
    def test_einsum_errors(self):
        # Need enough arguments
        assert_raises(ValueError, np.einsum)
        assert_raises(ValueError, np.einsum, "")

        # subscripts must be a string
        assert_raises(TypeError, np.einsum, 0, 0)

        # out parameter must be an array
        assert_raises(TypeError, np.einsum, "", 0, out='test')

        # order parameter must be a valid order
        assert_raises(TypeError, np.einsum, "", 0, order='W')

        # casting parameter must be a valid casting
        assert_raises(ValueError, np.einsum, "", 0, casting='blah')

        # dtype parameter must be a valid dtype
        assert_raises(TypeError, np.einsum, "", 0, dtype='bad_data_type')

        # other keyword arguments are rejected
        assert_raises(TypeError, np.einsum, "", 0, bad_arg=0)

        # number of operands must match count in subscripts string
        assert_raises(ValueError, np.einsum, "", 0, 0)
        assert_raises(ValueError, np.einsum, ",", 0, [0], [0])
        assert_raises(ValueError, np.einsum, ",", [0])

        # can't have more subscripts than dimensions in the operand
        assert_raises(ValueError, np.einsum, "i", 0)
        assert_raises(ValueError, np.einsum, "ij", [0,0])
        assert_raises(ValueError, np.einsum, "...i", 0)
        assert_raises(ValueError, np.einsum, "i...j", [0,0])
        assert_raises(ValueError, np.einsum, "i...", 0)
        assert_raises(ValueError, np.einsum, "ij...", [0,0])

        # invalid ellipsis
        assert_raises(ValueError, np.einsum, "i..", [0,0])
        assert_raises(ValueError, np.einsum, ".i...", [0,0])
        assert_raises(ValueError, np.einsum, "j->..j", [0,0])
        assert_raises(ValueError, np.einsum, "j->.j...", [0,0])

        # invalid subscript character
        assert_raises(ValueError, np.einsum, "i%...", [0,0])
        assert_raises(ValueError, np.einsum, "...j$", [0,0])
        assert_raises(ValueError, np.einsum, "i->&", [0,0])

        # output subscripts must appear in input
        assert_raises(ValueError, np.einsum, "i->ij", [0,0])

        # output subscripts may only be specified once
        assert_raises(ValueError, np.einsum, "ij->jij", [[0,0],[0,0]])

        # dimensions much match when being collapsed
        assert_raises(ValueError, np.einsum, "ii", np.arange(6).reshape(2,3))
        assert_raises(ValueError, np.einsum, "ii->i", np.arange(6).reshape(2,3))

    def test_einsum_views(self):
        # pass-through
        a = np.arange(6).reshape(2,3)

        b = np.einsum("", a)
        assert_(b.base is a)

        b = np.einsum("ij", a)
        assert_(b.base is a)
        assert_equal(b, a)

        # transpose
        a = np.arange(6).reshape(2,3)

        b = np.einsum("ji", a)
        assert_(b.base is a)
        assert_equal(b, a.T)

        # diagonal
        a = np.arange(9).reshape(3,3)

        b = np.einsum("ii->i", a)
        assert_(b.base is a)
        assert_equal(b, [a[i,i] for i in range(3)])

        # diagonal with various ways of broadcasting an additional dimension
        a = np.arange(27).reshape(3,3,3)

        b = np.einsum("ii->i", a)
        assert_(b.base is a)
        assert_equal(b, [[x[i,i] for i in range(3)] for x in a])

        b = np.einsum("ii...->i", a)
        assert_(b.base is a)
        assert_equal(b, [[x[i,i] for i in range(3)]
                         for x in a.transpose(2,0,1)])

        b = np.einsum("ii->i...", a)
        assert_(b.base is a)
        assert_equal(b, [a[:,i,i] for i in range(3)])

        b = np.einsum("jii->ij", a)
        assert_(b.base is a)
        assert_equal(b, [a[:,i,i] for i in range(3)])

        b = np.einsum("ii...->i...", a)
        assert_(b.base is a)
        assert_equal(b, [a.transpose(2,0,1)[:,i,i] for i in range(3)])

        b = np.einsum("i...i->i...", a)
        assert_(b.base is a)
        assert_equal(b, [a.transpose(1,0,2)[:,i,i] for i in range(3)])

        b = np.einsum("i...i->i", a)
        assert_(b.base is a)
        assert_equal(b, [[x[i,i] for i in range(3)]
                         for x in a.transpose(1,0,2)])

        # triple diagonal
        a = np.arange(27).reshape(3,3,3)

        b = np.einsum("iii->i", a)
        assert_(b.base is a)
        assert_equal(b, [a[i,i,i] for i in range(3)])

        # swap axes
        a = np.arange(24).reshape(2,3,4)

        b = np.einsum("ijk->jik", a)
        assert_(b.base is a)
        assert_equal(b, a.swapaxes(0,1))

    def check_einsum_sums(self, dtype):
        # sum(a, axis=-1)
        a = np.arange(10, dtype=dtype)
        assert_equal(np.einsum("i->", a), np.sum(a, axis=-1))

        a = np.arange(24, dtype=dtype).reshape(2,3,4)
        assert_equal(np.einsum("i->", a), np.sum(a, axis=-1))

        # sum(a, axis=0)
        a = np.arange(10, dtype=dtype)
        assert_equal(np.einsum("i...->", a), np.sum(a, axis=0))

        a = np.arange(24, dtype=dtype).reshape(2,3,4)
        assert_equal(np.einsum("i...->", a), np.sum(a, axis=0))

        # trace(a)
        a = np.arange(25, dtype=dtype).reshape(5,5)
        assert_equal(np.einsum("ii", a), np.trace(a))

        # multiply(a, b)
        a = np.arange(12, dtype=dtype).reshape(3,4)
        b = np.arange(24, dtype=dtype).reshape(2,3,4)
        assert_equal(np.einsum(",", a, b), np.multiply(a, b))

        # inner(a,b)
        a = np.arange(24, dtype=dtype).reshape(2,3,4)
        b = np.arange(4, dtype=dtype)
        assert_equal(np.einsum("i,i", a, b), np.inner(a, b))

        a = np.arange(24, dtype=dtype).reshape(2,3,4)
        b = np.arange(2, dtype=dtype)
        assert_equal(np.einsum("i...,i...", a, b), np.inner(a.T, b.T).T)

        # outer(a,b)
        a = np.arange(3, dtype=dtype)+1
        b = np.arange(4, dtype=dtype)+1
        assert_equal(np.einsum("i,j", a, b), np.outer(a, b))

        # Suppress the complex warnings for the 'as f8' tests
        ctx = WarningManager()
        ctx.__enter__()
        try:
            warnings.simplefilter('ignore', np.ComplexWarning)

            # matvec(a,b) / a.dot(b) where a is matrix, b is vector
            a = np.arange(20, dtype=dtype).reshape(4,5)
            b = np.arange(5, dtype=dtype)
            assert_equal(np.einsum("ij,j", a, b), np.dot(a, b))

            a = np.arange(20, dtype=dtype).reshape(4,5)
            b = np.arange(5, dtype=dtype)
            c = np.arange(4, dtype=dtype)
            np.einsum("ij,j", a, b, out=c,
                        dtype='f8', casting='unsafe')
            assert_equal(c,
                        np.dot(a.astype('f8'), b.astype('f8')).astype(dtype))

            a = np.arange(20, dtype=dtype).reshape(4,5)
            b = np.arange(5, dtype=dtype)
            assert_equal(np.einsum("ji,j", a.T, b.T), np.dot(b.T, a.T))

            a = np.arange(20, dtype=dtype).reshape(4,5)
            b = np.arange(5, dtype=dtype)
            c = np.arange(4, dtype=dtype)
            np.einsum("ji,j", a.T, b.T, out=c, dtype='f8', casting='unsafe')
            assert_equal(c,
                    np.dot(b.T.astype('f8'), a.T.astype('f8')).astype(dtype))

            # matmat(a,b) / a.dot(b) where a is matrix, b is matrix
            a = np.arange(20, dtype=dtype).reshape(4,5)
            b = np.arange(30, dtype=dtype).reshape(5,6)
            assert_equal(np.einsum("ij,jk", a, b), np.dot(a, b))

            a = np.arange(20, dtype=dtype).reshape(4,5)
            b = np.arange(30, dtype=dtype).reshape(5,6)
            c = np.arange(24, dtype=dtype).reshape(4,6)
            np.einsum("ij,jk", a, b, out=c, dtype='f8', casting='unsafe')
            assert_equal(c,
                        np.dot(a.astype('f8'), b.astype('f8')).astype(dtype))

            # matrix triple product (note this is not an efficient
            # way to multiply 3 matrices)
            a = np.arange(12, dtype=dtype).reshape(3,4)
            b = np.arange(20, dtype=dtype).reshape(4,5)
            c = np.arange(30, dtype=dtype).reshape(5,6)
            if dtype != 'f2':
                assert_equal(np.einsum("ij,jk,kl", a, b, c),
                                    a.dot(b).dot(c))

            a = np.arange(12, dtype=dtype).reshape(3,4)
            b = np.arange(20, dtype=dtype).reshape(4,5)
            c = np.arange(30, dtype=dtype).reshape(5,6)
            d = np.arange(18, dtype=dtype).reshape(3,6)
            np.einsum("ij,jk,kl", a, b, c, out=d,
                                dtype='f8', casting='unsafe')
            assert_equal(d, a.astype('f8').dot(b.astype('f8')
                        ).dot(c.astype('f8')).astype(dtype))

            # tensordot(a, b)
            if np.dtype(dtype) != np.dtype('f2'):
                a = np.arange(60, dtype=dtype).reshape(3,4,5)
                b = np.arange(24, dtype=dtype).reshape(4,3,2)
                assert_equal(np.einsum("ijk,jil->kl", a, b),
                                np.tensordot(a,b, axes=([1,0],[0,1])))

                a = np.arange(60, dtype=dtype).reshape(3,4,5)
                b = np.arange(24, dtype=dtype).reshape(4,3,2)
                c = np.arange(10, dtype=dtype).reshape(5,2)
                np.einsum("ijk,jil->kl", a, b, out=c,
                                        dtype='f8', casting='unsafe')
                assert_equal(c, np.tensordot(a.astype('f8'), b.astype('f8'),
                                        axes=([1,0],[0,1])).astype(dtype))
        finally:
            ctx.__exit__()

        # logical_and(logical_and(a!=0, b!=0), c!=0)
        a = np.array([1,   3,   -2,   0,   12,  13,   0,   1], dtype=dtype)
        b = np.array([0,   3.5, 0.,   -2,  0,   1,    3,   12], dtype=dtype)
        c = np.array([True,True,False,True,True,False,True,True])
        assert_equal(np.einsum("i,i,i->i", a, b, c,
                                dtype='?', casting='unsafe'),
                            logical_and(logical_and(a!=0, b!=0), c!=0))

        a = np.arange(9, dtype=dtype)
        assert_equal(np.einsum(",i->", 3, a), 3*np.sum(a))
        assert_equal(np.einsum("i,->", a, 3), 3*np.sum(a))

        # Various stride0, contiguous, and SSE aligned variants
        a = np.arange(64, dtype=dtype)
        if np.dtype(dtype).itemsize > 1:
            assert_equal(np.einsum(",",a,a), np.multiply(a,a))
            assert_equal(np.einsum("i,i", a, a), np.dot(a,a))
            assert_equal(np.einsum("i,->i", a, 2), 2*a)
            assert_equal(np.einsum(",i->i", 2, a), 2*a)
            assert_equal(np.einsum("i,->", a, 2), 2*np.sum(a))
            assert_equal(np.einsum(",i->", 2, a), 2*np.sum(a))

            assert_equal(np.einsum(",",a[1:],a[:-1]), np.multiply(a[1:],a[:-1]))
            assert_equal(np.einsum("i,i", a[1:], a[:-1]), np.dot(a[1:],a[:-1]))
            assert_equal(np.einsum("i,->i", a[1:], 2), 2*a[1:])
            assert_equal(np.einsum(",i->i", 2, a[1:]), 2*a[1:])
            assert_equal(np.einsum("i,->", a[1:], 2), 2*np.sum(a[1:]))
            assert_equal(np.einsum(",i->", 2, a[1:]), 2*np.sum(a[1:]))

        # An object array, summed as the data type
        a = np.arange(9, dtype=object)
        b = np.einsum("i->", a, dtype=dtype, casting='unsafe')
        assert_equal(b, np.sum(a))
        assert_equal(b.dtype, np.dtype(dtype))

    def test_einsum_sums_int8(self):
        self.check_einsum_sums('i1');

    def test_einsum_sums_uint8(self):
        self.check_einsum_sums('u1');

    def test_einsum_sums_int16(self):
        self.check_einsum_sums('i2');

    def test_einsum_sums_uint16(self):
        self.check_einsum_sums('u2');

    def test_einsum_sums_int32(self):
        self.check_einsum_sums('i4');

    def test_einsum_sums_uint32(self):
        self.check_einsum_sums('u4');

    def test_einsum_sums_int64(self):
        self.check_einsum_sums('i8');

    def test_einsum_sums_uint64(self):
        self.check_einsum_sums('u8');

    def test_einsum_sums_float16(self):
        self.check_einsum_sums('f2');

    def test_einsum_sums_float32(self):
        self.check_einsum_sums('f4');

    def test_einsum_sums_float64(self):
        self.check_einsum_sums('f8');

    def test_einsum_sums_longdouble(self):
        self.check_einsum_sums(np.longdouble);

    def test_einsum_sums_cfloat64(self):
        self.check_einsum_sums('c8');

    def test_einsum_sums_cfloat128(self):
        self.check_einsum_sums('c16');

    def test_einsum_sums_clongdouble(self):
        self.check_einsum_sums(np.clongdouble);

class TestNonarrayArgs(TestCase):
    # check that non-array arguments to functions wrap them in arrays
    def test_squeeze(self):
        A = [[[1,1,1],[2,2,2],[3,3,3]]]
        assert squeeze(A).shape == (3,3)

    def test_cumproduct(self):
        A = [[1,2,3],[4,5,6]]
        assert all(cumproduct(A) == array([1,2,6,24,120,720]))

    def test_size(self):
        A = [[1,2,3],[4,5,6]]
        assert size(A) == 6
        assert size(A,0) == 2
        assert size(A,1) == 3

    def test_mean(self):
        A = [[1,2,3],[4,5,6]]
        assert mean(A) == 3.5
        assert all(mean(A,0) == array([2.5,3.5,4.5]))
        assert all(mean(A,1) == array([2.,5.]))

    def test_std(self):
        A = [[1,2,3],[4,5,6]]
        assert_almost_equal(std(A), 1.707825127659933)
        assert_almost_equal(std(A,0), array([1.5, 1.5, 1.5]))
        assert_almost_equal(std(A,1), array([0.81649658, 0.81649658]))

    def test_var(self):
        A = [[1,2,3],[4,5,6]]
        assert_almost_equal(var(A), 2.9166666666666665)
        assert_almost_equal(var(A,0), array([2.25, 2.25, 2.25]))
        assert_almost_equal(var(A,1), array([0.66666667, 0.66666667]))


class TestBoolScalar(TestCase):
    def test_logical(self):
        f = False_
        t = True_
        s = "xyz"
        self.assertTrue((t and s) is s)
        self.assertTrue((f and s) is f)

    def test_bitwise_or(self):
        f = False_
        t = True_
        self.assertTrue((t | t) is t)
        self.assertTrue((f | t) is t)
        self.assertTrue((t | f) is t)
        self.assertTrue((f | f) is f)

    def test_bitwise_and(self):
        f = False_
        t = True_
        self.assertTrue((t & t) is t)
        self.assertTrue((f & t) is f)
        self.assertTrue((t & f) is f)
        self.assertTrue((f & f) is f)

    def test_bitwise_xor(self):
        f = False_
        t = True_
        self.assertTrue((t ^ t) is f)
        self.assertTrue((f ^ t) is t)
        self.assertTrue((t ^ f) is t)
        self.assertTrue((f ^ f) is f)


class TestSeterr(TestCase):
    def test_set(self):
        err = seterr()
        try:
            old = seterr(divide='warn')
            self.assertTrue(err == old)
            new = seterr()
            self.assertTrue(new['divide'] == 'warn')
            seterr(over='raise')
            self.assertTrue(geterr()['over'] == 'raise')
            self.assertTrue(new['divide'] == 'warn')
            seterr(**old)
            self.assertTrue(geterr() == old)
        finally:
            seterr(**err)

    def test_divide_err(self):
        err = seterr(divide='raise')
        try:
            try:
                array([1.]) / array([0.])
            except FloatingPointError:
                pass
            else:
                self.fail()
            seterr(divide='ignore')
            array([1.]) / array([0.])
        finally:
            seterr(**err)


class TestFloatExceptions(TestCase):
    def assert_raises_fpe(self, fpeerr, flop, x, y):
        ftype = type(x)
        try:
            flop(x, y)
            assert_(False,
                    "Type %s did not raise fpe error '%s'." % (ftype, fpeerr))
        except FloatingPointError, exc:
            assert_(str(exc).find(fpeerr) >= 0,
                    "Type %s raised wrong fpe error '%s'." % (ftype, exc))

    def assert_op_raises_fpe(self, fpeerr, flop, sc1, sc2):
        """Check that fpe exception is raised.

       Given a floating operation `flop` and two scalar values, check that
       the operation raises the floating point exception specified by
       `fpeerr`. Tests all variants with 0-d array scalars as well.

        """
        self.assert_raises_fpe(fpeerr, flop, sc1, sc2);
        self.assert_raises_fpe(fpeerr, flop, sc1[()], sc2);
        self.assert_raises_fpe(fpeerr, flop, sc1, sc2[()]);
        self.assert_raises_fpe(fpeerr, flop, sc1[()], sc2[()]);

    def test_floating_exceptions(self):
        """Test basic arithmetic function errors"""
        oldsettings = np.seterr(all='raise')
        try:
            # Test for all real and complex float types
            for typecode in np.typecodes['AllFloat']:
                ftype = np.obj2sctype(typecode)
                if np.dtype(ftype).kind == 'f':
                    # Get some extreme values for the type
                    fi = np.finfo(ftype)
                    ft_tiny = fi.tiny
                    ft_max = fi.max
                    ft_eps = fi.eps
                    underflow = 'underflow'
                    divbyzero = 'divide by zero'
                else:
                    # 'c', complex, corresponding real dtype
                    rtype = type(ftype(0).real)
                    fi = np.finfo(rtype)
                    ft_tiny = ftype(fi.tiny)
                    ft_max = ftype(fi.max)
                    ft_eps = ftype(fi.eps)
                    # The complex types raise different exceptions
                    underflow = ''
                    divbyzero = ''
                overflow = 'overflow'
                invalid = 'invalid'

                self.assert_raises_fpe(underflow,
                        lambda a,b:a/b, ft_tiny, ft_max)
                self.assert_raises_fpe(underflow,
                        lambda a,b:a*b, ft_tiny, ft_tiny)
                self.assert_raises_fpe(overflow,
                        lambda a,b:a*b, ft_max, ftype(2))
                self.assert_raises_fpe(overflow,
                        lambda a,b:a/b, ft_max, ftype(0.5))
                self.assert_raises_fpe(overflow,
                        lambda a,b:a+b, ft_max, ft_max*ft_eps)
                self.assert_raises_fpe(overflow,
                        lambda a,b:a-b, -ft_max, ft_max*ft_eps)
                self.assert_raises_fpe(overflow,
                        np.power, ftype(2), ftype(2**fi.nexp))
                self.assert_raises_fpe(divbyzero,
                        lambda a,b:a/b, ftype(1), ftype(0))
                self.assert_raises_fpe(invalid,
                        lambda a,b:a/b, ftype(np.inf), ftype(np.inf))
                self.assert_raises_fpe(invalid,
                        lambda a,b:a/b, ftype(0), ftype(0))
                self.assert_raises_fpe(invalid,
                        lambda a,b:a-b, ftype(np.inf), ftype(np.inf))
                self.assert_raises_fpe(invalid,
                        lambda a,b:a+b, ftype(np.inf), ftype(-np.inf))
                self.assert_raises_fpe(invalid,
                        lambda a,b:a*b, ftype(0), ftype(np.inf))
        finally:
            np.seterr(**oldsettings)

class TestTypes(TestCase):
    def check_promotion_cases(self, promote_func):
        """Tests that the scalars get coerced correctly."""
        i8, i16, i32, i64 = int8(0), int16(0), int32(0), int64(0)
        u8, u16, u32, u64 = uint8(0), uint16(0), uint32(0), uint64(0)
        f32, f64, fld = float32(0), float64(0), longdouble(0)
        c64, c128, cld = complex64(0), complex128(0), clongdouble(0)

        # coercion within the same type
        assert_equal(promote_func(i8,i16), np.dtype(int16))
        assert_equal(promote_func(i32,i8), np.dtype(int32))
        assert_equal(promote_func(i16,i64), np.dtype(int64))
        assert_equal(promote_func(u8,u32), np.dtype(uint32))
        assert_equal(promote_func(f32,f64), np.dtype(float64))
        assert_equal(promote_func(fld,f32), np.dtype(longdouble))
        assert_equal(promote_func(f64,fld), np.dtype(longdouble))
        assert_equal(promote_func(c128,c64), np.dtype(complex128))
        assert_equal(promote_func(cld,c128), np.dtype(clongdouble))
        assert_equal(promote_func(c64,fld), np.dtype(clongdouble))

        # coercion between types
        assert_equal(promote_func(i8,u8), np.dtype(int16))
        assert_equal(promote_func(u8,i32), np.dtype(int32))
        assert_equal(promote_func(i64,u32), np.dtype(int64))
        assert_equal(promote_func(u64,i32), np.dtype(float64))
        assert_equal(promote_func(i32,f32), np.dtype(float64))
        assert_equal(promote_func(i64,f32), np.dtype(float64))
        assert_equal(promote_func(f32,i16), np.dtype(float32))
        assert_equal(promote_func(f32,u32), np.dtype(float64))
        assert_equal(promote_func(f32,c64), np.dtype(complex64))
        assert_equal(promote_func(c128,f32), np.dtype(complex128))
        assert_equal(promote_func(cld,f64), np.dtype(clongdouble))

        # coercion between scalars and 1-D arrays
        assert_equal(promote_func(array([i8]),i64), np.dtype(int8))
        assert_equal(promote_func(u64,array([i32])), np.dtype(int32))
        assert_equal(promote_func(i64,array([u32])), np.dtype(uint32))
        assert_equal(promote_func(int32(-1),array([u64])), np.dtype(float64))
        assert_equal(promote_func(f64,array([f32])), np.dtype(float32))
        assert_equal(promote_func(fld,array([f32])), np.dtype(float32))
        assert_equal(promote_func(array([f64]),fld), np.dtype(float64))
        assert_equal(promote_func(fld,array([c64])), np.dtype(complex64))

    def test_coercion(self):
        def res_type(a, b):
            return np.add(a, b).dtype

        ctx = WarningManager()
        ctx.__enter__()
        warnings.simplefilter('ignore', np.ComplexWarning)

        self.check_promotion_cases(res_type)

        f64 = float64(0)
        c64 = complex64(0)
        ## Scalars do not coerce to complex if the value is real
        #assert_equal(res_type(c64,array([f64])), np.dtype(float64))
        # But they do if the value is complex
        assert_equal(res_type(complex64(3j),array([f64])),
                                                    np.dtype(complex128))

        # Scalars do coerce to complex even if the value is real
        # This is so "a+0j" can be reliably used to make something complex.
        assert_equal(res_type(c64,array([f64])), np.dtype(complex128))

        ctx.__exit__()


    def test_result_type(self):
        self.check_promotion_cases(np.result_type)

        f64 = float64(0)
        c64 = complex64(0)
        ## Scalars do not coerce to complex if the value is real
        #assert_equal(np.result_type(c64,array([f64])), np.dtype(float64))
        # But they do if the value is complex
        assert_equal(np.result_type(complex64(3j),array([f64])),
                                                    np.dtype(complex128))

        # Scalars do coerce to complex even if the value is real
        # This is so "a+0j" can be reliably used to make something complex.
        assert_equal(np.result_type(c64,array([f64])), np.dtype(complex128))


    def can_cast(self):
        assert_(np.can_cast(np.int32, np.int64))
        assert_(np.can_cast(np.float64, np.complex))
        assert_(not np.can_cast(np.complex, np.float))
        
        assert_(np.can_cast('i8', 'f8'))
        assert_(not np.can_cast('i8', 'f4'))
        assert_(np.can_cast('i4', 'S4'))

        assert_(np.can_cast('i8', 'i8', 'no'))
        assert_(not np.can_cast('<i8', '>i8', 'no'))

        assert_(np.can_cast('<i8', '>i8', 'equiv'))
        assert_(not np.can_cast('<i4', '>i8', 'equiv'))

        assert_(np.can_cast('<i4', '>i8', 'safe'))
        assert_(not np.can_cast('<i8', '>i4', 'safe'))

        assert_(np.can_cast('<i8', '>i4', 'same_kind'))
        assert_(not np.can_cast('<i8', '>u4', 'same_kind'))

        assert_(np.can_cast('<i8', '>u4', 'unsafe'))

        assert_raises(TypeError, np.can_cast, 'i4', None)
        assert_raises(TypeError, np.can_cast, None, 'i4')

class TestFromiter(TestCase):
    def makegen(self):
        for x in xrange(24):
            yield x**2

    def test_types(self):
        ai32 = fromiter(self.makegen(), int32)
        ai64 = fromiter(self.makegen(), int64)
        af = fromiter(self.makegen(), float)
        self.assertTrue(ai32.dtype == dtype(int32))
        self.assertTrue(ai64.dtype == dtype(int64))
        self.assertTrue(af.dtype == dtype(float))

    def test_lengths(self):
        expected = array(list(self.makegen()))
        a = fromiter(self.makegen(), int)
        a20 = fromiter(self.makegen(), int, 20)
        self.assertTrue(len(a) == len(expected))
        self.assertTrue(len(a20) == 20)
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
        self.assertTrue(alltrue(a == expected,axis=0))
        self.assertTrue(alltrue(a20 == expected[:20],axis=0))

class TestNonzero(TestCase):
    def test_nonzero_trivial(self):
        assert_equal(np.count_nonzero(array([])), 0)
        assert_equal(np.nonzero(array([])), ([],))

        assert_equal(np.count_nonzero(array(0)), 0)
        assert_equal(np.nonzero(array(0)), ([],))
        assert_equal(np.count_nonzero(array(1)), 1)
        assert_equal(np.nonzero(array(1)), ([0],))

    def test_nonzero_onedim(self):
        x = array([1,0,2,-1,0,0,8])
        assert_equal(np.count_nonzero(x), 4)
        assert_equal(np.nonzero(x), ([0, 2, 3, 6],))

        x = array([(1,2),(0,0),(1,1),(-1,3),(0,7)],
                            dtype=[('a','i4'),('b','i2')])
        assert_equal(np.count_nonzero(x['a']), 3)
        assert_equal(np.count_nonzero(x['b']), 4)
        assert_equal(np.nonzero(x['a']), ([0,2,3],))
        assert_equal(np.nonzero(x['b']), ([0,2,3,4],))

    def test_nonzero_twodim(self):
        x = array([[0,1,0],[2,0,3]])
        assert_equal(np.count_nonzero(x), 3)
        assert_equal(np.nonzero(x), ([0,1,1],[1,0,2]))

        x = np.eye(3)
        assert_equal(np.count_nonzero(x), 3)
        assert_equal(np.nonzero(x), ([0,1,2],[0,1,2]))

        x = array([[(0,1),(0,0),(1,11)],
                   [(1,1),(1,0),(0,0)],
                   [(0,0),(1,5),(0,1)]], dtype=[('a','f4'),('b','u1')])
        assert_equal(np.count_nonzero(x['a']), 4)
        assert_equal(np.count_nonzero(x['b']), 5)
        assert_equal(np.nonzero(x['a']), ([0,1,1,2],[2,0,1,1]))
        assert_equal(np.nonzero(x['b']), ([0,0,1,2,2],[0,2,0,1,2]))

        assert_equal(np.count_nonzero(x['a'].T), 4)
        assert_equal(np.count_nonzero(x['b'].T), 5)
        assert_equal(np.nonzero(x['a'].T), ([0,1,1,2],[1,1,2,0]))
        assert_equal(np.nonzero(x['b'].T), ([0,0,1,2,2],[0,1,2,0,2]))

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

class TestBaseRepr(TestCase):
    def test_base3(self):
        assert_equal(base_repr(3**5, 3), '100000')

    def test_positive(self):
        assert_equal(base_repr(12, 10), '12')
        assert_equal(base_repr(12, 10, 4), '000012')
        assert_equal(base_repr(12, 4), '30')
        assert_equal(base_repr(3731624803700888, 36), '10QR0ROFCEW')

    def test_negative(self):
        assert_equal(base_repr(-12, 10), '-12')
        assert_equal(base_repr(-12, 10, 4), '-000012')
        assert_equal(base_repr(-12, 4), '-30')

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
        """Test native int32 input with double array min/max and int32 out."""
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


class TestLikeFuncs(TestCase):
    '''Test zeros_like and empty_like'''

    def setUp(self):
        self.data = [(array([[1,2,3],[4,5,6]],dtype=int32), (2,3), int32),
                     (array([[1,2,3],[4,5,6]],dtype=float32), (2,3), float32),
                     ]

    def test_zeros_like(self):
        for d, dshape, dtype in self.data:
            dz = zeros_like(d)
            assert dz.shape == dshape
            assert dz.dtype.type == dtype
            assert all(abs(dz) == 0)

    def test_empty_like(self):
        for d, dshape, dtype in self.data:
            dz = zeros_like(d)
            assert dz.shape == dshape
            assert dz.dtype.type == dtype

class _TestCorrelate(TestCase):
    def _setup(self, dt):
        self.x = np.array([1, 2, 3, 4, 5], dtype=dt)
        self.y = np.array([-1, -2, -3], dtype=dt)
        self.z1 = np.array([ -3.,  -8., -14., -20., -26., -14.,  -5.], dtype=dt)
        self.z2 = np.array([ -5.,  -14., -26., -20., -14., -8.,  -3.], dtype=dt)

    def test_float(self):
        self._setup(np.float)
        z = np.correlate(self.x, self.y, 'full', old_behavior=self.old_behavior)
        assert_array_almost_equal(z, self.z1)
        z = np.correlate(self.y, self.x, 'full', old_behavior=self.old_behavior)
        assert_array_almost_equal(z, self.z2)

    def test_object(self):
        self._setup(Decimal)
        z = np.correlate(self.x, self.y, 'full', old_behavior=self.old_behavior)
        assert_array_almost_equal(z, self.z1)
        z = np.correlate(self.y, self.x, 'full', old_behavior=self.old_behavior)
        assert_array_almost_equal(z, self.z2)

class TestCorrelate(_TestCorrelate):
    old_behavior = True
    def _setup(self, dt):
        # correlate uses an unconventional definition so that correlate(a, b)
        # == correlate(b, a), so force the corresponding outputs to be the same
        # as well
        _TestCorrelate._setup(self, dt)
        self.z2 = self.z1

    @dec.deprecated()
    def test_complex(self):
        x = np.array([1, 2, 3, 4+1j], dtype=np.complex)
        y = np.array([-1, -2j, 3+1j], dtype=np.complex)
        r_z = np.array([3+1j, 6, 8-1j, 9+1j, -1-8j, -4-1j], dtype=np.complex)
        z = np.correlate(x, y, 'full', old_behavior=self.old_behavior)
        assert_array_almost_equal(z, r_z)

    @dec.deprecated()
    def test_float(self):
        _TestCorrelate.test_float(self)

    @dec.deprecated()
    def test_object(self):
        _TestCorrelate.test_object(self)

class TestCorrelateNew(_TestCorrelate):
    old_behavior = False
    def test_complex(self):
        x = np.array([1, 2, 3, 4+1j], dtype=np.complex)
        y = np.array([-1, -2j, 3+1j], dtype=np.complex)
        r_z = np.array([3-1j, 6, 8+1j, 11+5j, -5+8j, -4-1j], dtype=np.complex)
        #z = np.acorrelate(x, y, 'full')
        #assert_array_almost_equal(z, r_z)

        r_z = r_z[::-1].conjugate()
        z = np.correlate(y, x, 'full', old_behavior=self.old_behavior)
        assert_array_almost_equal(z, r_z)

class TestArgwhere:
    def test_2D(self):
        x = np.arange(6).reshape((2, 3))
        assert_array_equal(np.argwhere(x > 1),
                           [[0, 2],
                            [1, 0],
                            [1, 1],
                            [1, 2]])

    def test_list(self):
        assert_equal(np.argwhere([4, 0, 2, 1, 3]), [[0], [2], [3], [4]])

class TestStringFunction:
    def test_set_string_function(self):
        a = np.array([1])
        np.set_string_function(lambda x: "FOO", repr=True)
        assert_equal(repr(a), "FOO")
        np.set_string_function(None, repr=True)
        assert_equal(repr(a), "array([1])")

        np.set_string_function(lambda x: "FOO", repr=False)
        assert_equal(str(a), "FOO")
        np.set_string_function(None, repr=False)
        assert_equal(str(a), "[1]")

if __name__ == "__main__":
    run_module_suite()
