import sys
import platform
from decimal import Decimal

import numpy as np
from numpy.core import *
from numpy.random import rand, randint, randn
from numpy.testing import *
from numpy.core.multiarray import dot as dot_


class Vec(object):
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
                assert_(c1.shape == c2.shape)
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

class TestNonarrayArgs(TestCase):
    # check that non-array arguments to functions wrap them in arrays
    def test_squeeze(self):
        A = [[[1,1,1],[2,2,2],[3,3,3]]]
        assert_(squeeze(A).shape == (3,3))

    def test_cumproduct(self):
        A = [[1,2,3],[4,5,6]]
        assert_(all(cumproduct(A) == array([1,2,6,24,120,720])))

    def test_size(self):
        A = [[1,2,3],[4,5,6]]
        assert_(size(A) == 6)
        assert_(size(A,0) == 2)
        assert_(size(A,1) == 3)

    def test_mean(self):
        A = [[1,2,3],[4,5,6]]
        assert_(mean(A) == 3.5)
        assert_(all(mean(A,0) == array([2.5,3.5,4.5])))
        assert_(all(mean(A,1) == array([2.,5.])))

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
    def test_default(self):
        err = geterr()
        self.assertEqual(err, dict(
            divide='warn',
            invalid='warn',
            over='warn',
            under='ignore',
        ))

    def test_set(self):
        err = seterr()
        try:
            old = seterr(divide='print')
            self.assertTrue(err == old)
            new = seterr()
            self.assertTrue(new['divide'] == 'print')
            seterr(over='raise')
            self.assertTrue(geterr()['over'] == 'raise')
            self.assertTrue(new['divide'] == 'print')
            seterr(**old)
            self.assertTrue(geterr() == old)
        finally:
            seterr(**err)

    @dec.skipif(platform.machine() == "armv5tel", "See gh-413.")
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

    @dec.knownfailureif(True, "See ticket 1755")
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
        #Tests that the scalars get coerced correctly.
        b = np.bool_(0)
        i8, i16, i32, i64 = int8(0), int16(0), int32(0), int64(0)
        u8, u16, u32, u64 = uint8(0), uint16(0), uint32(0), uint64(0)
        f32, f64, fld = float32(0), float64(0), longdouble(0)
        c64, c128, cld = complex64(0), complex128(0), clongdouble(0)

        # coercion within the same kind
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

        # coercion between kinds
        assert_equal(promote_func(b,i32), np.dtype(int32))
        assert_equal(promote_func(b,u8), np.dtype(uint8))
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
        assert_equal(promote_func(array([b]),i8), np.dtype(int8))
        assert_equal(promote_func(array([b]),u8), np.dtype(uint8))
        assert_equal(promote_func(array([b]),i32), np.dtype(int32))
        assert_equal(promote_func(array([b]),u32), np.dtype(uint32))
        assert_equal(promote_func(array([i8]),i64), np.dtype(int8))
        assert_equal(promote_func(u64,array([i32])), np.dtype(int32))
        assert_equal(promote_func(i64,array([u32])), np.dtype(uint32))
        assert_equal(promote_func(int32(-1),array([u64])), np.dtype(float64))
        assert_equal(promote_func(f64,array([f32])), np.dtype(float32))
        assert_equal(promote_func(fld,array([f32])), np.dtype(float32))
        assert_equal(promote_func(array([f64]),fld), np.dtype(float64))
        assert_equal(promote_func(fld,array([c64])), np.dtype(complex64))
        assert_equal(promote_func(c64,array([f64])), np.dtype(complex128))
        assert_equal(promote_func(complex64(3j),array([f64])),
                                                    np.dtype(complex128))

        # coercion between scalars and 1-D arrays, where
        # the scalar has greater kind than the array
        assert_equal(promote_func(array([b]),f64), np.dtype(float64))
        assert_equal(promote_func(array([b]),i64), np.dtype(int64))
        assert_equal(promote_func(array([b]),u64), np.dtype(uint64))
        assert_equal(promote_func(array([i8]),f64), np.dtype(float64))
        assert_equal(promote_func(array([u16]),f64), np.dtype(float64))
        # uint and int are treated as the same "kind" for
        # the purposes of array-scalar promotion.
        assert_equal(promote_func(array([u16]), i32), np.dtype(uint16))
        # float and complex are treated as the same "kind" for
        # the purposes of array-scalar promotion, so that you can do
        # (0j + float32array) to get a complex64 array instead of
        # a complex128 array.
        assert_equal(promote_func(array([f32]),c128), np.dtype(complex64))

    def test_coercion(self):
        def res_type(a, b):
            return np.add(a, b).dtype
        self.check_promotion_cases(res_type)

        # Use-case: float/complex scalar * bool/int8 array
        #           shouldn't narrow the float/complex type
        for a in [np.array([True,False]), np.array([-3,12], dtype=np.int8)]:
            b = 1.234 * a
            assert_equal(b.dtype, np.dtype('f8'), "array type %s" % a.dtype)
            b = np.longdouble(1.234) * a
            assert_equal(b.dtype, np.dtype(np.longdouble),
                                                "array type %s" % a.dtype)
            b = np.float64(1.234) * a
            assert_equal(b.dtype, np.dtype('f8'), "array type %s" % a.dtype)
            b = np.float32(1.234) * a
            assert_equal(b.dtype, np.dtype('f4'), "array type %s" % a.dtype)
            b = np.float16(1.234) * a
            assert_equal(b.dtype, np.dtype('f2'), "array type %s" % a.dtype)

            b = 1.234j * a
            assert_equal(b.dtype, np.dtype('c16'), "array type %s" % a.dtype)
            b = np.clongdouble(1.234j) * a
            assert_equal(b.dtype, np.dtype(np.clongdouble),
                                                "array type %s" % a.dtype)
            b = np.complex128(1.234j) * a
            assert_equal(b.dtype, np.dtype('c16'), "array type %s" % a.dtype)
            b = np.complex64(1.234j) * a
            assert_equal(b.dtype, np.dtype('c8'), "array type %s" % a.dtype)

        # The following use-case is problematic, and to resolve its
        # tricky side-effects requires more changes.
        #
        ## Use-case: (1-t)*a, where 't' is a boolean array and 'a' is
        ##            a float32, shouldn't promote to float64
        #a = np.array([1.0, 1.5], dtype=np.float32)
        #t = np.array([True, False])
        #b = t*a
        #assert_equal(b, [1.0, 0.0])
        #assert_equal(b.dtype, np.dtype('f4'))
        #b = (1-t)*a
        #assert_equal(b, [0.0, 1.5])
        #assert_equal(b.dtype, np.dtype('f4'))
        ## Probably ~t (bitwise negation) is more proper to use here,
        ## but this is arguably less intuitive to understand at a glance, and
        ## would fail if 't' is actually an integer array instead of boolean:
        #b = (~t)*a
        #assert_equal(b, [0.0, 1.5])
        #assert_equal(b.dtype, np.dtype('f4'))

    def test_result_type(self):
        self.check_promotion_cases(np.result_type)

    def test_promote_types_endian(self):
        # promote_types should always return native-endian types
        assert_equal(np.promote_types('<i8', '<i8'), np.dtype('i8'))
        assert_equal(np.promote_types('>i8', '>i8'), np.dtype('i8'))

        assert_equal(np.promote_types('>i8', '>U16'), np.dtype('U16'))
        assert_equal(np.promote_types('<i8', '<U16'), np.dtype('U16'))
        assert_equal(np.promote_types('>U16', '>i8'), np.dtype('U16'))
        assert_equal(np.promote_types('<U16', '<i8'), np.dtype('U16'))

        assert_equal(np.promote_types('<S5', '<U8'), np.dtype('U8'))
        assert_equal(np.promote_types('>S5', '>U8'), np.dtype('U8'))
        assert_equal(np.promote_types('<U8', '<S5'), np.dtype('U8'))
        assert_equal(np.promote_types('>U8', '>S5'), np.dtype('U8'))
        assert_equal(np.promote_types('<U5', '<U8'), np.dtype('U8'))
        assert_equal(np.promote_types('>U8', '>U5'), np.dtype('U8'))

        assert_equal(np.promote_types('<M8', '<M8'), np.dtype('M8'))
        assert_equal(np.promote_types('>M8', '>M8'), np.dtype('M8'))
        assert_equal(np.promote_types('<m8', '<m8'), np.dtype('m8'))
        assert_equal(np.promote_types('>m8', '>m8'), np.dtype('m8'))


    def test_can_cast(self):
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
        assert_equal(np.count_nonzero(array([], dtype='?')), 0)
        assert_equal(np.nonzero(array([])), ([],))

        assert_equal(np.count_nonzero(array(0)), 0)
        assert_equal(np.count_nonzero(array(0, dtype='?')), 0)
        assert_equal(np.nonzero(array(0)), ([],))
        assert_equal(np.count_nonzero(array(1)), 1)
        assert_equal(np.count_nonzero(array(1, dtype='?')), 1)
        assert_equal(np.nonzero(array(1)), ([0],))

    def test_nonzero_onedim(self):
        x = array([1,0,2,-1,0,0,8])
        assert_equal(np.count_nonzero(x), 4)
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
        assert_((array([a[0][V>0],a[1][V>0],a[2][V>0]]) == a[:,V>0]).all())

    def test_boolean_edgecase(self):
        a = np.array([], dtype='int32')
        b = np.array([], dtype='bool')
        c = a[b]
        assert_equal(c, [])
        assert_equal(c.dtype, np.dtype('int32'))


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
        assert_(res)
        assert_(type(res) is bool)
        res = array_equal(array([1,2]), array([1,2,3]))
        assert_(not res)
        assert_(type(res) is bool)
        res = array_equal(array([1,2]), array([3,4]))
        assert_(not res)
        assert_(type(res) is bool)
        res = array_equal(array([1,2]), array([1,3]))
        assert_(not res)
        assert_(type(res) is bool)

    def test_array_equiv(self):
        res = array_equiv(array([1,2]), array([1,2]))
        assert_(res)
        assert_(type(res) is bool)
        res = array_equiv(array([1,2]), array([1,2,3]))
        assert_(not res)
        assert_(type(res) is bool)
        res = array_equiv(array([1,2]), array([3,4]))
        assert_(not res)
        assert_(type(res) is bool)
        res = array_equiv(array([1,2]), array([1,3]))
        assert_(not res)
        assert_(type(res) is bool)

        res = array_equiv(array([1,1]), array([1]))
        assert_(res)
        assert_(type(res) is bool)
        res = array_equiv(array([1,1]), array([[1],[1]]))
        assert_(res)
        assert_(type(res) is bool)
        res = array_equiv(array([1,2]), array([2]))
        assert_(not res)
        assert_(type(res) is bool)
        res = array_equiv(array([1,2]), array([[1],[2]]))
        assert_(not res)
        assert_(type(res) is bool)
        res = array_equiv(array([1,2]), array([[1,2,3],[4,5,6],[7,8,9]]))
        assert_(not res)
        assert_(type(res) is bool)


def assert_array_strict_equal(x, y):
    assert_array_equal(x, y)
    # Check flags
    assert_(x.flags == y.flags)
    # check endianness
    assert_(x.dtype.isnative == y.dtype.isnative)


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
        assert_(not data.dtype.isnative)
        return data

    def _generate_int_data(self, n, m):
        return (10 * rand(n, m)).astype(int64)

    def _generate_int32_data(self, n, m):
        return (10 * rand(n, m)).astype(int32)

    # Now the real test cases
    def test_simple_double(self):
        #Test native double input with scalar min/max.
        a   = self._generate_data(self.nr, self.nc)
        m   = 0.1
        M   = 0.6
        ac  = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_simple_int(self):
        #Test native int input with scalar min/max.
        a   = self._generate_int_data(self.nr, self.nc)
        a   = a.astype(int)
        m   = -2
        M   = 4
        ac  = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_array_double(self):
        #Test native double input with array min/max.
        a   = self._generate_data(self.nr, self.nc)
        m   = zeros(a.shape)
        M   = m + 0.5
        ac  = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_simple_nonnative(self):
        #Test non native double input with scalar min/max.
        #Test native double input with non native double scalar min/max.
        a   = self._generate_non_native_data(self.nr, self.nc)
        m   = -0.5
        M   = 0.6
        ac  = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

        #Test native double input with non native double scalar min/max.
        a   = self._generate_data(self.nr, self.nc)
        m   = -0.5
        M   = self._neg_byteorder(0.6)
        assert_(not M.dtype.isnative)
        ac  = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

    def test_simple_complex(self):
        #Test native complex input with native double scalar min/max.
        #Test native input with complex double scalar min/max.
        a   = 3 * self._generate_data_complex(self.nr, self.nc)
        m   = -0.5
        M   = 1.
        ac  = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

        #Test native input with complex double scalar min/max.
        a   = 3 * self._generate_data(self.nr, self.nc)
        m   = -0.5 + 1.j
        M   = 1. + 2.j
        ac  = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_clip_non_contig(self):
        #Test clip for non contiguous native input and native scalar min/max.
        a   = self._generate_data(self.nr * 2, self.nc * 3)
        a   = a[::2, ::3]
        assert_(not a.flags['F_CONTIGUOUS'])
        assert_(not a.flags['C_CONTIGUOUS'])
        ac  = self.fastclip(a, -1.6, 1.7)
        act = self.clip(a, -1.6, 1.7)
        assert_array_strict_equal(ac, act)

    def test_simple_out(self):
        #Test native double input with scalar min/max.
        a   = self._generate_data(self.nr, self.nc)
        m   = -0.5
        M   = 0.6
        ac  = zeros(a.shape)
        act = zeros(a.shape)
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_simple_int32_inout(self):
        #Test native int32 input with double min/max and int32 out.
        a   = self._generate_int32_data(self.nr, self.nc)
        m   = float64(0)
        M   = float64(2)
        ac  = zeros(a.shape, dtype = int32)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_simple_int64_out(self):
        #Test native int32 input with int32 scalar min/max and int64 out.
        a   = self._generate_int32_data(self.nr, self.nc)
        m   = int32(-1)
        M   = int32(1)
        ac  = zeros(a.shape, dtype = int64)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_simple_int64_inout(self):
        #Test native int32 input with double array min/max and int32 out.
        a   = self._generate_int32_data(self.nr, self.nc)
        m   = zeros(a.shape, float64)
        M   = float64(1)
        ac  = zeros(a.shape, dtype = int32)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_simple_int32_out(self):
        #Test native double input with scalar min/max and int out.
        a   = self._generate_data(self.nr, self.nc)
        m   = -1.0
        M   = 2.0
        ac  = zeros(a.shape, dtype = int32)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_simple_inplace_01(self):
        #Test native double input with array min/max in-place.
        a   = self._generate_data(self.nr, self.nc)
        ac  = a.copy()
        m   = zeros(a.shape)
        M   = 1.0
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_strict_equal(a, ac)

    def test_simple_inplace_02(self):
        #Test native double input with scalar min/max in-place.
        a   = self._generate_data(self.nr, self.nc)
        ac  = a.copy()
        m   = -0.5
        M   = 0.6
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_strict_equal(a, ac)

    def test_noncontig_inplace(self):
        #Test non contiguous double input with double scalar min/max in-place.
        a   = self._generate_data(self.nr * 2, self.nc * 3)
        a   = a[::2, ::3]
        assert_(not a.flags['F_CONTIGUOUS'])
        assert_(not a.flags['C_CONTIGUOUS'])
        ac  = a.copy()
        m   = -0.5
        M   = 0.6
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_equal(a, ac)

    def test_type_cast_01(self):
        #Test native double input with scalar min/max.
        a   = self._generate_data(self.nr, self.nc)
        m   = -0.5
        M   = 0.6
        ac  = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_02(self):
        #Test native int32 input with int32 scalar min/max.
        a   = self._generate_int_data(self.nr, self.nc)
        a   = a.astype(int32)
        m   = -2
        M   = 4
        ac  = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_03(self):
        #Test native int32 input with float64 scalar min/max.
        a   = self._generate_int32_data(self.nr, self.nc)
        m   = -2
        M   = 4
        ac  = self.fastclip(a, float64(m), float64(M))
        act = self.clip(a, float64(m), float64(M))
        assert_array_strict_equal(ac, act)

    def test_type_cast_04(self):
        #Test native int32 input with float32 scalar min/max.
        a   = self._generate_int32_data(self.nr, self.nc)
        m   = float32(-2)
        M   = float32(4)
        act = self.fastclip(a,m,M)
        ac  = self.clip(a,m,M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_05(self):
        #Test native int32 with double arrays min/max.
        a   = self._generate_int_data(self.nr, self.nc)
        m   = -0.5
        M   = 1.
        ac  = self.fastclip(a, m * zeros(a.shape), M)
        act = self.clip(a, m * zeros(a.shape), M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_06(self):
        #Test native with NON native scalar min/max.
        a   = self._generate_data(self.nr, self.nc)
        m   = 0.5
        m_s = self._neg_byteorder(m)
        M   = 1.
        act = self.clip(a, m_s, M)
        ac  = self.fastclip(a, m_s, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_07(self):
        #Test NON native with native array min/max.
        a   = self._generate_data(self.nr, self.nc)
        m   = -0.5 * ones(a.shape)
        M   = 1.
        a_s = self._neg_byteorder(a)
        assert_(not a_s.dtype.isnative)
        act = a_s.clip(m, M)
        ac  = self.fastclip(a_s, m, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_08(self):
        #Test NON native with native scalar min/max.
        a   = self._generate_data(self.nr, self.nc)
        m   = -0.5
        M   = 1.
        a_s = self._neg_byteorder(a)
        assert_(not a_s.dtype.isnative)
        ac  = self.fastclip(a_s, m , M)
        act = a_s.clip(m, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_09(self):
        #Test native with NON native array min/max.
        a   = self._generate_data(self.nr, self.nc)
        m   = -0.5 * ones(a.shape)
        M   = 1.
        m_s = self._neg_byteorder(m)
        assert_(not m_s.dtype.isnative)
        ac  = self.fastclip(a, m_s , M)
        act = self.clip(a, m_s, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_10(self):
        #Test native int32 with float min/max and float out for output argument.
        a   = self._generate_int_data(self.nr, self.nc)
        b   = zeros(a.shape, dtype = float32)
        m   = float32(-0.5)
        M   = float32(1)
        act = self.clip(a, m, M, out = b)
        ac  = self.fastclip(a, m , M, out = b)
        assert_array_strict_equal(ac, act)

    def test_type_cast_11(self):
        #Test non native with native scalar, min/max, out non native
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
        #Test native int32 input and min/max and float out
        a   = self._generate_int_data(self.nr, self.nc)
        b   = zeros(a.shape, dtype = float32)
        m   = int32(0)
        M   = int32(1)
        act = self.clip(a, m, M, out = b)
        ac  = self.fastclip(a, m , M, out = b)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_simple(self):
        #Test native double input with scalar min/max
        a   = self._generate_data(self.nr, self.nc)
        m   = -0.5
        M   = 0.6
        ac  = zeros(a.shape)
        act = zeros(a.shape)
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_simple2(self):
        #Test native int32 input with double min/max and int32 out
        a   = self._generate_int32_data(self.nr, self.nc)
        m   = float64(0)
        M   = float64(2)
        ac  = zeros(a.shape, dtype = int32)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_simple_int32(self):
        #Test native int32 input with int32 scalar min/max and int64 out
        a   = self._generate_int32_data(self.nr, self.nc)
        m   = int32(-1)
        M   = int32(1)
        ac  = zeros(a.shape, dtype = int64)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_array_int32(self):
        #Test native int32 input with double array min/max and int32 out
        a   = self._generate_int32_data(self.nr, self.nc)
        m   = zeros(a.shape, float64)
        M   = float64(1)
        ac  = zeros(a.shape, dtype = int32)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_array_outint32(self):
        #Test native double input with scalar min/max and int out
        a   = self._generate_data(self.nr, self.nc)
        m   = -1.0
        M   = 2.0
        ac  = zeros(a.shape, dtype = int32)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_inplace_array(self):
        #Test native double input with array min/max
        a   = self._generate_data(self.nr, self.nc)
        ac  = a.copy()
        m   = zeros(a.shape)
        M   = 1.0
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_strict_equal(a, ac)

    def test_clip_inplace_simple(self):
        #Test native double input with scalar min/max
        a   = self._generate_data(self.nr, self.nc)
        ac  = a.copy()
        m   = -0.5
        M   = 0.6
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_strict_equal(a, ac)

    def test_clip_func_takes_out(self):
        # Ensure that the clip() function takes an out= argument.
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = -0.5
        M = 0.6
        a2 = clip(a, m, M, out=a)
        self.clip(a, m, M, ac)
        assert_array_strict_equal(a2, ac)
        self.assertTrue(a2 is a)


class TestAllclose(object):
    rtol = 1e-5
    atol = 1e-8

    def setUp(self):
        self.olderr = np.seterr(invalid='ignore')

    def tearDown(self):
        np.seterr(**self.olderr)

    def tst_allclose(self,x,y):
        assert_(allclose(x,y), "%s and %s not close" % (x,y))

    def tst_not_allclose(self,x,y):
        assert_(not allclose(x,y), "%s and %s shouldn't be close" % (x,y))

    def test_ip_allclose(self):
        #Parametric test factory.
        arr = array([100,1000])
        aran = arange(125).reshape((5,5,5))

        atol = self.atol
        rtol = self.rtol

        data = [([1,0], [1,0]),
                ([atol], [0]),
                ([1], [1+rtol+atol]),
                (arr, arr + arr*rtol),
                (arr, arr + arr*rtol + atol*2),
                (aran, aran + aran*rtol),
                (inf, inf),
                (inf, [inf])]

        for (x,y) in data:
            yield (self.tst_allclose,x,y)

    def test_ip_not_allclose(self):
        #Parametric test factory.
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


class TestIsclose(object):
    rtol = 1e-5
    atol = 1e-8

    def setup(self):
        atol = self.atol
        rtol = self.rtol
        arr = array([100,1000])
        aran = arange(125).reshape((5,5,5))

        self.all_close_tests = [
                ([1, 0], [1, 0]),
                ([atol], [0]),
                ([1], [1 + rtol + atol]),
                (arr, arr + arr*rtol),
                (arr, arr + arr*rtol + atol),
                (aran, aran + aran*rtol),
                (inf, inf),
                (inf, [inf]),
                ([inf, -inf], [inf, -inf]),
                ]
        self.none_close_tests = [
                ([inf, 0], [1, inf]),
                ([inf, -inf], [1, 0]),
                ([inf, inf], [1, -inf]),
                ([inf, inf], [1, 0]),
                ([nan, 0], [nan, -inf]),
                ([atol*2], [0]),
                ([1], [1 + rtol + atol*2]),
                (aran, aran + rtol*1.1*aran + atol*1.1),
                (array([inf, 1]), array([0, inf])),
                ]
        self.some_close_tests = [
                ([inf, 0], [inf, atol*2]),
                ([atol, 1, 1e6*(1 + 2*rtol) + atol], [0, nan, 1e6]),
                (arange(3), [0, 1, 2.1]),
                (nan, [nan, nan, nan]),
                ([0], [atol, inf, -inf, nan]),
                (0, [atol, inf, -inf, nan]),
                ]
        self.some_close_results = [
                [True, False],
                [True, False, False],
                [True, True, False],
                [False, False, False],
                [True, False, False, False],
                [True, False, False, False],
                ]

    def test_ip_isclose(self):
        self.setup()
        tests = self.some_close_tests
        results = self.some_close_results
        for (x, y), result in zip(tests, results):
            yield (assert_array_equal, isclose(x, y), result)

    def tst_all_isclose(self, x, y):
        assert_(all(isclose(x, y)), "%s and %s not close" % (x, y))

    def tst_none_isclose(self, x, y):
        msg = "%s and %s shouldn't be close"
        assert_(not any(isclose(x, y)), msg % (x, y))

    def tst_isclose_allclose(self, x, y):
        msg = "isclose.all() and allclose aren't same for %s and %s"
        assert_array_equal(isclose(x, y).all(), allclose(x, y), msg % (x, y))

    def test_ip_all_isclose(self):
        self.setup()
        for (x,y) in self.all_close_tests:
            yield (self.tst_all_isclose, x, y)

    def test_ip_none_isclose(self):
        self.setup()
        for (x,y) in self.none_close_tests:
            yield (self.tst_none_isclose, x, y)

    def test_ip_isclose_allclose(self):
        self.setup()
        tests = (self.all_close_tests + self.none_close_tests +
                 self.some_close_tests)
        for (x, y) in tests:
            yield (self.tst_isclose_allclose, x, y)

    def test_equal_nan(self):
        assert_array_equal(isclose(nan, nan, equal_nan=True), [True])
        arr = array([1.0, nan])
        assert_array_equal(isclose(arr, arr, equal_nan=True), [True, True])

    def test_masked_arrays(self):
        x = np.ma.masked_where([True, True, False], np.arange(3))
        assert_(type(x) == type(isclose(2, x)))

        x = np.ma.masked_where([True, True, False], [nan, inf, nan])
        assert_(type(x) == type(isclose(inf, x)))

        x = np.ma.masked_where([True, True, False], [nan, nan, nan])
        y = isclose(nan, x, equal_nan=True)
        assert_(type(x) == type(y))
        # Ensure that the mask isn't modified...
        assert_array_equal([True, True, False], y.mask)

    def test_scalar_return(self):
        assert_(isscalar(isclose(1, 1)))

    def test_no_parameter_modification(self):
        x = array([inf, 1])
        y = array([0, inf])
        isclose(x, y)
        assert_array_equal(x, array([inf, 1]))
        assert_array_equal(y, array([0, inf]))


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
    '''Test ones_like, zeros_like, and empty_like'''

    def setUp(self):
        self.data = [
                # Array scalars
                (array(3.), None),
                (array(3), 'f8'),
                # 1D arrays
                (arange(6, dtype='f4'), None),
                (arange(6), 'c16'),
                # 2D C-layout arrays
                (arange(6).reshape(2,3), None),
                (arange(6).reshape(3,2), 'i1'),
                # 2D F-layout arrays
                (arange(6).reshape((2,3), order='F'), None),
                (arange(6).reshape((3,2), order='F'), 'i1'),
                # 3D C-layout arrays
                (arange(24).reshape(2,3,4), None),
                (arange(24).reshape(4,3,2), 'f4'),
                # 3D F-layout arrays
                (arange(24).reshape((2,3,4), order='F'), None),
                (arange(24).reshape((4,3,2), order='F'), 'f4'),
                # 3D non-C/F-layout arrays
                (arange(24).reshape(2,3,4).swapaxes(0,1), None),
                (arange(24).reshape(4,3,2).swapaxes(0,1), '?'),
                     ]

    def check_like_function(self, like_function, value):
        for d, dtype in self.data:
            # default (K) order, dtype
            dz = like_function(d, dtype=dtype)
            assert_equal(dz.shape, d.shape)
            assert_equal(array(dz.strides)*d.dtype.itemsize,
                         array(d.strides)*dz.dtype.itemsize)
            assert_equal(d.flags.c_contiguous, dz.flags.c_contiguous)
            assert_equal(d.flags.f_contiguous, dz.flags.f_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            if not value is None:
                assert_(all(dz == value))

            # C order, default dtype
            dz = like_function(d, order='C', dtype=dtype)
            assert_equal(dz.shape, d.shape)
            assert_(dz.flags.c_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            if not value is None:
                assert_(all(dz == value))

            # F order, default dtype
            dz = like_function(d, order='F', dtype=dtype)
            assert_equal(dz.shape, d.shape)
            assert_(dz.flags.f_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            if not value is None:
                assert_(all(dz == value))

            # A order
            dz = like_function(d, order='A', dtype=dtype)
            assert_equal(dz.shape, d.shape)
            if d.flags.f_contiguous:
                assert_(dz.flags.f_contiguous)
            else:
                assert_(dz.flags.c_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            if not value is None:
                assert_(all(dz == value))

        # Test the 'subok' parameter
        a = np.matrix([[1,2],[3,4]])

        b = like_function(a)
        assert_(type(b) is np.matrix)

        b = like_function(a, subok=False)
        assert_(not (type(b) is np.matrix))

    def test_ones_like(self):
        self.check_like_function(np.ones_like, 1)

    def test_zeros_like(self):
        self.check_like_function(np.zeros_like, 0)

    def test_empty_like(self):
        self.check_like_function(np.empty_like, None)

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

class TestArgwhere(object):
    def test_2D(self):
        x = np.arange(6).reshape((2, 3))
        assert_array_equal(np.argwhere(x > 1),
                           [[0, 2],
                            [1, 0],
                            [1, 1],
                            [1, 2]])

    def test_list(self):
        assert_equal(np.argwhere([4, 0, 2, 1, 3]), [[0], [2], [3], [4]])

class TestStringFunction(object):
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
