import sys

import numpy as np
from numpy.testing import *
import numpy.core.umath_tests as umt

class TestUfunc(TestCase):
    def test_reduceat_shifting_sum(self) :
        L = 6
        x = np.arange(L)
        idx = np.array(zip(np.arange(L-2), np.arange(L-2)+2)).ravel()
        assert_array_equal(np.add.reduceat(x,idx)[::2], [1,3,5,7])

    def test_generic_loops(self) :
        """Test generic loops.

        The loops to be tested are:

            PyUFunc_ff_f_As_dd_d
            PyUFunc_ff_f
            PyUFunc_dd_d
            PyUFunc_gg_g
            PyUFunc_FF_F_As_DD_D
            PyUFunc_DD_D
            PyUFunc_FF_F
            PyUFunc_GG_G
            PyUFunc_OO_O
            PyUFunc_OO_O_method
            PyUFunc_f_f_As_d_d
            PyUFunc_d_d
            PyUFunc_f_f
            PyUFunc_g_g
            PyUFunc_F_F_As_D_D
            PyUFunc_F_F
            PyUFunc_D_D
            PyUFunc_G_G
            PyUFunc_O_O
            PyUFunc_O_O_method
            PyUFunc_On_Om

        Where:

            f -- float
            d -- double
            g -- long double
            F -- complex float
            D -- complex double
            G -- complex long double
            O -- python object

        It is difficult to assure that each of these loops is entered from the
        Python level as the special cased loops are a moving target and the
        corresponding types are architecture dependent. We probably need to
        define C level testing ufuncs to get at them. For the time being, I've
        just looked at the signatures registered in the build directory to find
        relevant functions.

        Fixme, currently untested:

            PyUFunc_ff_f_As_dd_d
            PyUFunc_FF_F_As_DD_D
            PyUFunc_f_f_As_d_d
            PyUFunc_F_F_As_D_D
            PyUFunc_On_Om

        """
        fone = np.exp
        ftwo = lambda x,y : x**y
        fone_val = 1
        ftwo_val = 1
        # check unary PyUFunc_f_f.
        msg = "PyUFunc_f_f"
        x = np.zeros(10, dtype=np.single)[0::2]
        assert_almost_equal(fone(x), fone_val, err_msg=msg)
        # check unary PyUFunc_d_d.
        msg = "PyUFunc_d_d"
        x = np.zeros(10, dtype=np.double)[0::2]
        assert_almost_equal(fone(x), fone_val, err_msg=msg)
        # check unary PyUFunc_g_g.
        msg = "PyUFunc_g_g"
        x = np.zeros(10, dtype=np.longdouble)[0::2]
        assert_almost_equal(fone(x), fone_val, err_msg=msg)
        # check unary PyUFunc_F_F.
        msg = "PyUFunc_F_F"
        x = np.zeros(10, dtype=np.csingle)[0::2]
        assert_almost_equal(fone(x), fone_val, err_msg=msg)
        # check unary PyUFunc_D_D.
        msg = "PyUFunc_D_D"
        x = np.zeros(10, dtype=np.cdouble)[0::2]
        assert_almost_equal(fone(x), fone_val, err_msg=msg)
        # check unary PyUFunc_G_G.
        msg = "PyUFunc_G_G"
        x = np.zeros(10, dtype=np.clongdouble)[0::2]
        assert_almost_equal(fone(x), fone_val, err_msg=msg)

        # check binary PyUFunc_ff_f.
        msg = "PyUFunc_ff_f"
        x = np.ones(10, dtype=np.single)[0::2]
        assert_almost_equal(ftwo(x,x), ftwo_val, err_msg=msg)
        # check binary PyUFunc_dd_d.
        msg = "PyUFunc_dd_d"
        x = np.ones(10, dtype=np.double)[0::2]
        assert_almost_equal(ftwo(x,x), ftwo_val, err_msg=msg)
        # check binary PyUFunc_gg_g.
        msg = "PyUFunc_gg_g"
        x = np.ones(10, dtype=np.longdouble)[0::2]
        assert_almost_equal(ftwo(x,x), ftwo_val, err_msg=msg)
        # check binary PyUFunc_FF_F.
        msg = "PyUFunc_FF_F"
        x = np.ones(10, dtype=np.csingle)[0::2]
        assert_almost_equal(ftwo(x,x), ftwo_val, err_msg=msg)
        # check binary PyUFunc_DD_D.
        msg = "PyUFunc_DD_D"
        x = np.ones(10, dtype=np.cdouble)[0::2]
        assert_almost_equal(ftwo(x,x), ftwo_val, err_msg=msg)
        # check binary PyUFunc_GG_G.
        msg = "PyUFunc_GG_G"
        x = np.ones(10, dtype=np.clongdouble)[0::2]
        assert_almost_equal(ftwo(x,x), ftwo_val, err_msg=msg)

        # class to use in testing object method loops
        class foo :
            def logical_not(self) :
                return np.bool_(1)
            def logical_and(self, obj) :
                return np.bool_(1)

        # check unary PyUFunc_O_O
        msg = "PyUFunc_O_O"
        x = np.ones(10, dtype=np.object)[0::2]
        assert np.all(np.abs(x) == 1), msg
        # check unary PyUFunc_O_O_method
        msg = "PyUFunc_O_O_method"
        x = np.zeros(10, dtype=np.object)[0::2]
        for i in range(len(x)) :
            x[i] = foo()
        assert np.all(np.logical_not(x) == True), msg

        # check binary PyUFunc_OO_O
        msg = "PyUFunc_OO_O"
        x = np.ones(10, dtype=np.object)[0::2]
        assert np.all(np.add(x,x) == 2), msg
        # check binary PyUFunc_OO_O_method
        msg = "PyUFunc_OO_O_method"
        x = np.zeros(10, dtype=np.object)[0::2]
        for i in range(len(x)) :
            x[i] = foo()
        assert np.all(np.logical_and(x,x) == 1), msg

        # check PyUFunc_On_Om
        # fixme -- I don't know how to do this yet

    def test_all_ufunc(self) :
        """Try to check presence and results of all ufuncs.

        The list of ufuncs comes from generate_umath.py and is as follows:

        =====  ====  =============  ===============  ========================
        done   args   function        types                notes
        =====  ====  =============  ===============  ========================
        n      1     conjugate      nums + O
        n      1     absolute       nums + O         complex -> real
        n      1     negative       nums + O
        n      1     sign           nums + O         -> int
        n      1     invert         bool + ints + O  flts raise an error
        n      1     degrees        real + M         cmplx raise an error
        n      1     radians        real + M         cmplx raise an error
        n      1     arccos         flts + M
        n      1     arccosh        flts + M
        n      1     arcsin         flts + M
        n      1     arcsinh        flts + M
        n      1     arctan         flts + M
        n      1     arctanh        flts + M
        n      1     cos            flts + M
        n      1     sin            flts + M
        n      1     tan            flts + M
        n      1     cosh           flts + M
        n      1     sinh           flts + M
        n      1     tanh           flts + M
        n      1     exp            flts + M
        n      1     expm1          flts + M
        n      1     log            flts + M
        n      1     log10          flts + M
        n      1     log1p          flts + M
        n      1     sqrt           flts + M         real x < 0 raises error
        n      1     ceil           real + M
        n      1     trunc          real + M
        n      1     floor          real + M
        n      1     fabs           real + M
        n      1     rint           flts + M
        n      1     isnan          flts             -> bool
        n      1     isinf          flts             -> bool
        n      1     isfinite       flts             -> bool
        n      1     signbit        real             -> bool
        n      1     modf           real             -> (frac, int)
        n      1     logical_not    bool + nums + M  -> bool
        n      2     left_shift     ints + O         flts raise an error
        n      2     right_shift    ints + O         flts raise an error
        n      2     add            bool + nums + O  boolean + is ||
        n      2     subtract       bool + nums + O  boolean - is ^
        n      2     multiply       bool + nums + O  boolean * is &
        n      2     divide         nums + O
        n      2     floor_divide   nums + O
        n      2     true_divide    nums + O         bBhH -> f, iIlLqQ -> d
        n      2     fmod           nums + M
        n      2     power          nums + O
        n      2     greater        bool + nums + O  -> bool
        n      2     greater_equal  bool + nums + O  -> bool
        n      2     less           bool + nums + O  -> bool
        n      2     less_equal     bool + nums + O  -> bool
        n      2     equal          bool + nums + O  -> bool
        n      2     not_equal      bool + nums + O  -> bool
        n      2     logical_and    bool + nums + M  -> bool
        n      2     logical_or     bool + nums + M  -> bool
        n      2     logical_xor    bool + nums + M  -> bool
        n      2     maximum        bool + nums + O
        n      2     minimum        bool + nums + O
        n      2     bitwise_and    bool + ints + O  flts raise an error
        n      2     bitwise_or     bool + ints + O  flts raise an error
        n      2     bitwise_xor    bool + ints + O  flts raise an error
        n      2     arctan2        real + M
        n      2     remainder      ints + real + O
        n      2     hypot          real + M
        =====  ====  =============  ===============  ========================

        Types other than those listed will be accepted, but they are cast to
        the smallest compatible type for which the function is defined. The
        casting rules are:

        bool -> int8 -> float32
        ints -> double

        """
        pass


    def test_signature(self):
        # the arguments to test_signature are: nin, nout, core_signature
        # pass
        assert_equal(umt.test_signature(2,1,"(i),(i)->()"), 1)

        # pass. empty core signature; treat as plain ufunc (with trivial core)
        assert_equal(umt.test_signature(2,1,"(),()->()"), 0)

        # in the following calls, a ValueError should be raised because
        # of error in core signature
        # error: extra parenthesis
        msg = "core_sig: extra parenthesis"
        try:
            ret = umt.test_signature(2,1,"((i)),(i)->()")
            assert_equal(ret, None, err_msg=msg)
        except ValueError: None
        # error: parenthesis matching
        msg = "core_sig: parenthesis matching"
        try:
            ret = umt.test_signature(2,1,"(i),)i(->()")
            assert_equal(ret, None, err_msg=msg)
        except ValueError: None
        # error: incomplete signature. letters outside of parenthesis are ignored
        msg = "core_sig: incomplete signature"
        try:
            ret = umt.test_signature(2,1,"(i),->()")
            assert_equal(ret, None, err_msg=msg)
        except ValueError: None
        # error: incomplete signature. 2 output arguments are specified
        msg = "core_sig: incomplete signature"
        try:
            ret = umt.test_signature(2,2,"(i),(i)->()")
            assert_equal(ret, None, err_msg=msg)
        except ValueError: None

        # more complicated names for variables
        assert_equal(umt.test_signature(2,1,"(i1,i2),(J_1)->(_kAB)"),1)

    def test_get_signature(self):
        assert_equal(umt.inner1d.signature, "(i),(i)->()")

    def test_forced_sig(self):
        a = 0.5*np.arange(3,dtype='f8')
        assert_equal(np.add(a,0.5), [0.5, 1, 1.5])
        assert_equal(np.add(a,0.5,sig='i',casting='unsafe'), [0, 0, 1])
        assert_equal(np.add(a,0.5,sig='ii->i',casting='unsafe'), [0, 0, 1])
        assert_equal(np.add(a,0.5,sig=('i4',),casting='unsafe'), [0, 0, 1])
        assert_equal(np.add(a,0.5,sig=('i4','i4','i4'),
                                            casting='unsafe'), [0, 0, 1])

        b = np.zeros((3,),dtype='f8')
        np.add(a,0.5,out=b)
        assert_equal(b, [0.5, 1, 1.5])
        b[:] = 0
        np.add(a,0.5,sig='i',out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])
        b[:] = 0
        np.add(a,0.5,sig='ii->i',out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])
        b[:] = 0
        np.add(a,0.5,sig=('i4',),out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])
        b[:] = 0
        np.add(a,0.5,sig=('i4','i4','i4'),out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])
        

    def test_inner1d(self):
        a = np.arange(6).reshape((2,3))
        assert_array_equal(umt.inner1d(a,a), np.sum(a*a,axis=-1))

    def test_broadcast(self):
        msg = "broadcast"
        a = np.arange(4).reshape((2,1,2))
        b = np.arange(4).reshape((1,2,2))
        assert_array_equal(umt.inner1d(a,b), np.sum(a*b,axis=-1), err_msg=msg)
        msg = "extend & broadcast loop dimensions"
        b = np.arange(4).reshape((2,2))
        assert_array_equal(umt.inner1d(a,b), np.sum(a*b,axis=-1), err_msg=msg)
        msg = "broadcast in core dimensions"
        a = np.arange(8).reshape((4,2))
        b = np.arange(4).reshape((4,1))
        assert_array_equal(umt.inner1d(a,b), np.sum(a*b,axis=-1), err_msg=msg)
        msg = "extend & broadcast core and loop dimensions"
        a = np.arange(8).reshape((4,2))
        b = np.array(7)
        assert_array_equal(umt.inner1d(a,b), np.sum(a*b,axis=-1), err_msg=msg)
        msg = "broadcast should fail"
        a = np.arange(2).reshape((2,1,1))
        b = np.arange(3).reshape((3,1,1))
        try:
            ret = umt.inner1d(a,b)
            assert_equal(ret, None, err_msg=msg)
        except ValueError: None

    def test_type_cast(self):
        msg = "type cast"
        a = np.arange(6, dtype='short').reshape((2,3))
        assert_array_equal(umt.inner1d(a,a), np.sum(a*a,axis=-1), err_msg=msg)
        msg = "type cast on one argument"
        a = np.arange(6).reshape((2,3))
        b = a+0.1
        assert_array_almost_equal(umt.inner1d(a,a), np.sum(a*a,axis=-1),
            err_msg=msg)

    def test_endian(self):
        msg = "big endian"
        a = np.arange(6, dtype='>i4').reshape((2,3))
        assert_array_equal(umt.inner1d(a,a), np.sum(a*a,axis=-1), err_msg=msg)
        msg = "little endian"
        a = np.arange(6, dtype='<i4').reshape((2,3))
        assert_array_equal(umt.inner1d(a,a), np.sum(a*a,axis=-1), err_msg=msg)

    def test_incontiguous_array(self):
        msg = "incontiguous memory layout of array"
        x = np.arange(64).reshape((2,2,2,2,2,2))
        a = x[:,0,:,0,:,0]
        b = x[:,1,:,1,:,1]
        a[0,0,0] = -1
        msg2 = "make sure it references to the original array"
        assert_equal(x[0,0,0,0,0,0], -1, err_msg=msg2)
        assert_array_equal(umt.inner1d(a,b), np.sum(a*b,axis=-1), err_msg=msg)
        x = np.arange(24).reshape(2,3,4)
        a = x.T
        b = x.T
        a[0,0,0] = -1
        assert_equal(x[0,0,0], -1, err_msg=msg2)
        assert_array_equal(umt.inner1d(a,b), np.sum(a*b,axis=-1), err_msg=msg)

    def test_output_argument(self):
        msg = "output argument"
        a = np.arange(12).reshape((2,3,2))
        b = np.arange(4).reshape((2,1,2)) + 1
        c = np.zeros((2,3),dtype='int')
        umt.inner1d(a,b,c)
        assert_array_equal(c, np.sum(a*b,axis=-1), err_msg=msg)
        c[:] = -1
        umt.inner1d(a,b,out=c)
        assert_array_equal(c, np.sum(a*b,axis=-1), err_msg=msg)

        msg = "output argument with type cast"
        c = np.zeros((2,3),dtype='int16')
        umt.inner1d(a,b,c)
        assert_array_equal(c, np.sum(a*b,axis=-1), err_msg=msg)
        c[:] = -1
        umt.inner1d(a,b,out=c)
        assert_array_equal(c, np.sum(a*b,axis=-1), err_msg=msg)

        msg = "output argument with incontiguous layout"
        c = np.zeros((2,3,4),dtype='int16')
        umt.inner1d(a,b,c[...,0])
        assert_array_equal(c[...,0], np.sum(a*b,axis=-1), err_msg=msg)
        c[:] = -1
        umt.inner1d(a,b,out=c[...,0])
        assert_array_equal(c[...,0], np.sum(a*b,axis=-1), err_msg=msg)

    def test_innerwt(self):
        a = np.arange(6).reshape((2,3))
        b = np.arange(10,16).reshape((2,3))
        w = np.arange(20,26).reshape((2,3))
        assert_array_equal(umt.innerwt(a,b,w), np.sum(a*b*w,axis=-1))
        a = np.arange(100,124).reshape((2,3,4))
        b = np.arange(200,224).reshape((2,3,4))
        w = np.arange(300,324).reshape((2,3,4))
        assert_array_equal(umt.innerwt(a,b,w), np.sum(a*b*w,axis=-1))

    def test_matrix_multiply(self):
        self.compare_matrix_multiply_results(np.long)
        self.compare_matrix_multiply_results(np.double)

    def compare_matrix_multiply_results(self, tp):
        d1 = np.array(rand(2,3,4), dtype=tp)
        d2 = np.array(rand(2,3,4), dtype=tp)
        msg = "matrix multiply on type %s" % d1.dtype.name

        def permute_n(n):
            if n == 1:
                return ([0],)
            ret = ()
            base = permute_n(n-1)
            for perm in base:
                for i in xrange(n):
                    new = perm + [n-1]
                    new[n-1] = new[i]
                    new[i] = n-1
                    ret += (new,)
            return ret

        def slice_n(n):
            if n == 0:
                return ((),)
            ret = ()
            base = slice_n(n-1)
            for sl in base:
                ret += (sl+(slice(None),),)
                ret += (sl+(slice(0,1),),)
            return ret

        def broadcastable(s1,s2):
            return s1 == s2 or s1 == 1 or s2 == 1

        permute_3 = permute_n(3)
        slice_3 = slice_n(3) + ((slice(None,None,-1),)*3,)

        ref = True
        for p1 in permute_3:
            for p2 in permute_3:
                for s1 in slice_3:
                    for s2 in slice_3:
                        a1 = d1.transpose(p1)[s1]
                        a2 = d2.transpose(p2)[s2]
                        ref = ref and a1.base != None and a1.base.base != None
                        ref = ref and a2.base != None and a2.base.base != None
                        if broadcastable(a1.shape[-1], a2.shape[-2]) and \
                           broadcastable(a1.shape[0], a2.shape[0]):
                            assert_array_almost_equal(
                                umt.matrix_multiply(a1,a2),
                                np.sum(a2[...,np.newaxis].swapaxes(-3,-1) *
                                       a1[...,np.newaxis,:], axis=-1),
                                err_msg = msg+' %s %s' % (str(a1.shape),
                                                          str(a2.shape)))

        assert_equal(ref, True, err_msg="reference check")

if __name__ == "__main__":
    run_module_suite()
