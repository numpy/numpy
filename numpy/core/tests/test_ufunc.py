import numpy as np
from numpy.testing import *

class TestUfunc(NumpyTestCase):
    def test_reduceat_shifting_sum(self) :
        L = 6
        x = np.arange(L)
        idx = np.array(zip(np.arange(L-2), np.arange(L-2)+2)).ravel()
        assert_array_equal(np.add.reduceat(x,idx)[::2],
                           [1,3,5,7])
    def check_generic_loops(self) :
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

        # check unary PyUFunc_O_0
        msg = "PyUFunc_O_O"
        x = np.ones(10, dtype=np.object)[0::2]
        assert np.all(np.abs(x) == 1), msg
        # check unary PyUFunc_O_0_method
        msg = "PyUFunc_O_O_method"
        x = np.zeros(10, dtype=np.object)[0::2]
        for i in range(len(x)) :
            x[i] = foo()
        assert np.all(np.logical_not(x) == True), msg

        # check binary PyUFunc_OO_0
        msg = "PyUFunc_OO_O"
        x = np.ones(10, dtype=np.object)[0::2]
        assert np.all(np.add(x,x) == 2), msg
        # check binary PyUFunc_OO_0_method
        msg = "PyUFunc_OO_O_method"
        x = np.zeros(10, dtype=np.object)[0::2]
        for i in range(len(x)) :
            x[i] = foo()
        assert np.all(np.logical_and(x,x) == 1), msg

        # check PyUFunc_On_Om
        # fixme -- I don't know how to do this yet

    def check_all_ufunc(self) :
        """Try to check presence and results of all ufuncs.

        The list of ufuncs comes from generate_umath.py and is as follows:

        =====  =============  ===============  ========================
        done     function        types                notes
        =====  =============  ===============  ========================
        n      add            bool + nums + O  boolean + is ||
        n      subtract       bool + nums + O  boolean - is ^
        n      multiply       bool + nums + O  boolean * is &
        n      divide         nums + O
        n      floor_divide   nums + O
        n      true_divide    nums + O         bBhH -> f, iIlLqQ -> d
        n      conjugate      nums + O
        n      fmod           nums + M
        n      square         nums + O
        n      reciprocal     nums + O
        n      ones_like      nums + O
        n      power          nums + O
        n      absolute       nums + O         complex -> real
        n      negative       nums + O
        n      sign           nums + O         -> int
        n      greater        bool + nums + O  -> bool
        n      greater_equal  bool + nums + O  -> bool
        n      less           bool + nums + O  -> bool
        n      less_equal     bool + nums + O  -> bool
        n      equal          bool + nums + O  -> bool
        n      not_equal      bool + nums + O  -> bool
        n      logical_and    bool + nums + M  -> bool
        n      logical_not    bool + nums + M  -> bool
        n      logical_or     bool + nums + M  -> bool
        n      logical_xor    bool + nums + M  -> bool
        n      maximum        bool + nums + O
        n      minimum        bool + nums + O
        n      bitwise_and    bool + ints + O  flts raise an error
        n      bitwise_or     bool + ints + O  flts raise an error
        n      bitwise_xor    bool + ints + O  flts raise an error
        n      invert         bool + ints + O  flts raise an error
        n      left_shift     ints + O         flts raise an error
        n      right_shift    ints + O         flts raise an error
        n      degrees        real + M         cmplx raise an error
        n      radians        real + M         cmplx raise an error
        n      arccos         flts + M
        n      arccosh        flts + M
        n      arcsin         flts + M
        n      arcsinh        flts + M
        n      arctan         flts + M
        n      arctanh        flts + M
        n      cos            flts + M
        n      sin            flts + M
        n      tan            flts + M
        n      cosh           flts + M
        n      sinh           flts + M
        n      tanh           flts + M
        n      exp            flts + M
        n      expm1          flts + M
        n      log            flts + M
        n      log10          flts + M
        n      log1p          flts + M
        n      sqrt           flts + M         real x < 0 raises error
        n      ceil           real + M
        n      floor          real + M
        n      fabs           real + M
        n      rint           flts + M
        n      arctan2        real + M
        n      remainder      ints + real + O
        n      hypot          real + M
        n      isnan          flts             -> bool
        n      isinf          flts             -> bool
        n      isfinite       flts             -> bool
        n      signbit        real             -> bool
        n      modf           real             -> (frac, int)
        =====  =============  ===============  ========================

        Types other than those listed will be accepted, but they are cast to
        the smallest compatible type for which the function is defined. The
        casting rules are:

        bool -> int8 -> float32
        ints -> double

        """
        pass

if __name__ == "__main__":
    NumpyTest().run()
