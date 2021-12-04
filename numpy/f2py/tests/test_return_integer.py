import pytest

from numpy import array
from numpy.testing import assert_, assert_raises
from . import util


class TestReturnInteger(util.F2PyTest):
    def check_function(self, t, tname):
        assert_(t(123) == 123, repr(t(123)))
        assert_(t(123.6) == 123)
        assert_(t("123") == 123)
        assert_(t(-123) == -123)
        assert_(t([123]) == 123)
        assert_(t((123, )) == 123)
        assert_(t(array(123)) == 123)
        assert_(t(array([123])) == 123)
        assert_(t(array([[123]])) == 123)
        assert_(t(array([123], "b")) == 123)
        assert_(t(array([123], "h")) == 123)
        assert_(t(array([123], "i")) == 123)
        assert_(t(array([123], "l")) == 123)
        assert_(t(array([123], "B")) == 123)
        assert_(t(array([123], "f")) == 123)
        assert_(t(array([123], "d")) == 123)

        # assert_raises(ValueError, t, array([123],'S3'))
        assert_raises(ValueError, t, "abc")

        assert_raises(IndexError, t, [])
        assert_raises(IndexError, t, ())

        assert_raises(Exception, t, t)
        assert_raises(Exception, t, {})

        if tname in ["t8", "s8"]:
            assert_raises(OverflowError, t, 100000000000000000000000)
            assert_raises(OverflowError, t, 10000000011111111111111.23)


class TestFReturnInteger(TestReturnInteger):
    sources = [
        util.getpath("tests", "src", "return_integer", "foo77.f"),
        util.getpath("tests", "src", "return_integer", "foo90.f90"),
    ]

    @pytest.mark.parametrize("name",
                             "t0,t1,t2,t4,t8,s0,s1,s2,s4,s8".split(","))
    def test_all_f77(self, name):
        self.check_function(getattr(self.module, name), name)

    @pytest.mark.parametrize("name",
                             "t0,t1,t2,t4,t8,s0,s1,s2,s4,s8".split(","))
    def test_all_f90(self, name):
        self.check_function(getattr(self.module.f90_return_integer, name),
                            name)
