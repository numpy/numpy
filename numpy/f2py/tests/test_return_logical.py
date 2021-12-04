import pytest

from numpy import array
from numpy.testing import assert_, assert_raises
from . import util


class TestReturnLogical(util.F2PyTest):
    def check_function(self, t):
        assert_(t(True) == 1, repr(t(True)))
        assert_(t(False) == 0, repr(t(False)))
        assert_(t(0) == 0)
        assert_(t(None) == 0)
        assert_(t(0.0) == 0)
        assert_(t(0j) == 0)
        assert_(t(1j) == 1)
        assert_(t(234) == 1)
        assert_(t(234.6) == 1)
        assert_(t(234.6 + 3j) == 1)
        assert_(t("234") == 1)
        assert_(t("aaa") == 1)
        assert_(t("") == 0)
        assert_(t([]) == 0)
        assert_(t(()) == 0)
        assert_(t({}) == 0)
        assert_(t(t) == 1)
        assert_(t(-234) == 1)
        assert_(t(10**100) == 1)
        assert_(t([234]) == 1)
        assert_(t((234, )) == 1)
        assert_(t(array(234)) == 1)
        assert_(t(array([234])) == 1)
        assert_(t(array([[234]])) == 1)
        assert_(t(array([234], "b")) == 1)
        assert_(t(array([234], "h")) == 1)
        assert_(t(array([234], "i")) == 1)
        assert_(t(array([234], "l")) == 1)
        assert_(t(array([234], "f")) == 1)
        assert_(t(array([234], "d")) == 1)
        assert_(t(array([234 + 3j], "F")) == 1)
        assert_(t(array([234], "D")) == 1)
        assert_(t(array(0)) == 0)
        assert_(t(array([0])) == 0)
        assert_(t(array([[0]])) == 0)
        assert_(t(array([0j])) == 0)
        assert_(t(array([1])) == 1)
        assert_raises(ValueError, t, array([0, 0]))


class TestFReturnLogical(TestReturnLogical):
    sources = [
        util.getpath("tests", "src", "return_logical", "foo77.f"),
        util.getpath("tests", "src", "return_logical", "foo90.f90"),
    ]

    @pytest.mark.slow
    @pytest.mark.parametrize("name", "t0,t1,t2,t4,s0,s1,s2,s4".split(","))
    def test_all_f77(self, name):
        self.check_function(getattr(self.module, name))

    @pytest.mark.slow
    @pytest.mark.parametrize("name",
                             "t0,t1,t2,t4,t8,s0,s1,s2,s4,s8".split(","))
    def test_all_f90(self, name):
        self.check_function(getattr(self.module.f90_return_logical, name))
