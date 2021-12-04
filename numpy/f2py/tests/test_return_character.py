import pytest

from numpy import array
from numpy.testing import assert_
from . import util
import platform

IS_S390X = platform.machine() == "s390x"


class TestReturnCharacter(util.F2PyTest):
    def check_function(self, t, tname):
        if tname in ["t0", "t1", "s0", "s1"]:
            assert_(t(23) == b"2")
            r = t("ab")
            assert_(r == b"a", repr(r))
            r = t(array("ab"))
            assert_(r == b"a", repr(r))
            r = t(array(77, "u1"))
            assert_(r == b"M", repr(r))
            # assert_(_raises(ValueError, t, array([77,87])))
            # assert_(_raises(ValueError, t, array(77)))
        elif tname in ["ts", "ss"]:
            assert_(t(23) == b"23", repr(t(23)))
            assert_(t("123456789abcdef") == b"123456789a")
        elif tname in ["t5", "s5"]:
            assert_(t(23) == b"23", repr(t(23)))
            assert_(t("ab") == b"ab", repr(t("ab")))
            assert_(t("123456789abcdef") == b"12345")
        else:
            raise NotImplementedError


class TestFReturnCharacter(TestReturnCharacter):
    sources = [
        util.getpath("tests", "src", "return_character", "foo77.f"),
        util.getpath("tests", "src", "return_character", "foo90.f90"),
    ]

    @pytest.mark.xfail(IS_S390X, reason="callback returns ' '")
    @pytest.mark.parametrize("name", "t0,t1,t5,s0,s1,s5,ss".split(","))
    def test_all_f77(self, name):
        self.check_function(getattr(self.module, name), name)

    @pytest.mark.xfail(IS_S390X, reason="callback returns ' '")
    @pytest.mark.parametrize("name", "t0,t1,t5,ts,s0,s1,s5,ss".split(","))
    def test_all_f90(self, name):
        self.check_function(getattr(self.module.f90_return_char, name), name)
