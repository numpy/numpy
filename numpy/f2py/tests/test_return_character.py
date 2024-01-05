import pytest

from numpy import array
from . import util
import platform

IS_S390X = platform.machine() == "s390x"


@pytest.fixture(scope="module")
def retchar_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestF77ReturnCharacter",
        sources=[
            util.getpath("tests", "src", "return_character", "foo77.f"),
            util.getpath("tests", "src", "return_character", "foo90.f90"),
        ],
    )
    return spec


def check_function(modcomp, tname):
    t = getattr(modcomp, tname)
    if tname in ["t0", "t1", "s0", "s1"]:
        assert t("23") == b"2"
        assert t("ab") == b"a"
        assert t(array("ab")) == b"a"
        assert t(array(77, "u1")) == b"M"
    elif tname in ["ts", "ss"]:
        assert t(23) == b"23"
        assert t("123456789abcdef") == b"123456789a"
    elif tname in ["t5", "s5"]:
        assert t(23) == b"23"
        assert t("ab") == b"ab"
        assert t("123456789abcdef") == b"12345"
    else:
        raise NotImplementedError


@pytest.mark.xfail(IS_S390X, reason="callback returns ' '")
@pytest.mark.parametrize("name", "t0,t1,t5,s0,s1,s5,ss".split(","))
@pytest.mark.parametrize("_mod", ["retchar_spec"], indirect=True)
def test_all_f77(_mod, name):
    check_function(_mod, name)


@pytest.mark.xfail(IS_S390X, reason="callback returns ' '")
@pytest.mark.parametrize("name", "t0,t1,t5,ts,s0,s1,s5,ss".split(","))
@pytest.mark.parametrize("_mod", ["retchar_spec"], indirect=True)
def test_all_f90(_mod, name):
    check_function(_mod.f90_return_char, name)
