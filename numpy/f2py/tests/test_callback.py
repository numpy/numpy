import math
import textwrap
import sys
import pytest
import threading
import traceback
import time

import numpy as np
from numpy.testing import IS_PYPY
from . import util


@pytest.fixture(scope="module")
def f77_callback_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestF77Callback",
        sources=[util.getpath("tests", "src", "callback", "foo.f")],
    )
    return spec


@pytest.fixture(scope="module")
def f90_callback_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestF90Callback",
        sources=[util.getpath("tests", "src", "callback", "gh17797.f90")],
    )
    return spec


@pytest.fixture(scope="module")
def gh18335_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestGH18335",
        sources=[util.getpath("tests", "src", "callback", "gh18335.f90")],
    )
    return spec


@pytest.fixture(scope="module")
def gh25211_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestGH25211",
        sources=[
            util.getpath("tests", "src", "callback", "gh25211.f"),
            util.getpath("tests", "src", "callback", "gh25211.pyf"),
        ],
        module_name="callback2",
    )
    return spec


# TestF77Callback


def check_function(mod, name):
    t = getattr(mod, name)
    r = t(lambda: 4)
    assert r == 4
    r = t(lambda a: 5, fun_extra_args=(6,))
    assert r == 5
    r = t(lambda a: a, fun_extra_args=(6,))
    assert r == 6
    r = t(lambda a: 5 + a, fun_extra_args=(7,))
    assert r == 12
    r = t(lambda a: math.degrees(a), fun_extra_args=(math.pi,))
    assert r == 180
    r = t(math.degrees, fun_extra_args=(math.pi,))
    assert r == 180

    r = t(mod.func, fun_extra_args=(6,))
    assert r == 17
    r = t(mod.func0)
    assert r == 11
    r = t(mod.func0._cpointer)
    assert r == 11


@pytest.mark.parametrize("_mod", ["f77_callback_spec"], indirect=True)
@pytest.mark.parametrize("name", ["t", "t2"])
def test_all(_mod, name):
    check_function(_mod, name)


@pytest.mark.xfail(IS_PYPY, reason="PyPy cannot modify tp_doc after PyType_Ready")
@pytest.mark.parametrize("_mod", ["f77_callback_spec"], indirect=True)
def test_docstring(_mod):
    expected = textwrap.dedent(
        """\
    a = t(fun,[fun_extra_args])

    Wrapper for ``t``.

    Parameters
    ----------
    fun : call-back function

    Other Parameters
    ----------------
    fun_extra_args : input tuple, optional
        Default: ()

    Returns
    -------
    a : int

    Notes
    -----
    Call-back functions::

        def fun(): return a
        Return objects:
            a : int
    """
    )
    assert _mod.t.__doc__ == expected


@pytest.mark.skipif(
    sys.platform == "win32", reason="Fails with MinGW64 Gfortran (Issue #9673)"
)
@pytest.mark.parametrize("_mod", ["f77_callback_spec"], indirect=True)
def test_string_callback(_mod):
    def callback(code):
        if code == "r":
            return 0
        else:
            return 1

    f = getattr(_mod, "string_callback")
    r = f(callback)
    assert r == 0


@pytest.mark.skipif(
    sys.platform == "win32", reason="Fails with MinGW64 Gfortran (Issue #9673)"
)
@pytest.mark.parametrize("_mod", ["f77_callback_spec"], indirect=True)
def test_string_callback_array(_mod):
    # See gh-10027
    cu1 = np.zeros((1,), "S8")
    cu2 = np.zeros((1, 8), "c")
    cu3 = np.array([""], "S8")

    def callback(cu, lencu):
        if cu.shape != (lencu,):
            return 1
        if cu.dtype != "S8":
            return 2
        if not np.all(cu == b""):
            return 3
        return 0

    f = getattr(_mod, "string_callback_array")
    for cu in [cu1, cu2, cu3]:
        res = f(callback, cu, cu.size)
        assert res == 0


@pytest.mark.parametrize("_mod", ["f77_callback_spec"], indirect=True)
def test_threadsafety(_mod):
    # Segfaults if the callback handling is not threadsafe

    errors = []

    def cb():
        # Sleep here to make it more likely for another thread
        # to call their callback at the same time.
        time.sleep(1e-3)

        # Check reentrancy
        r = _mod.t(lambda: 123)
        assert r == 123

        return 42

    def runner(name):
        try:
            for j in range(50):
                r = _mod.t(cb)
                assert r == 42
                check_function(_mod, name)
        except Exception:
            errors.append(traceback.format_exc())

    threads = [
        threading.Thread(target=runner, args=(arg,))
        for arg in ("t", "t2")
        for n in range(20)
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    errors = "\n\n".join(errors)
    if errors:
        raise AssertionError(errors)


@pytest.mark.parametrize("_mod", ["f77_callback_spec"], indirect=True)
def test_hidden_callback(_mod):
    try:
        _mod.hidden_callback(2)
    except Exception as msg:
        assert str(msg).startswith("Callback global_f not defined")

    try:
        _mod.hidden_callback2(2)
    except Exception as msg:
        assert str(msg).startswith("cb: Callback global_f not defined")

    _mod.global_f = lambda x: x + 1
    r = _mod.hidden_callback(2)
    assert r == 3

    _mod.global_f = lambda x: x + 2
    r = _mod.hidden_callback(2)
    assert r == 4

    del _mod.global_f
    try:
        _mod.hidden_callback(2)
    except Exception as msg:
        assert str(msg).startswith("Callback global_f not defined")

    _mod.global_f = lambda x=0: x + 3
    r = _mod.hidden_callback(2)
    assert r == 5

    # reproducer of gh18341
    r = _mod.hidden_callback2(2)
    assert r == 3


# TestF90Callback


@pytest.mark.parametrize("_mod", ["f90_callback_spec"], indirect=True)
def test_gh17797(_mod):
    def incr(x):
        return x + 123

    y = np.array([1, 2, 3], dtype=np.int64)
    r = _mod.gh17797(incr, y)
    assert r == 123 + 1 + 2 + 3


# TestGH18335


@pytest.mark.parametrize("_mod", ["gh18335_spec"], indirect=True)
def test_gh18335(_mod):
    def foo(x):
        x[0] += 1

    r = _mod.gh18335(foo)
    assert r == 123 + 1


@pytest.mark.parametrize("_mod", ["gh25211_spec"], indirect=True)
def test_gh25211(_mod):
    def bar(x):
        return x * x

    res = _mod.foo(bar)
    assert res == 110
