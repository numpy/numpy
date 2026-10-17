"""Pin down NumPy's behaviour in a sub-interpreter.

``_multiarray_umath`` declares
``Py_mod_multiple_interpreters = Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED``,
which CPython only enforces when the interpreter sets
``check_multi_interp_extensions = 1``.  A legacy interpreter does not, so NumPy
still imports there and only warns.  Both halves are tested, in subprocesses:
NumPy must not already be loaded in the main interpreter, and a sub-interpreter
that imports it shares its C state.
"""
import sys
import textwrap

import pytest

from numpy.testing import HAS_SUBPROCESSES
from numpy.testing._private.utils import run_subprocess

pytestmark = pytest.mark.skipif(
    not HAS_SUBPROCESSES, reason="platform cannot start subprocesses")

# Run inside the sub-interpreter.  Kept separate because `textwrap.dedent` will
# not dedent an outer template that embeds a flush-left script.
IMPORT_NUMPY = textwrap.dedent("""
    import warnings

    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        import numpy
    print("imported", [(w.category.__name__, str(w.message)) for w in log],
          flush=True)
    """)


@pytest.mark.skipif(
    sys.version_info < (3, 14), reason="`concurrent.interpreters` needs Python 3.14")
def test_isolated_interpreter_refuses_numpy():
    # `concurrent.interpreters` sets ``check_multi_interp_extensions = 1``,
    # which is what turns the declaration into an ImportError.
    code = textwrap.dedent("""
        from concurrent import interpreters

        interp = interpreters.create()
        try:
            interp.exec("import numpy")
        except interpreters.ExecutionFailed as exc:
            print(exc.excinfo.type.__name__, repr(exc.excinfo.msg))
        finally:
            interp.close()
        """)
    out = run_subprocess((sys.executable, "-c", code)).stdout
    assert "ImportError" in out
    assert "does not support loading in subinterpreters" in out


@pytest.mark.skipif(
    sys.version_info < (3, 13), reason="`_interpreters.new_config` needs Python 3.13")
def test_legacy_interpreter_imports_numpy_with_a_warning():
    # A legacy interpreter leaves ``check_multi_interp_extensions`` at 0, so the
    # declaration is not enforced and all NumPy does is warn from
    # `_reload_guard`.  Only the private `_interpreters` module builds that
    # configuration; `concurrent.interpreters` always enforces the check.
    code = textwrap.dedent("""
        import _interpreters

        interp = _interpreters.create(_interpreters.new_config("legacy"))
        try:
            exc = _interpreters.exec(interp, {script!r})
            if exc is not None:
                raise SystemExit("sub-interpreter failed: " + str(exc))
        finally:
            _interpreters.destroy(interp)
        """).format(script=IMPORT_NUMPY)
    out = run_subprocess((sys.executable, "-c", code)).stdout
    assert "imported [('UserWarning'" in out
    assert "does not properly support sub-interpreters" in out
