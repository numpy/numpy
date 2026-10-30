import sys
import textwrap

import pytest

from numpy.testing import HAS_SUBPROCESSES
from numpy.testing._private.utils import NOGIL_BUILD, run_subprocess


@pytest.mark.skipif(not HAS_SUBPROCESSES, reason="platform cannot start subprocesses")
@pytest.mark.skipif(NOGIL_BUILD, reason="free-threaded Python enforces the "
                    "multi-interpreter check in every sub-interpreter")
def test_legacy_interpreter_imports_numpy_with_a_warning():
    pytest.importorskip("_interpreters", reason="needs Python 3.13+")
    # In a new process, because NumPy can only be loaded once per process.
    subinterp_code = textwrap.dedent("""
        import warnings

        with warnings.catch_warnings(record=True) as log:
            warnings.simplefilter("always")
            import numpy as np

        msgs = [str(w.message) for w in log if w.category is UserWarning]
        if not any("does not properly support sub-interpreters" in m
                   for m in msgs):
            raise SystemExit(f"no sub-interpreter warning, got: {msgs}")
        if np.arange(4).sum() != 6:
            raise SystemExit("NumPy does not work in the sub-interpreter")
        """)
    code = textwrap.dedent("""
        import sys
        import _interpreters

        interp = _interpreters.create("legacy")
        try:
            exc = _interpreters.exec(interp, sys.argv[1])
        finally:
            _interpreters.destroy(interp)
        if exc is not None:
            raise SystemExit(exc.errdisplay)
        """)
    run_subprocess((sys.executable, "-c", code, subinterp_code))
