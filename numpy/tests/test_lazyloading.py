import sys
import textwrap

import pytest

from numpy.testing import HAS_SUBPROCESSES
from numpy.testing._private.utils import run_subprocess


@pytest.mark.skipif(not HAS_SUBPROCESSES, reason="platform cannot start subprocesses")
def test_lazy_load():
    # gh-22045. lazyload doesn't import submodule names into the namespace

    # Test within a new process, to ensure that we do not mess with the
    # global state during the test run (could lead to cryptic test failures).
    # This is generally unsafe, especially, since we also reload the C-modules.
    code = textwrap.dedent(r"""
        import sys
        from importlib.util import LazyLoader, find_spec, module_from_spec

        # create lazy load of numpy as np
        spec = find_spec("numpy")
        module = module_from_spec(spec)
        sys.modules["numpy"] = module
        loader = LazyLoader(spec.loader)
        loader.exec_module(module)
        np = module

        # test a subpackage import
        from numpy.lib import recfunctions  # noqa: F401

        # test triggering the import of the package
        np.ndarray
        """)
    run_subprocess((sys.executable, '-c', code))


@pytest.mark.skipif(not HAS_SUBPROCESSES, reason="platform cannot start subprocesses")
@pytest.mark.skipif(sys.version_info < (3, 15),
                    reason="__lazy_modules__ needs Python 3.15")
def test_lazy_modules_not_imported():
    # Modules declared in `__lazy_modules__` are only loaded on first use.
    code = textwrap.dedent(r"""
        import sys
        import numpy as np
        lazy = {"platform", "numpy.linalg", "numpy.polynomial.legendre"}
        assert not lazy & set(sys.modules), lazy & set(sys.modules)
        np.polynomial.Polynomial([1, 2])(3)
        assert "numpy.polynomial.legendre" not in sys.modules
        assert np.polyfit([0, 1, 2], [0, 1, 2], 1).round(3).tolist() == [1.0, 0.0]
        assert "numpy.linalg" in sys.modules
        print("ok")
        """)
    p = run_subprocess([sys.executable, "-c", code])
    assert p.stdout.strip() == "ok"
