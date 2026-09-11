"""Shims that let the NumPy test suite import, collect, and run cleanly when the
*optional* ``hypothesis`` dependency is not installed.

``hypothesis`` is a supplement to the deterministic pytest suite (property-based
fuzzing).  It recently grew a Rust extension, which makes it hard to install on
platforms without a prebuilt wheel (riscv64, win-arm64, ppc64le, s390x, CPython
pre-releases, QEMU-emulated targets, ...).  To keep it optional, test modules
import the names they need from here instead of straight from ``hypothesis``.

When ``hypothesis`` is available, these names are simply re-exported.  When it is
missing, ``given``/``settings`` become identity decorators (so the decorated test
is still collected as a plain function) and the strategy objects become no-op
stand-ins (so evaluating the decorator arguments does not raise at import time).
Each property-based test is expected to also carry
``@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis is not installed")``
so it is reported as skipped rather than executed.
"""

try:
    import hypothesis
    from hypothesis import given, settings, strategies
    from hypothesis.extra import numpy as hynp
    from hypothesis.extra.numpy import arrays
    from hypothesis.strategies import sampled_from
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

    def given(*args, **kwargs):
        # Identity decorator: leaves the test a function so pytest collects it
        # and the accompanying skipif marker takes effect.
        return lambda func: func

    def settings(*args, **kwargs):
        return lambda func: func

    class _Swallow:
        """Callable / attribute / ``|``-chainable no-op standing in for the
        ``hypothesis`` strategy objects used in decorator arguments."""

        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, name):
            return self

        def __or__(self, other):
            return self

        __ror__ = __or__

    _swallow = _Swallow()
    strategies = hynp = arrays = sampled_from = _swallow

    class _HypothesisStub:
        """Stand-in for the top-level ``hypothesis`` module, for tests that use
        ``@hypothesis.given(...)`` / ``hypothesis.strategies`` directly."""

        given = staticmethod(given)
        settings = staticmethod(settings)
        strategies = _swallow

        def __getattr__(self, name):
            return _swallow

    hypothesis = _HypothesisStub()

# Alias used by some test modules (``import hypothesis.strategies as st``).
st = strategies
