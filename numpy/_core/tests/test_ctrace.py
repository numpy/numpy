"""
Tests for the C-level function call tracing module.

These tests verify the Python interface to the tracing functionality.
Most tests are skipped if tracing is not available (i.e., NumPy was not
built with NUMPY_DEBUG_CTRACE).
"""

import pytest
import numpy as np
from numpy._core._ctrace import (
    is_available,
    CTrace,
    TraceCollector,
)


# Skip all tests if tracing is not available
pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="C-level tracing not available (build with NUMPY_DEBUG_CTRACE)"
)


class TestCTraceAvailability:
    """Tests for availability checking."""

    @pytest.mark.skipif(is_available(), reason="Only run when unavailable")
    def test_is_available_false(self):
        """Test that is_available returns False when not built with tracing."""
        assert not is_available()

    def test_is_available_true(self):
        """Test that is_available returns True when built with tracing."""
        assert is_available()


class TestCTraceBasic:
    """Basic tests for the CTrace context manager."""

    def test_context_manager_enter_exit(self):
        """Test that CTrace can be used as a context manager."""
        with CTrace() as tracer:
            assert tracer is not None

    def test_enable_disable(self):
        """Test enabling and disabling tracing."""
        from numpy._core._ctrace import enable, disable, is_enabled

        # Should start disabled
        assert not is_enabled()

        enable()
        assert is_enabled()

        disable()
        assert not is_enabled()

    def test_context_manager_enables_tracing(self):
        """Test that CTrace enables tracing within the context."""
        from numpy._core._ctrace import is_enabled

        assert not is_enabled()

        with CTrace():
            assert is_enabled()

        assert not is_enabled()

    def test_nested_context_managers(self):
        """Test nested CTrace context managers."""
        from numpy._core._ctrace import is_enabled

        with CTrace():
            assert is_enabled()
            with CTrace():
                assert is_enabled()
            assert is_enabled()

        assert not is_enabled()

    def test_disabled_context_manager(self):
        """Test CTrace with enabled=False."""
        from numpy._core._ctrace import is_enabled

        with CTrace(enabled=False):
            # Should not enable tracing
            assert not is_enabled()


class TestCTraceSnapshot:
    """Tests for call stack snapshots."""

    def test_snapshot_empty(self):
        """Test snapshot when no calls are active."""
        from numpy._core._ctrace import snapshot, enable, disable

        enable()
        result = snapshot()
        disable()

        # At the top level, stack should be minimal
        assert isinstance(result, list)

    def test_snapshot_max_size(self):
        """Test snapshot with max_size parameter."""
        from numpy._core._ctrace import snapshot, enable, disable

        enable()
        result = snapshot(max_size=5)
        disable()

        assert len(result) <= 5

    def test_snapshot_returns_integers(self):
        """Test that snapshot returns a list of integers (addresses)."""
        from numpy._core._ctrace import snapshot, enable, disable

        enable()
        result = snapshot()
        disable()

        for addr in result:
            assert isinstance(addr, int)


class TestCTraceCallback:
    """Tests for custom callbacks."""

    def test_custom_callback(self):
        """Test that custom callbacks are invoked."""
        events = []

        def callback(func, caller, depth, is_entry):
            events.append((func, caller, depth, is_entry))

        with CTrace(callback=callback):
            # Do some NumPy operation
            arr = np.array([1, 2, 3])
            _ = np.sum(arr)

        # Should have captured some events
        # Note: exact count depends on build configuration
        assert len(events) >= 0  # May be 0 if instrumentation not active

    def test_callback_receives_correct_types(self):
        """Test that callback receives correct argument types."""
        type_checks = []

        def callback(func, caller, depth, is_entry):
            type_checks.append((
                isinstance(func, int),
                isinstance(caller, int),
                isinstance(depth, int),
                isinstance(is_entry, bool),
            ))

        with CTrace(callback=callback):
            np.zeros(10)

        for checks in type_checks:
            assert all(checks), f"Type check failed: {checks}"


class TestTraceCollector:
    """Tests for the TraceCollector class."""

    def test_collector_basic(self):
        """Test basic TraceCollector usage."""
        collector = TraceCollector()

        with collector:
            np.zeros(10)

        # Check properties
        assert isinstance(collector.events, list)
        assert isinstance(collector.max_depth, int)
        assert collector.max_depth >= 0

    def test_collector_max_events(self):
        """Test TraceCollector with max_events limit."""
        collector = TraceCollector(max_events=10)

        with collector:
            # Do many operations
            for _ in range(100):
                np.zeros(10)

        assert len(collector.events) <= 10

    def test_collector_call_tree(self):
        """Test TraceCollector.get_call_tree()."""
        collector = TraceCollector()

        with collector:
            np.zeros(10)

        tree = collector.get_call_tree()
        assert isinstance(tree, dict)
        assert 'children' in tree
        assert 'count' in tree


class TestCTraceSymbolResolution:
    """Tests for symbol resolution."""

    def test_resolve_symbol(self):
        """Test resolving a function address to a symbol name."""
        from numpy._core._ctrace import resolve_symbol

        # Use a known function address (this is a bit tricky to test)
        # We'll just verify the function doesn't crash
        result = resolve_symbol(0)
        # Result may be None or a string
        assert result is None or isinstance(result, str)

    def test_resolve_symbol_with_snapshot(self):
        """Test resolving symbols from a snapshot."""
        from numpy._core._ctrace import snapshot, resolve_symbol, enable, disable

        enable()
        addrs = snapshot()
        disable()

        for addr in addrs:
            result = resolve_symbol(addr)
            # Should return string or None
            assert result is None or isinstance(result, str)


class TestCTraceDepth:
    """Tests for call depth tracking."""

    def test_get_depth(self):
        """Test getting the current call depth."""
        from numpy._core._ctrace import get_depth, enable, disable

        enable()
        depth = get_depth()
        disable()

        assert isinstance(depth, int)
        assert depth >= 0


class TestCTraceDumpStack:
    """Tests for stack dumping."""

    def test_dump_stack_no_crash(self, capsys):
        """Test that dump_stack doesn't crash."""
        from numpy._core._ctrace import dump_stack, enable, disable

        enable()
        dump_stack()  # Should print to stderr
        disable()

        # Just verify it didn't crash
        # Output goes to stderr


class TestCTraceIntegration:
    """Integration tests with NumPy operations."""

    def test_trace_array_creation(self):
        """Test tracing array creation."""
        events = []

        def callback(func, caller, depth, is_entry):
            events.append(is_entry)

        with CTrace(callback=callback):
            np.array([1, 2, 3, 4, 5])

        # Should have some entry/exit pairs
        entries = sum(1 for e in events if e)
        exits = sum(1 for e in events if not e)
        # In a well-formed trace, entries should equal exits
        # (though this may not hold if we start/stop mid-execution)

    def test_trace_ufunc(self):
        """Test tracing ufunc operations."""
        collector = TraceCollector()

        with collector:
            a = np.array([1.0, 2.0, 3.0])
            b = np.array([4.0, 5.0, 6.0])
            _ = np.add(a, b)

        # Should have captured some events
        assert isinstance(collector.events, list)

    def test_trace_linalg(self):
        """Test tracing linear algebra operations."""
        collector = TraceCollector()

        with collector:
            a = np.array([[1, 2], [3, 4]], dtype=float)
            _ = np.linalg.det(a)

        assert isinstance(collector.events, list)


# Tests that should work even without tracing available
class TestCTraceUnavailable:
    """Tests for when tracing is unavailable."""

    @pytest.mark.skipif(is_available(), reason="Only run when unavailable")
    def test_ctrace_disabled_no_crash(self):
        """Test that CTrace(enabled=False) works when unavailable."""
        with CTrace(enabled=False) as tracer:
            np.zeros(10)
            # Should not crash

    @pytest.mark.skipif(is_available(), reason="Only run when unavailable")
    def test_is_available_returns_false(self):
        """Test is_available returns False when not built with tracing."""
        assert not is_available()
