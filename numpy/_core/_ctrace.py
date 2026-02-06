"""
NumPy Debug-only C-Level Function Call Tracing - Python Interface

This module provides a Python interface to the debug-only C-level function
call tracing mechanism. It is only functional when NumPy is built with
NUMPY_DEBUG_CTRACE defined.

See numpy/prof.md for the full design document.

Example usage:
    >>> from numpy._core._ctrace import CTrace
    >>> with CTrace() as tracer:
    ...     # NumPy operations here will be traced
    ...     arr = np.array([1, 2, 3])
    ...     result = np.sum(arr)
    >>> tracer.dump_stack()  # Print the call stack

Note: This module is for debugging purposes only and is not part of the
public NumPy API.
"""

import contextlib
import sys
from typing import Callable, List, Optional

__all__ = ['CTrace', 'is_available', 'enable', 'disable', 'dump_stack']


def _check_availability():
    """Check if C-level tracing is available in this build."""
    try:
        from numpy._core import _ctrace_impl
        return hasattr(_ctrace_impl, 'enable')
    except ImportError:
        return False


_AVAILABLE = None


def is_available() -> bool:
    """
    Check if C-level tracing is available.

    Returns
    -------
    bool
        True if tracing is available (NumPy built with NUMPY_DEBUG_CTRACE),
        False otherwise.
    """
    global _AVAILABLE
    if _AVAILABLE is None:
        _AVAILABLE = _check_availability()
    return _AVAILABLE


def _get_impl():
    """Get the implementation module, raising if unavailable."""
    if not is_available():
        raise RuntimeError(
            "C-level tracing is not available. "
            "NumPy must be built with NUMPY_DEBUG_CTRACE defined. "
            "See numpy/prof.md for details."
        )
    from numpy._core import _ctrace_impl
    return _ctrace_impl


def enable() -> None:
    """
    Enable C-level tracing for the current thread.

    Raises
    ------
    RuntimeError
        If tracing is not available in this build.
    """
    _get_impl().enable()


def disable() -> None:
    """
    Disable C-level tracing for the current thread.

    Raises
    ------
    RuntimeError
        If tracing is not available in this build.
    """
    _get_impl().disable()


def is_enabled() -> bool:
    """
    Check if tracing is currently enabled for this thread.

    Returns
    -------
    bool
        True if tracing is enabled, False otherwise.

    Raises
    ------
    RuntimeError
        If tracing is not available in this build.
    """
    return _get_impl().is_enabled()


def get_depth() -> int:
    """
    Get the current call stack depth.

    Returns
    -------
    int
        Current nesting depth (0 if at top level).

    Raises
    ------
    RuntimeError
        If tracing is not available in this build.
    """
    return _get_impl().get_depth()


def snapshot(max_size: int = 100) -> List[int]:
    """
    Snapshot the current C call stack.

    Parameters
    ----------
    max_size : int, optional
        Maximum number of stack frames to capture. Default is 100.

    Returns
    -------
    list of int
        List of function addresses (as integers) from top to bottom.

    Raises
    ------
    RuntimeError
        If tracing is not available in this build.
    """
    return _get_impl().snapshot(max_size)


def resolve_symbol(addr: int) -> Optional[str]:
    """
    Resolve a function address to a symbol name.

    Parameters
    ----------
    addr : int
        Function address to resolve.

    Returns
    -------
    str or None
        Symbol name if resolution succeeded, None otherwise.

    Raises
    ------
    RuntimeError
        If tracing is not available in this build.
    """
    return _get_impl().resolve_symbol(addr)


def dump_stack() -> None:
    """
    Dump the current C call stack to stderr.

    Raises
    ------
    RuntimeError
        If tracing is not available in this build.
    """
    _get_impl().dump_stack()


def set_callback(callback: Optional[Callable[[int, int, int, bool], None]]) -> None:
    """
    Set a custom callback for trace events.

    Parameters
    ----------
    callback : callable or None
        A function with signature (func_addr, caller_addr, depth, is_entry).
        If None, the default callback (print to stderr) is used.

    Raises
    ------
    RuntimeError
        If tracing is not available in this build.
    """
    _get_impl().set_callback(callback)


class CTrace:
    """
    Context manager for C-level function call tracing.

    This class provides a convenient way to enable tracing for a specific
    block of code.

    Parameters
    ----------
    callback : callable, optional
        Custom callback for trace events. If not provided, traces are
        printed to stderr.
    enabled : bool, optional
        Whether to actually enable tracing. Default is True. Set to False
        to create a no-op context manager.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy._core._ctrace import CTrace
    >>> with CTrace():
    ...     arr = np.array([1, 2, 3])
    ...     result = np.sum(arr)

    Using a custom callback:
    >>> traces = []
    >>> def my_callback(func, caller, depth, is_entry):
    ...     traces.append((func, depth, is_entry))
    >>> with CTrace(callback=my_callback):
    ...     np.zeros(10)
    >>> print(f"Captured {len(traces)} trace events")
    """

    def __init__(
        self,
        callback: Optional[Callable[[int, int, int, bool], None]] = None,
        enabled: bool = True
    ):
        self._callback = callback
        self._enabled = enabled and is_available()
        self._was_enabled = False
        self._traces: List[tuple] = []

    def __enter__(self) -> 'CTrace':
        if not self._enabled:
            return self

        impl = _get_impl()

        # Save previous state
        self._was_enabled = impl.is_enabled()

        # Set callback if provided
        if self._callback is not None:
            impl.set_callback(self._callback)

        # Enable tracing
        impl.enable()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._enabled:
            return

        impl = _get_impl()

        # Restore previous state
        if not self._was_enabled:
            impl.disable()

        # Reset callback to default
        if self._callback is not None:
            impl.set_callback(None)

    def snapshot(self, max_size: int = 100) -> List[int]:
        """Take a snapshot of the current call stack."""
        if not self._enabled:
            return []
        return snapshot(max_size)

    def dump_stack(self) -> None:
        """Dump the current call stack to stderr."""
        if self._enabled:
            dump_stack()


class TraceCollector:
    """
    A trace collector that stores all trace events.

    This is useful for post-hoc analysis of execution paths.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy._core._ctrace import TraceCollector
    >>> collector = TraceCollector()
    >>> with collector:
    ...     np.zeros(10)
    >>> print(f"Max depth: {collector.max_depth}")
    >>> for event in collector.events[:5]:
    ...     print(event)
    """

    def __init__(self, max_events: int = 100000):
        self._max_events = max_events
        self._events: List[tuple] = []
        self._max_depth = 0
        self._ctrace: Optional[CTrace] = None

    def _callback(self, func: int, caller: int, depth: int, is_entry: bool) -> None:
        if len(self._events) < self._max_events:
            self._events.append((func, caller, depth, is_entry))
            if depth > self._max_depth:
                self._max_depth = depth

    def __enter__(self) -> 'TraceCollector':
        self._events.clear()
        self._max_depth = 0
        self._ctrace = CTrace(callback=self._callback)
        self._ctrace.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._ctrace:
            self._ctrace.__exit__(exc_type, exc_val, exc_tb)
            self._ctrace = None

    @property
    def events(self) -> List[tuple]:
        """List of (func_addr, caller_addr, depth, is_entry) tuples."""
        return self._events

    @property
    def max_depth(self) -> int:
        """Maximum call depth observed."""
        return self._max_depth

    def get_call_tree(self) -> dict:
        """
        Build a call tree from the collected events.

        Returns
        -------
        dict
            Nested dictionary representing the call tree.
        """
        root: dict = {'children': {}, 'count': 0}
        stack = [root]

        for func, caller, depth, is_entry in self._events:
            if is_entry:
                # Ensure stack matches depth
                while len(stack) > depth + 1:
                    stack.pop()

                parent = stack[-1]
                if func not in parent['children']:
                    parent['children'][func] = {'children': {}, 'count': 0}
                parent['children'][func]['count'] += 1
                stack.append(parent['children'][func])
            else:
                if len(stack) > 1:
                    stack.pop()

        return root

    def print_summary(self, file=None) -> None:
        """Print a summary of the collected traces."""
        if file is None:
            file = sys.stderr

        print(f"Trace Summary:", file=file)
        print(f"  Total events: {len(self._events)}", file=file)
        print(f"  Max depth: {self._max_depth}", file=file)

        # Count unique functions
        funcs = set()
        for func, caller, depth, is_entry in self._events:
            funcs.add(func)
        print(f"  Unique functions: {len(funcs)}", file=file)
