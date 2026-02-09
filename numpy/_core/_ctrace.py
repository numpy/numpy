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

__all__ = ['CTrace', 'TraceCollector', 'is_available', 'enable', 'disable', 'dump_stack']


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
    block of code and access the collected traces via utility methods.

    Parameters
    ----------
    callback : callable, optional
        Custom callback for trace events. If provided, this callback is
        called for each event AND traces are still collected internally.
    enabled : bool, optional
        Whether to actually enable tracing. Default is True. Set to False
        to create a no-op context manager.
    max_events : int, optional
        Maximum number of trace events to collect. Default is 100000.
    filter_func : callable, optional
        A function that takes (func_name, depth, is_entry) and returns
        True to include the event, False to skip it.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy._core._ctrace import CTrace
    >>> with CTrace() as ct:
    ...     arr = np.array([1, 2, 3])
    ...     result = np.sum(arr)
    >>> ct.print_trace()  # Print the collected trace

    Using filters:
    >>> with CTrace(filter_func=lambda name, d, e: 'ufunc' in name) as ct:
    ...     np.zeros(10)
    >>> ct.print_summary()
    """

    # Common noisy symbols to filter out by default
    DEFAULT_SKIP_PATTERNS = [
        "Py_TYPE", "Py_INCREF", "Py_DECREF", "Py_XINCREF", "Py_XDECREF",
        "_Py_IsImmortal", "_Py_NewRef", "_ZL7Py_TYPE", "_ZL17PyType_Has",
        "Py_SIZE", "PyType_HasFeature", "Py_IS_TYPE", "_ZL9Py_INCREF",
        "_ZL10Py_IS_TYPE", "_ZL9Py_DECREF", "_ZL10Py_XDECREF"
    ]

    def __init__(
        self,
        callback: Optional[Callable[[int, int, int, bool], None]] = None,
        enabled: bool = True,
        max_events: int = 100000,
        filter_func: Optional[Callable[[str, int, bool], bool]] = None
    ):
        self._user_callback = callback
        self._enabled = enabled and is_available()
        self._was_enabled = False
        self._max_events = max_events
        self._filter_func = filter_func
        self._events: List[tuple] = []  # (func_addr, caller_addr, depth, is_entry, func_name)
        self._max_depth = 0
        self._symbol_cache: dict = {}

    def _resolve_cached(self, addr: int) -> str:
        """Resolve symbol with caching."""
        if addr not in self._symbol_cache:
            name = resolve_symbol(addr)
            self._symbol_cache[addr] = name if name else f"0x{addr:x}"
        return self._symbol_cache[addr]

    def _internal_callback(self, func: int, caller: int, depth: int, is_entry: bool) -> None:
        """Internal callback that collects traces."""
        func_name = self._resolve_cached(func)

        # Apply filter if provided
        if self._filter_func is not None:
            if not self._filter_func(func_name, depth, is_entry):
                return

        # Collect event
        if len(self._events) < self._max_events:
            self._events.append((func, caller, depth, is_entry, func_name))
            if depth > self._max_depth:
                self._max_depth = depth

        # Call user callback if provided
        if self._user_callback is not None:
            self._user_callback(func, caller, depth, is_entry)

    def __enter__(self) -> 'CTrace':
        if not self._enabled:
            return self

        impl = _get_impl()

        # Clear previous data
        self._events.clear()
        self._max_depth = 0
        self._symbol_cache.clear()

        # Save previous state
        self._was_enabled = impl.is_enabled()

        # Set our internal callback
        impl.set_callback(self._internal_callback)

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
        impl.set_callback(None)

    @property
    def events(self) -> List[tuple]:
        """List of (func_addr, caller_addr, depth, is_entry, func_name) tuples."""
        return self._events

    @property
    def max_depth(self) -> int:
        """Maximum call depth observed."""
        return self._max_depth

    @property
    def event_count(self) -> int:
        """Number of trace events collected."""
        return len(self._events)

    def snapshot(self, max_size: int = 100) -> List[int]:
        """Take a snapshot of the current call stack."""
        if not self._enabled:
            return []
        return snapshot(max_size)

    def dump_stack(self) -> None:
        """Dump the current call stack to stderr."""
        if self._enabled:
            dump_stack()

    def print_trace(
        self,
        file=None,
        max_depth: Optional[int] = None,
        skip_patterns: Optional[List[str]] = None,
        show_exit: bool = True
    ) -> None:
        """
        Print the collected trace as an indented call tree.

        Parameters
        ----------
        file : file-like, optional
            Output file. Defaults to sys.stdout.
        max_depth : int, optional
            Maximum depth to print. None means no limit.
        skip_patterns : list of str, optional
            List of substrings to filter out. If None, uses DEFAULT_SKIP_PATTERNS.
            Pass empty list [] to disable filtering.
        show_exit : bool, optional
            Whether to show function exit events. Default is True.
        """
        if file is None:
            file = sys.stdout

        if skip_patterns is None:
            skip_patterns = self.DEFAULT_SKIP_PATTERNS

        for func, caller, depth, is_entry, func_name in self._events:
            # Apply skip patterns
            if skip_patterns and any(s in func_name for s in skip_patterns):
                continue

            # Apply max depth filter
            if max_depth is not None and depth > max_depth:
                continue

            # Skip exit events if requested
            if not show_exit and not is_entry:
                continue

            indent = "  " * min(depth, 40)
            arrow = ">" if is_entry else "<"
            print(f"{indent}{arrow} {func_name}", file=file)

    def print_summary(self, file=None) -> None:
        """
        Print a summary of the collected traces.

        Parameters
        ----------
        file : file-like, optional
            Output file. Defaults to sys.stdout.
        """
        if file is None:
            file = sys.stdout

        print(f"Trace Summary:", file=file)
        print(f"  Total events: {len(self._events)}", file=file)
        print(f"  Max depth: {self._max_depth}", file=file)

        # Count unique functions
        funcs = set(e[4] for e in self._events)  # func_name is at index 4
        print(f"  Unique functions: {len(funcs)}", file=file)

    def print_hotspots(self, top_n: int = 10, file=None) -> None:
        """
        Print the most frequently called functions.

        Parameters
        ----------
        top_n : int, optional
            Number of top functions to show. Default is 10.
        file : file-like, optional
            Output file. Defaults to sys.stdout.
        """
        if file is None:
            file = sys.stdout

        # Count entry events per function
        counts: dict = {}
        for func, caller, depth, is_entry, func_name in self._events:
            if is_entry:
                counts[func_name] = counts.get(func_name, 0) + 1

        # Sort by count
        sorted_funcs = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        print(f"Top {top_n} hotspots:", file=file)
        for i, (name, count) in enumerate(sorted_funcs[:top_n], 1):
            print(f"  {i:2d}. {count:6d} calls: {name}", file=file)

    def get_call_tree(self) -> dict:
        """
        Build a call tree from the collected events.

        Returns
        -------
        dict
            Nested dictionary representing the call tree with keys:
            'name', 'children', 'count'.
        """
        root: dict = {'name': 'root', 'children': {}, 'count': 0}
        stack = [root]

        for func, caller, depth, is_entry, func_name in self._events:
            if is_entry:
                # Ensure stack matches depth
                while len(stack) > depth + 1:
                    stack.pop()

                parent = stack[-1]
                if func_name not in parent['children']:
                    parent['children'][func_name] = {
                        'name': func_name,
                        'children': {},
                        'count': 0
                    }
                parent['children'][func_name]['count'] += 1
                stack.append(parent['children'][func_name])
            else:
                if len(stack) > 1:
                    stack.pop()

        return root

    def get_flat_profile(self) -> List[tuple]:
        """
        Get a flat profile of function call counts.

        Returns
        -------
        list of tuple
            List of (func_name, call_count) sorted by count descending.
        """
        counts: dict = {}
        for func, caller, depth, is_entry, func_name in self._events:
            if is_entry:
                counts[func_name] = counts.get(func_name, 0) + 1

        return sorted(counts.items(), key=lambda x: x[1], reverse=True)


# TraceCollector is now deprecated - use CTrace directly which has all the same functionality
TraceCollector = CTrace  # Alias for backward compatibility
