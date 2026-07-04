"""
Thread-safe wrappers for np.dot, np.matmul, and np.inner.

Motivation
----------
OpenBLAS 0.3.28 and 0.3.29 have a critical race condition in their
multi-level blocked DGEMM implementation (driver/level3/level3_thread.c)
that causes silently **incorrect results** when ``np.dot(A, B.T)`` is
called concurrently from multiple Python threads on large arrays.

Root cause (confirmed via OpenBLAS source analysis)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OpenBLAS template-compiles ``gemm_driver`` into four separate functions,
one per transpose variant (NN, NT, TN, TT).  A ``level3_lock`` mutex meant
to serialize concurrent GEMM calls is declared as a function-local ``static``
variable — giving each variant its **own, independent lock**.  Concurrent
calls with different transpose flags (e.g., one thread uses NN, another uses
NT) do not serialize and race on shared internal state → silently wrong results.

This is a pure OpenBLAS bug.  NumPy's own code (``PyArray_MatrixProduct2``,
``cblas_matmat``) is correct.  The fix is in OpenBLAS >= 0.3.30, shipped
with NumPy >= 2.3.5 via PR #30049.

Correct workarounds on affected NumPy / OpenBLAS versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **threadpoolctl** (recommended)::

       from threadpoolctl import threadpool_limits
       with threadpool_limits(limits=1, user_api="blas"):
           result = np.dot(A, B.T)

2. **Environment variable** (process-wide)::

       OPENBLAS_NUM_THREADS=1 python your_script.py

3. **threading.Lock** (serialises np.dot calls across Python threads)::

       _dot_lock = threading.Lock()
       with _dot_lock:
           result = np.dot(A, B.T)

Note on ``np.ascontiguousarray``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Making ``B.T`` contiguous forces the ``NN`` DGEMM variant instead of ``NT``,
which does avoid the specific NN vs NT lock mismatch — however it does NOT
protect against NT vs TT or other variant pairs that may race.  It is not a
comprehensive fix.  The recommended workaround is threadpoolctl.

This module provides:

``safe_dot(a, b)``       — thread-safe ``np.dot`` via threadpoolctl or Lock
``safe_matmul(a, b)``    — thread-safe ``np.matmul`` / ``@``
``safe_inner(a, b)``     — thread-safe ``np.inner``
``dot``, ``matmul``, ``inner`` — convenience aliases

References
----------
- Bug report: https://github.com/numpy/numpy/issues/29391
- NumPy fix:  https://github.com/numpy/numpy/pull/30049 (NumPy 2.3.5)
- OpenBLAS:   https://github.com/OpenMathLib/OpenBLAS/issues/5836
"""

from __future__ import annotations

import threading

import numpy as np

try:
    from numpy._core._blas_version_check import (
        get_blas_threading_info,
        warn_if_unsafe_openblas_threading,
    )
    _HAS_VERSION_CHECK = True
except ImportError:
    _HAS_VERSION_CHECK = False

    def warn_if_unsafe_openblas_threading():
        pass

    def get_blas_threading_info():
        return {"is_threading_safe": None}


__all__ = [
    "safe_dot",
    "safe_matmul",
    "safe_inner",
    "dot",
    "matmul",
    "inner",
    "configure",
]


# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

# Strategy to use for thread safety:
#   "threadpoolctl" — limit BLAS threads to 1 within each call (recommended)
#   "lock"          — serialize all calls via a module-level threading.Lock
#   "auto"          — try threadpoolctl, fall back to lock
_STRATEGY = "auto"

# The module-level lock used when strategy == "lock"
_DOT_LOCK = threading.Lock()

# Cached threadpoolctl availability
try:
    import threadpoolctl as _threadpoolctl
    _HAS_THREADPOOLCTL = True
except ImportError:
    _threadpoolctl = None
    _HAS_THREADPOOLCTL = False


def configure(strategy: str = "auto") -> None:
    """
    Configure the thread-safety strategy used by ``safe_dot`` and friends.

    Parameters
    ----------
    strategy : {"auto", "threadpoolctl", "lock"}
        ``"auto"``         — Use threadpoolctl if available, else lock.
        ``"threadpoolctl"`` — Limit BLAS threads to 1 around each call.
        ``"lock"``          — Use a threading.Lock to serialize BLAS calls.

    Raises
    ------
    ValueError
        If an invalid strategy is specified.
    ImportError
        If ``strategy="threadpoolctl"`` but threadpoolctl is not installed.

    Examples
    --------
    >>> from numpy._thread_safe_dot import configure
    >>> configure("threadpoolctl")   # Use threadpoolctl
    >>> configure("lock")            # Use a Lock (no dependency)
    """
    global _STRATEGY
    valid = {"auto", "threadpoolctl", "lock"}
    if strategy not in valid:
        raise ValueError(f"strategy must be one of {valid!r}, got {strategy!r}")
    if strategy == "threadpoolctl" and not _HAS_THREADPOOLCTL:
        raise ImportError(
            "threadpoolctl is not installed. "
            "Install it with: pip install threadpoolctl"
        )
    _STRATEGY = strategy


# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------

def _call_safely(fn, *args, **kwargs):
    """
    Call ``fn(*args, **kwargs)`` with the active thread-safety strategy.

    ``"threadpoolctl"``
        Wraps the call in ``threadpool_limits(limits=1, user_api="blas")``.
        This limits OpenBLAS to a single internal thread for the duration of
        the call, preventing concurrent variant dispatch inside gemm_driver.
        The BLAS thread count is restored on exit.

    ``"lock"``
        Acquires ``_DOT_LOCK`` before calling ``fn``, serialising all
        safe_dot / safe_matmul / safe_inner calls globally.  This is safe
        but eliminates Python-level parallelism.

    ``"auto"``
        Prefers threadpoolctl if available, falls back to lock.
    """
    strategy = _STRATEGY
    if strategy == "auto":
        strategy = "threadpoolctl" if _HAS_THREADPOOLCTL else "lock"

    if strategy == "threadpoolctl":
        with _threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            return fn(*args, **kwargs)
    else:  # "lock"
        with _DOT_LOCK:
            return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def safe_dot(a, b, out=None):
    """
    Thread-safe replacement for :func:`numpy.dot`.

    Computes ``np.dot(a, b, out=out)`` and guarantees correct results even
    when called concurrently from multiple threads, including when *b* is a
    transposed view (``B.T``), on NumPy installations that bundle OpenBLAS
    0.3.28 or 0.3.29 (see https://github.com/numpy/numpy/issues/29391).

    Parameters
    ----------
    a : array_like
        First operand.
    b : array_like
        Second operand.  May be a transposed / non-contiguous view.
    out : ndarray, optional
        Output array.

    Returns
    -------
    output : ndarray
        The dot product of *a* and *b*.

    Notes
    -----
    Thread safety is achieved by one of:

    - **threadpoolctl** (if installed, default): limits OpenBLAS to 1 thread
      per call, preventing the internal lock contention across GEMM variants.
    - **threading.Lock** (fallback): serialises concurrent BLAS calls.

    Install threadpoolctl for better performance::

        pip install threadpoolctl

    Examples
    --------
    >>> import threading
    >>> import numpy as np
    >>> from numpy._thread_safe_dot import safe_dot
    >>>
    >>> def worker(results, idx):
    ...     A = np.ones((100_000, 3), dtype=np.float64) * idx
    ...     M = np.eye(3, dtype=np.float64)
    ...     results[idx] = safe_dot(A, M.T)   # safe even on OpenBLAS 0.3.28/29
    >>>
    >>> results = [None] * 10
    >>> threads = [threading.Thread(target=worker, args=(results, i))
    ...            for i in range(10)]
    >>> for t in threads: t.start()
    >>> for t in threads: t.join()
    >>> all(np.all(results[i] == i) for i in range(10))
    True

    See Also
    --------
    numpy.dot : Standard dot product.
    safe_matmul : Thread-safe matrix multiplication.
    configure : Choose the thread-safety strategy.
    """
    warn_if_unsafe_openblas_threading()
    if out is not None:
        return _call_safely(np.dot, a, b, out=out)
    return _call_safely(np.dot, a, b)


def safe_matmul(a, b, out=None):
    """
    Thread-safe replacement for :func:`numpy.matmul` (the ``@`` operator).

    Guarantees correct results when called concurrently from multiple threads
    with transposed array arguments, including on OpenBLAS 0.3.28 / 0.3.29
    (see https://github.com/numpy/numpy/issues/29391).

    Parameters
    ----------
    a : array_like
        First matrix operand (≥ 1-D).
    b : array_like
        Second matrix operand.
    out : ndarray, optional
        Output array.

    Returns
    -------
    output : ndarray
        Same as ``np.matmul(a, b, out=out)``.

    Examples
    --------
    >>> import threading, numpy as np
    >>> from numpy._thread_safe_dot import safe_matmul
    >>>
    >>> def worker(results, idx):
    ...     A = np.random.randn(100_000, 4)
    ...     B = np.random.randn(4, 4)
    ...     results[idx] = safe_matmul(A, B.T)
    >>>
    >>> results = [None] * 8
    >>> threads = [threading.Thread(target=worker, args=(results, i))
    ...            for i in range(8)]
    >>> for t in threads: t.start()
    >>> for t in threads: t.join()

    See Also
    --------
    numpy.matmul : Standard matrix multiplication.
    safe_dot : Thread-safe dot product.
    """
    warn_if_unsafe_openblas_threading()
    if out is not None:
        return _call_safely(np.matmul, a, b, out=out)
    return _call_safely(np.matmul, a, b)


def safe_inner(a, b):
    """
    Thread-safe replacement for :func:`numpy.inner`.

    ``np.inner`` dispatches to the same BLAS DGEMM path as ``np.dot`` for
    large arrays and can be affected by the same OpenBLAS race condition.

    Parameters
    ----------
    a : array_like
        First input.
    b : array_like
        Second input.

    Returns
    -------
    output : scalar or ndarray
        Same as ``np.inner(a, b)``.

    See Also
    --------
    numpy.inner : Standard inner product.
    safe_dot : Thread-safe dot product.
    """
    warn_if_unsafe_openblas_threading()
    return _call_safely(np.inner, a, b)


# ---------------------------------------------------------------------------
# Convenience aliases (mirror the np.* naming convention)
# ---------------------------------------------------------------------------
dot = safe_dot
matmul = safe_matmul
inner = safe_inner
