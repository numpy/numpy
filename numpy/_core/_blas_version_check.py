"""
OpenBLAS version detection and threading safety warnings for NumPy.

This module detects whether the installed OpenBLAS version is affected by
the thread-unsafe scratch buffer bug (fixed in OpenBLAS >= 0.3.30) and
issues appropriate RuntimeWarnings so users can take action.

Background
----------
OpenBLAS <= 0.3.29 contains a race condition in its multi-level blocked
DGEMM implementation. When the computation is large enough to use the
blocked kernel path (typically arrays with >= ~50k–100k rows), two threads
can simultaneously read/write the same internal scratch buffer, producing
silently incorrect results.

The bug specifically affects ``np.dot(A, B.T)`` and similar operations
where one argument is a non-contiguous (transposed) array, because only
then does OpenBLAS need to allocate and use the scratch buffer.

References
----------
- NumPy issue:  https://github.com/numpy/numpy/issues/29391
- NumPy PR fix: https://github.com/numpy/numpy/pull/30049
- OpenBLAS fix: https://github.com/OpenMathLib/OpenBLAS/pull/XXXX
"""

import re
import threading
import warnings


# The first OpenBLAS version that contains the thread-safety fix.
_SAFE_OPENBLAS_VERSION = (0, 3, 30)
_SAFE_OPENBLAS_VERSION_STR = "0.3.30"

# Per-thread flag to avoid spamming repeated warnings in hot loops.
_warned_state = threading.local()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_version(version_str):
    """
    Parse a version string like "0.3.29" into a comparable integer tuple.

    Returns
    -------
    tuple of int, e.g. (0, 3, 29), or None if parsing fails.
    """
    if not version_str or version_str == "unknown":
        return None
    match = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", version_str.strip())
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3) or "0")
    return (major, minor, patch)


def _get_openblas_version():
    """
    Attempt to extract the OpenBLAS version from NumPy's build/runtime info.

    Tries multiple strategies in order of reliability, because the NumPy
    config API changed significantly between NumPy 1.x and 2.x.

    Returns
    -------
    str or None
        Version string like ``"0.3.29"``, ``"unknown"`` (OpenBLAS detected
        but version not determinable), or ``None`` (not using OpenBLAS).
    """
    import numpy as np

    # --- Strategy 1: NumPy >= 2.0 show_config(mode="dicts") ---------------
    try:
        cfg = np.show_config(mode="dicts")
        build_deps = cfg.get("Build Dependencies", {})
        blas_cfg = build_deps.get("blas", {})
        name = str(blas_cfg.get("name", "")).lower()
        version = str(blas_cfg.get("version", ""))
        if "openblas" in name and version:
            return version
        # Also check the "found" field
        found = str(blas_cfg.get("found", "")).lower()
        if "openblas" in found and version:
            return version
    except Exception:
        pass

    # --- Strategy 2: Runtime BLAS info via np.core.multiarray -------------
    try:
        info_list = np.__config__.blas_ilp64_opt_info  # noqa: F841
    except AttributeError:
        pass

    # --- Strategy 3: Parse np.__config__ for older NumPy ------------------
    try:
        blas_info = np.__config__.blas_opt_info
        libs = blas_info.get("libraries", [])
        if any("openblas" in lib.lower() for lib in libs):
            # Knows it's OpenBLAS but can't determine the version from this API
            return "unknown"
        # Not OpenBLAS (MKL, ATLAS, etc.) → not affected
        return None
    except AttributeError:
        pass

    # --- Strategy 4: scipy-openblas bundle (NumPy's preferred bundling) ---
    try:
        # NumPy >= 1.26 bundles scipy-openblas; its version is embedded in
        # the shared library filename like: libscipy_openblas64_-56d6093b.so
        # The OpenBLAS version can be read from the library's openblas_config.
        import ctypes
        import ctypes.util

        for lib_name in ("openblas", "scipy_openblas64_", "scipy_openblas"):
            path = ctypes.util.find_library(lib_name)
            if path:
                try:
                    lib = ctypes.CDLL(path)
                    config_fn = lib.openblas_get_config
                    config_fn.restype = ctypes.c_char_p
                    config_str = config_fn().decode("utf-8", errors="replace")
                    # Config string looks like:
                    # "OpenBLAS 0.3.29 DYNAMIC_ARCH ... PTHREADS MAX_THREADS=128"
                    m = re.search(r"OpenBLAS\s+(\d+\.\d+(?:\.\d+)?)", config_str)
                    if m:
                        return m.group(1)
                except Exception:
                    continue
    except Exception:
        pass

    return None


def _is_threading_safe(version_str):
    """
    Determine whether the given OpenBLAS version is safe for concurrent dot.

    Parameters
    ----------
    version_str : str or None
        Version string returned by ``_get_openblas_version()``.

    Returns
    -------
    True   — safe (>= 0.3.30, or not OpenBLAS)
    False  — known buggy (< 0.3.30)
    None   — OpenBLAS detected but version unknown; treat as potentially unsafe
    """
    if version_str is None:
        return True  # Not using OpenBLAS; not affected
    if version_str == "unknown":
        return None  # Can't determine; warn cautiously
    parsed = _parse_version(version_str)
    if parsed is None:
        return None
    return parsed >= _SAFE_OPENBLAS_VERSION


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def warn_if_unsafe_openblas_threading():
    """
    Emit a :class:`RuntimeWarning` if NumPy's OpenBLAS is affected by the
    concurrent ``np.dot`` race condition (numpy/numpy#29391).

    The warning is emitted **at most once per thread** to avoid flooding
    logs in hot paths.  It informs the user about the issue and the safe
    workaround (``np.ascontiguousarray``).

    This function is a no-op when:

    - NumPy is not linked against OpenBLAS, or
    - The OpenBLAS version is >= 0.3.30 (the fix release), or
    - The warning has already been emitted in this thread.
    """
    if getattr(_warned_state, "emitted", False):
        return

    version = _get_openblas_version()
    safe = _is_threading_safe(version)

    if safe is False:
        _warned_state.emitted = True
        warnings.warn(
            f"NumPy is linked against OpenBLAS {version}, which contains a "
            f"known race condition affecting concurrent np.dot() calls when "
            f"one argument is a transposed (non-contiguous) array "
            f"(see https://github.com/numpy/numpy/issues/29391). "
            f"Results may be silently INCORRECT in multi-threaded code. "
            f"\n\nWorkarounds:"
            f"\n  1. Upgrade OpenBLAS to >= {_SAFE_OPENBLAS_VERSION_STR}"
            f"\n  2. Use np.ascontiguousarray(B) before np.dot(A, B.T)"
            f"\n  3. Use numpy._thread_safe_dot.safe_dot(A, B.T)",
            RuntimeWarning,
            stacklevel=3,
        )
    elif safe is None:
        _warned_state.emitted = True
        warnings.warn(
            f"NumPy is using OpenBLAS (version could not be determined). "
            f"If the version is < {_SAFE_OPENBLAS_VERSION_STR}, concurrent "
            f"np.dot() calls with transposed arguments may silently produce "
            f"incorrect results (numpy/numpy#29391). "
            f"Use np.ascontiguousarray(B) before np.dot(A, B.T) as a safe "
            f"workaround.",
            RuntimeWarning,
            stacklevel=3,
        )


def get_blas_threading_info():
    """
    Return a summary of the BLAS configuration and its threading safety status.

    Useful for diagnostic output and test skipping logic.

    Returns
    -------
    dict with keys:

    ``library``
        Name of the BLAS library in use (``"openblas"``, ``"mkl"``,
        ``"atlas"``, or ``"unknown"``).
    ``version``
        Version string (e.g. ``"0.3.29"``), ``"unknown"``, or ``None``.
    ``is_threading_safe``
        ``True`` / ``False`` / ``None`` (see :func:`_is_threading_safe`).
    ``safe_version_threshold``
        The first version known to be safe: ``"0.3.30"``.
    ``issue_url``
        Link to the GitHub issue.
    """
    version = _get_openblas_version()
    safe = _is_threading_safe(version)

    # Determine library name
    if version is not None:
        library = "openblas"
    else:
        library = "unknown"
        try:
            import numpy as np
            blas_info = np.__config__.blas_opt_info
            libs = blas_info.get("libraries", [])
            if any("mkl" in l.lower() for l in libs):
                library = "mkl"
                safe = True   # Intel MKL is thread-safe
            elif any("atlas" in l.lower() for l in libs):
                library = "atlas"
        except Exception:
            pass

    return {
        "library": library,
        "version": version,
        "is_threading_safe": safe,
        "safe_version_threshold": _SAFE_OPENBLAS_VERSION_STR,
        "issue_url": "https://github.com/numpy/numpy/issues/29391",
    }
