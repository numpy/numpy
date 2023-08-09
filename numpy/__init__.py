"""
NumPy
=====

Provides
  1. An array object of arbitrary homogeneous items
  2. Fast mathematical operations over arrays
  3. Linear Algebra, Fourier Transforms, Random Number Generation

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided
with the code, and a loose standing reference guide, available from
`the NumPy homepage <https://numpy.org>`_.

We recommend exploring the docstrings using
`IPython <https://ipython.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.  See below for further
instructions.

The docstring examples assume that `numpy` has been imported as ``np``::

  >>> import numpy as np

Code snippets are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

  >>> help(np.sort)
  ... # doctest: +SKIP

For some objects, ``np.info(obj)`` may provide additional help.  This is
particularly true if you see the line "Help on ufunc object:" at the top
of the help() page.  Ufuncs are implemented in C, not Python, for speed.
The native Python help() does not know how to view their help, but our
np.info() function does.

To search for documents containing a keyword, do::

  >>> np.lookfor('keyword')
  ... # doctest: +SKIP

Available subpackages
---------------------
lib
    Basic functions used by several sub-packages.
random
    Core Random Tools
linalg
    Core Linear Algebra Tools
fft
    Core FFT routines
polynomial
    Polynomial tools
testing
    NumPy testing tools
distutils
    Enhancements to distutils with support for
    Fortran compilers support and more (for Python <= 3.11)

Utilities
---------
test
    Run numpy unittests
show_config
    Show numpy build configuration
__version__
    NumPy version string

Viewing documentation using IPython
-----------------------------------

Start IPython and import `numpy` usually under the alias ``np``: `import
numpy as np`.  Then, directly past or use the ``%cpaste`` magic to paste
examples into the shell.  To see which functions are available in `numpy`,
type ``np.<TAB>`` (where ``<TAB>`` refers to the TAB key), or use
``np.*cos*?<ENTER>`` (where ``<ENTER>`` refers to the ENTER key) to narrow
down the list.  To view the docstring for a function, use
``np.cos?<ENTER>`` (to view the docstring) and ``np.cos??<ENTER>`` (to view
the source code).

Copies vs. in-place operation
-----------------------------
Most of the functions in `numpy` return a copy of the array argument
(e.g., `np.sort`).  In-place versions of these functions are often
available as array methods, i.e. ``x = np.array([1,2,3]); x.sort()``.
Exceptions to this rule are documented.

"""
import os
import sys
import warnings

from ._globals import _NoValue, _CopyMode

# We first need to detect if we're being called as part of the numpy setup
# procedure itself in a reliable manner.
try:
    __NUMPY_SETUP__
except NameError:
    __NUMPY_SETUP__ = False

if __NUMPY_SETUP__:
    sys.stderr.write('Running from numpy source directory.\n')
else:
    # Allow distributors to run custom init code before importing numpy.core
    from . import _distributor_init

    try:
        from numpy.__config__ import show as show_config
    except ImportError as e:
        msg = """Error importing numpy: you should not try to import numpy from
        its source directory; please exit the numpy source tree, and relaunch
        your python interpreter from there."""
        raise ImportError(msg) from e

    from . import core
    from .core import (
        _no_nep50_warning, memmap, Inf, Infinity, NaN, iinfo, finfo,
        False_, ScalarType, True_, abs, absolute, add, all, allclose, alltrue,
        amax, amin, any, arange, arccos, arccosh, arcsin, arcsinh, arctan,
        arctan2, arctanh, argmax, argmin, argpartition, argsort, argwhere,
        around, array, array2string, array_equal, array_equiv, array_repr,
        array_str, asanyarray, asarray, ascontiguousarray, asfortranarray,
        atleast_1d, atleast_2d, atleast_3d, base_repr, binary_repr, 
        bitwise_and, bitwise_not, bitwise_or, bitwise_xor, block, bool_,
        broadcast, busday_count, busday_offset, busdaycalendar, byte, bytes_,
        can_cast, cbrt, cdouble, ceil, cfloat, char, character, chararray,
        choose, clip, clongdouble, clongfloat, compare_chararrays, complex_,
        complexfloating, compress, concatenate, conj, conjugate, convolve,
        copysign, copyto, correlate, cos, cosh, count_nonzero, cross, csingle,
        cumprod, cumproduct, cumsum, datetime64, datetime_as_string, 
        datetime_data, deg2rad, degrees, diagonal, divide, divmod, dot, 
        double, dtype, e, einsum, einsum_path, empty, empty_like, equal,
        errstate, euler_gamma, exp, exp2, expm1, fabs, find_common_type, 
        flatiter, flatnonzero, flexible, 
        float_, float_power, floating, floor, floor_divide, fmax, fmin, fmod, 
        format_float_positional, format_float_scientific, format_parser, 
        frexp, from_dlpack, frombuffer, fromfile, fromfunction, fromiter, 
        frompyfunc, fromstring, full, full_like, gcd, generic, geomspace, 
        get_printoptions, getbufsize, geterr, geterrcall, greater, 
        greater_equal, half, heaviside, hstack, hypot, identity, iinfo, 
        indices, inexact, inf, infty, inner, int_,
        intc, integer, invert, is_busday, isclose, isfinite, isfortran,
        isinf, isnan, isnat, isscalar, issctype, issubdtype, lcm, ldexp,
        left_shift, less, less_equal, lexsort, linspace, little_endian, log, 
        log10, log1p, log2, logaddexp, logaddexp2, logical_and, logical_not, 
        logical_or, logical_xor, logspace, longcomplex, longdouble, 
        longfloat, longlong, matmul, max, maximum, maximum_sctype, 
        may_share_memory, mean, min, min_scalar_type, minimum, mod, 
        modf, moveaxis, multiply, nan, nbytes, ndarray, ndim, nditer, 
        negative, nested_iters, newaxis, nextafter, nonzero, not_equal,
        number, obj2sctype, object_, ones, ones_like, outer, partition,
        pi, positive, power, printoptions, prod, product, promote_types, 
        ptp, put, putmask, rad2deg, radians, ravel, rec, recarray, reciprocal,
        record, remainder, repeat, require, reshape, resize, result_type, 
        right_shift, rint, roll, rollaxis, round, round_, sctype2char, 
        sctypeDict, sctypes, searchsorted, set_printoptions,
        set_string_function, setbufsize, seterr, seterrcall, shape,
        shares_memory, short, sign, signbit, signedinteger, sin, single, 
        singlecomplex, sinh, size, sometrue, sort, spacing, sqrt, square, 
        squeeze, stack, std, str_, string_, subtract, sum, swapaxes, take,
        tan, tanh, tensordot, timedelta64, trace, transpose, 
        true_divide, trunc, typecodes, ubyte, ufunc, uint, uintc, ulonglong, 
        unicode_, unsignedinteger, ushort, var, vdot, void, vstack, where, 
        zeros, zeros_like, _get_promotion_state, _set_promotion_state
    )

    sized_aliases = ["int8", "int16", "int32", "int64", "intp", 
                     "uint8", "uint16", "uint32", "uint64", "uintp",
                     "float16", "float32", "float64", "float96", "float128",
                     "complex64", "complex128", "complex192", "complex256"]
    
    for sa in sized_aliases:
        try:
            globals()[sa] = getattr(core, sa)
        except AttributeError:
            pass
    del sa, sized_aliases

    from . import lib
    # NOTE: to be revisited following future namespace cleanup.
    # See gh-14454 and gh-15672 for discussion.
    from .lib import (
        DataSource, angle, append, apply_along_axis, apply_over_axes,
        array_split, asarray_chkfinite, asfarray, average, bartlett,
        bincount, blackman, broadcast_arrays, broadcast_shapes,
        broadcast_to, byte_bounds, c_, column_stack, common_type,
        copy, corrcoef, cov, delete, diag, diag_indices,
        diag_indices_from, diagflat, diff, digitize, dsplit, dstack,
        ediff1d, emath, expand_dims, extract, eye, fill_diagonal, fix,
        flip, fliplr, flipud, fromregex, get_array_wrap, genfromtxt,
        get_include, gradient, hamming, hanning, histogram, histogram2d,
        histogram_bin_edges, histogramdd, hsplit, i0, imag, in1d,
        index_exp, info, insert, interp, intersect1d, iscomplex,
        iscomplexobj, isin, isneginf, isreal, isrealobj, issubclass_,
        issubsctype, iterable, ix_, kaiser, kron, load, loadtxt, mask_indices,
        median, meshgrid, mgrid, mintypecode, msort, nan_to_num, 
        nanargmax, nanargmin, nancumprod, nancumsum, nanmax, nanmean,
        nanmedian, nanmin, nanpercentile, nanprod, nanquantile, nanstd,
        nansum, nanvar, ndenumerate, ndindex, ogrid, packbits, pad,
        percentile, piecewise, place, poly, poly1d, polyadd, polyder,
        polydiv, polyfit, polyint, polymul, polysub, polyval,
        put_along_axis, quantile, r_, ravel_multi_index, real, real_if_close,
        roots, rot90, row_stack, s_, save, savetxt, savez, savez_compressed,
        select, setdiff1d, setxor1d, show_runtime, sinc, sort_complex, split,
        take_along_axis, tile, tracemalloc_domain, trapz, tri, tril,
        tril_indices, tril_indices_from, typename, union1d, unique, unpackbits,
        unravel_index, unwrap, vander, vectorize, vsplit, trim_zeros,
        triu, triu_indices, triu_indices_from, isposinf, RankWarning, disp,
        deprecate, deprecate_with_doc, who, safe_eval, recfromtxt, recfromcsv
    )
    from . import matrixlib as _mat
    from .matrixlib import (
        asmatrix, bmat, defmatrix, mat, matrix, test
    )

    # We build warning messages for former attributes
    _msg = (
        "module 'numpy' has no attribute '{n}'.\n"
        "`np.{n}` was a deprecated alias for the builtin `{n}`. "
        "To avoid this error in existing code, use `{n}` by itself. "
        "Doing this will not modify any behavior and is safe. {extended_msg}\n"
        "The aliases was originally deprecated in NumPy 1.20; for more "
        "details and guidance see the original release note at:\n"
        "    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations")

    _specific_msg = (
        "If you specifically wanted the numpy scalar type, use `np.{}` here.")

    _int_extended_msg = (
        "When replacing `np.{}`, you may wish to use e.g. `np.int64` "
        "or `np.int32` to specify the precision. If you wish to review "
        "your current use, check the release note link for "
        "additional information.")

    _type_info = [
        ("object", ""),  # The NumPy scalar only exists by name.
        ("bool", _specific_msg.format("bool_")),
        ("float", _specific_msg.format("float64")),
        ("complex", _specific_msg.format("complex128")),
        ("str", _specific_msg.format("str_")),
        ("int", _int_extended_msg.format("int"))]

    __former_attrs__ = {
         n: _msg.format(n=n, extended_msg=extended_msg)
         for n, extended_msg in _type_info
     }


    # Some of these could be defined right away, but most were aliases to
    # the Python objects and only removed in NumPy 1.24.  Defining them should
    # probably wait for NumPy 1.26 or 2.0.
    # When defined, these should possibly not be added to `__all__` to avoid
    # import with `from numpy import *`.
    __future_scalars__ = {"bool", "long", "ulong", "str", "bytes", "object"}

    # now that numpy core module is imported, can initialize limits
    core.getlimits._register_known_types()

    __all__ = ['exceptions']
    __all__.extend(['__version__', 'show_config'])
    __all__.extend(core.__all__)
    __all__.extend(_mat.__all__)
    __all__.extend(lib.__all__)
    __all__.extend(['linalg', 'fft', 'random', 'ctypeslib', 'ma'])

    # Filter out Cython harmless warnings
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    def __getattr__(attr):
        # Warn for expired attributes
        import warnings

        if attr == "linalg":
            import numpy.linalg as linalg
            return linalg
        elif attr == "fft":
            import numpy.fft as fft
            return fft
        elif attr == "dtypes":
            import numpy.dtypes as dtypes
            return dtypes
        elif attr == "random":
            import numpy.random as random
            return random
        elif attr == "polynomial":
            import numpy.polynomial as polynomial
            return polynomial
        elif attr == "ma":
            import numpy.ma as ma
            return ma
        elif attr == "ctypeslib":
            import numpy.ctypeslib as ctypeslib
            return ctypeslib
        elif attr == "exceptions":
            import numpy.exceptions as exceptions
            return exceptions
        elif attr == 'testing':
            import numpy.testing as testing
            return testing
        elif attr == "matlib":
            import numpy.matlib as matlib
            return matlib

        if attr in __future_scalars__:
            # And future warnings for those that will change, but also give
            # the AttributeError
            warnings.warn(
                f"In the future `np.{attr}` will be defined as the "
                "corresponding NumPy scalar.", FutureWarning, stacklevel=2)

        if attr in __former_attrs__:
            raise AttributeError(__former_attrs__[attr])

        raise AttributeError("module {!r} has no attribute "
                             "{!r}".format(__name__, attr))

    def __dir__():
        public_symbols = globals().keys() | {'testing'}
        public_symbols -= {"core", "matrixlib"}
        return list(public_symbols)

    # Pytest testing
    from numpy._pytesttester import PytestTester
    test = PytestTester(__name__)
    del PytestTester

    def _sanity_check():
        """
        Quick sanity checks for common bugs caused by environment.
        There are some cases e.g. with wrong BLAS ABI that cause wrong
        results under specific runtime conditions that are not necessarily
        achieved during test suite runs, and it is useful to catch those early.

        See https://github.com/numpy/numpy/issues/8577 and other
        similar bug reports.

        """
        try:
            x = ones(2, dtype=float32)
            if not abs(x.dot(x) - float32(2.0)) < 1e-5:
                raise AssertionError()
        except AssertionError:
            msg = ("The current Numpy installation ({!r}) fails to "
                   "pass simple sanity checks. This can be caused for example "
                   "by incorrect BLAS library being linked in, or by mixing "
                   "package managers (pip, conda, apt, ...). Search closed "
                   "numpy issues for similar problems.")
            raise RuntimeError(msg.format(__file__)) from None

    _sanity_check()
    del _sanity_check

    def _mac_os_check():
        """
        Quick Sanity check for Mac OS look for accelerate build bugs.
        Testing numpy polyfit calls init_dgelsd(LAPACK)
        """
        try:
            c = array([3., 2., 1.])
            x = linspace(0, 2, 5)
            y = polyval(c, x)
            _ = polyfit(x, y, 2, cov=True)
        except ValueError:
            pass

    if sys.platform == "darwin":
        with warnings.catch_warnings(record=True) as w:
            _mac_os_check()
            # Throw runtime error, if the test failed Check for warning and error_message
            if len(w) > 0:
                error_message = "{}: {}".format(w[-1].category.__name__, str(w[-1].message))
                msg = (
                    "Polyfit sanity test emitted a warning, most likely due "
                    "to using a buggy Accelerate backend."
                    "\nIf you compiled yourself, more information is available at:"
                    "\nhttps://numpy.org/doc/stable/user/building.html#accelerated-blas-lapack-libraries"
                    "\nOtherwise report this to the vendor "
                    "that provided NumPy.\n{}\n".format(error_message))
                raise RuntimeError(msg)
        del w
    del _mac_os_check

    def hugepage_setup():
        """
        We usually use madvise hugepages support, but on some old kernels it
        is slow and thus better avoided. Specifically kernel version 4.6 
        had a bug fix which probably fixed this:
        https://github.com/torvalds/linux/commit/7cf91a98e607c2f935dbcc177d70011e95b8faff
        """
        use_hugepage = os.environ.get("NUMPY_MADVISE_HUGEPAGE", None)
        if sys.platform == "linux" and use_hugepage is None:
            # If there is an issue with parsing the kernel version,
            # set use_hugepage to 0. Usage of LooseVersion will handle
            # the kernel version parsing better, but avoided since it
            # will increase the import time. 
            # See: #16679 for related discussion.
            try:
                use_hugepage = 1
                kernel_version = os.uname().release.split(".")[:2]
                kernel_version = tuple(int(v) for v in kernel_version)
                if kernel_version < (4, 6):
                    use_hugepage = 0
            except ValueError:
                use_hugepage = 0
            finally:
                del kernel_version
        elif use_hugepage is None:
            # This is not Linux, so it should not matter, just enable anyway
            use_hugepage = 1
        else:
            use_hugepage = int(use_hugepage)

    # Note that this will currently only make a difference on Linux
    core.multiarray._set_madvise_hugepage(hugepage_setup())
    del hugepage_setup

    # Give a warning if NumPy is reloaded or imported on a sub-interpreter
    # We do this from python, since the C-module may not be reloaded and
    # it is tidier organized.
    core.multiarray._multiarray_umath._reload_guard()

    # TODO: Switch to defaulting to "weak".
    core._set_promotion_state(
        os.environ.get("NPY_PROMOTION_STATE", "legacy"))

    # Tell PyInstaller where to find hook-numpy.py
    def _pyinstaller_hooks_dir():
        from pathlib import Path
        return [str(Path(__file__).with_name("_pyinstaller").resolve())]


# get the version using versioneer
from .version import __version__, git_revision as __git_version__

# Remove symbols imported for internal use
del os, sys, warnings
