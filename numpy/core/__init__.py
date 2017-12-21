from __future__ import division, absolute_import, print_function

from .info import __doc__
from numpy.version import version as __version__

# disables OpenBLAS affinity setting of the main thread that limits
# python threads or processes to one core
import os
env_added = []
for envkey in ['OPENBLAS_MAIN_FREE', 'GOTOBLAS_MAIN_FREE']:
    if envkey not in os.environ:
        os.environ[envkey] = '1'
        env_added.append(envkey)

try:
    from . import multiarray
except ImportError as exc:
    msg = """
Importing the multiarray numpy extension module failed.  Most
likely you are trying to import a failed build of numpy.
If you're working with a numpy git repo, try `git clean -xdf` (removes all
files not under version control).  Otherwise reinstall numpy.

Original error was: %s
""" % (exc,)
    raise ImportError(msg)
finally:
    for envkey in env_added:
        del os.environ[envkey]
del envkey
del env_added
del os

from . import umath
from . import _internal  # for freeze programs
from . import numerictypes as nt
multiarray.set_typeDict(nt.sctypeDict)
from . import numeric
from .numeric import (
    ALLOW_THREADS, AxisError, BUFSIZE, CLIP, ComplexWarning, ERR_CALL,
    ERR_DEFAULT, ERR_IGNORE, ERR_LOG, ERR_PRINT, ERR_RAISE, ERR_WARN,
    FLOATING_POINT_SUPPORT, FPE_DIVIDEBYZERO, FPE_INVALID, FPE_OVERFLOW,
    FPE_UNDERFLOW, False_, Inf, Infinity, MAXDIMS, MAY_SHARE_BOUNDS,
    MAY_SHARE_EXACT, NAN, NINF, NZERO, NaN, PINF, PZERO, RAISE,
    SHIFT_DIVIDEBYZERO, SHIFT_INVALID, SHIFT_OVERFLOW, SHIFT_UNDERFLOW,
    ScalarType, TooHardError, True_, UFUNC_BUFSIZE_DEFAULT, UFUNC_PYVALS_NAME,
    WRAP, absolute, absolute_import, add, alen, all, allclose, alltrue, amax,
    amin, any, arange, arccos, arccosh, arcsin, arcsinh, arctan, arctan2,
    arctanh, argmax, argmin, argpartition, argsort, argwhere, around, array,
    array2string, array_equal, array_equiv, array_repr, array_str, arrayprint,
    asanyarray, asarray, ascontiguousarray, asfortranarray, base_repr,
    basestring, binary_repr, bitwise_and, bitwise_not, bitwise_or, bitwise_xor,
    bool8, bool_, broadcast, builtins, busday_count, busday_offset,
    busdaycalendar, byte, bytes0, bytes_, can_cast, cast, cbrt, cdouble, ceil,
    cfloat, character, choose, clip, clongdouble, clongfloat, collections,
    compare_chararrays, complex128, complex64, complex_, complexfloating,
    compress, concatenate, conj, conjugate, convolve, copysign, copyto,
    correlate, cos, cosh, count_nonzero, cross, csingle, cumprod, cumproduct,
    cumsum, datetime64, datetime_as_string, datetime_data, deg2rad, degrees,
    diagonal, divide, division, divmod, dot, double, dtype, e, empty,
    empty_like, equal, errstate, euler_gamma, exp, exp2, expm1, extend_all,
    fabs, fastCopyAndTranspose, find_common_type, flatiter, flatnonzero,
    flexible, float16, float32, float64, float_, float_power, floating, floor,
    floor_divide, fmax, fmin, fmod, format_float_positional,
    format_float_scientific, frexp, frombuffer, fromfile, fromfunction,
    fromiter, fromnumeric, frompyfunc, fromstring, full, full_like, gcd,
    generic, get_printoptions, getbufsize, geterr, geterrcall, geterrobj,
    greater, greater_equal, half, heaviside, hypot, identity, indices, inexact,
    inf, infty, inner, int0, int16, int32, int64, int8, int_, int_asbuffer,
    intc, integer, intp, invert, is_busday, isclose, isfinite, isfortran,
    isinf, isnan, isnat, isscalar, issctype, issubdtype, itertools, lcm, ldexp,
    left_shift, less, less_equal, lexsort, little_endian, load, loads, log,
    log10, log1p, log2, logaddexp, logaddexp2, logical_and, logical_not,
    logical_or, logical_xor, long, longcomplex, longdouble, longfloat,
    longlong, matmul, maximum, maximum_sctype, may_share_memory, mean,
    min_scalar_type, minimum, mod, modf, moveaxis, multiarray, multiply, nan,
    nbytes, ndarray, ndim, nditer, negative, nested_iters, newaxis, nextafter,
    nonzero, normalize_axis_index, normalize_axis_tuple, not_equal, np, number,
    numbers, numerictypes, obj2sctype, object0, object_, ones, ones_like,
    operator, outer, partition, pi, pickle, positive, power, print_function,
    prod, product, promote_types, ptp, put, putmask, rad2deg, radians, rank,
    ravel, reciprocal, remainder, repeat, require, reshape, resize,
    result_type, right_shift, rint, roll, rollaxis, round_, sctype2char,
    sctypeDict, sctypeNA, sctypes, searchsorted, set_numeric_ops,
    set_printoptions, set_string_function, setbufsize, seterr, seterrcall,
    seterrobj, shape, shares_memory, short, sign, signbit, signedinteger, sin,
    single, singlecomplex, sinh, size, sometrue, sort, spacing, sqrt, square,
    squeeze, std, str0, str_, string_, subtract, sum, swapaxes, sys, take, tan,
    tanh, tensordot, timedelta64, trace, transpose, true_divide, trunc,
    typeDict, typeNA, typecodes, ubyte, ufunc, uint, uint0, uint16, uint32,
    uint64, uint8, uintc, uintp, ulonglong, umath, unicode, unicode_,
    unsignedinteger, ushort, var, vdot, void, void0, warnings, where, zeros,
    zeros_like)
from . import fromnumeric
from .fromnumeric import (
    VisibleDeprecationWarning, absolute_import, alen, all, alltrue, amax, amin,
    any, argmax, argmin, argpartition, argsort, around, array, asanyarray,
    asarray, choose, clip, compress, concatenate, cumprod, cumproduct, cumsum,
    diagonal, division, mean, mu, ndim, nonzero, np, nt, partition,
    print_function, prod, product, ptp, put, rank, ravel, repeat, reshape,
    resize, round_, searchsorted, shape, size, sometrue, sort, squeeze, std,
    sum, swapaxes, take, trace, transpose, types, um, var, warnings)
from . import defchararray as char
from . import records as rec
from .records import (absolute_import, array, bytes, division, find_duplicate,
                      format_parser, fromarrays, fromfile, fromrecords,
                      fromstring, get_printoptions, get_remaining_size,
                      isfileobj, long, ndarray, nt, numfmt, os, print_function,
                      recarray, record, sb, sys)
from .memmap import (absolute_import, basestring, division, dtype, dtypedescr,
                     is_pathlib_path, long, memmap, mode_equivalents, ndarray,
                     np, print_function, uint8, valid_filemodes,
                     writeable_filemodes)
from .defchararray import chararray
from . import function_base
from .function_base import (MAY_SHARE_BOUNDS, NaN, TooHardError,
                            absolute_import, asanyarray, division, geomspace,
                            linspace, logspace, operator, print_function,
                            result_type, shares_memory, warnings)
from . import machar
from .machar import (MachAr, absolute_import, any, division, errstate,
                     print_function)
from . import getlimits
from .getlimits import (MachAr, MachArLike, absolute_import, array, division,
                        exp2, finfo, iinfo, inf, log10, ntypes, numeric,
                        print_function, umath, warnings)
from . import shape_base
from .shape_base import (absolute_import, array, asanyarray, atleast_1d,
                         atleast_2d, atleast_3d, block, division, hstack,
                         newaxis, normalize_axis_index, print_function, stack,
                         vstack)
from . import einsumfunc
from .einsumfunc import (absolute_import, asanyarray, asarray, c_einsum,
                         division, dot, einsum, einsum_path, einsum_symbols,
                         einsum_symbols_set, print_function, result_type,
                         tensordot)
del nt

from .fromnumeric import amax as max, amin as min, round_ as round
from .numeric import absolute as abs

__all__ = ['char', 'rec', 'memmap']
__all__ += numeric.__all__
__all__ += fromnumeric.__all__
__all__ += rec.__all__
__all__ += ['chararray']
__all__ += function_base.__all__
__all__ += machar.__all__
__all__ += getlimits.__all__
__all__ += shape_base.__all__
__all__ += einsumfunc.__all__


from numpy.testing import _numpy_tester
test = _numpy_tester().test
bench = _numpy_tester().bench

# Make it possible so that ufuncs can be pickled
#  Here are the loading and unloading functions
# The name numpy.core._ufunc_reconstruct must be
#   available for unpickling to work.
def _ufunc_reconstruct(module, name):
    # The `fromlist` kwarg is required to ensure that `mod` points to the
    # inner-most module rather than the parent package when module name is
    # nested. This makes it possible to pickle non-toplevel ufuncs such as
    # scipy.special.expit for instance.
    mod = __import__(module, fromlist=[name])
    return getattr(mod, name)

def _ufunc_reduce(func):
    from pickle import whichmodule
    name = func.__name__
    return _ufunc_reconstruct, (whichmodule(func, name), name)


import sys
if sys.version_info[0] >= 3:
    import copyreg
else:
    import copy_reg as copyreg

copyreg.pickle(ufunc, _ufunc_reduce, _ufunc_reconstruct)
# Unclutter namespace (must keep _ufunc_reconstruct for unpickling)
del copyreg
del sys
del _ufunc_reduce
