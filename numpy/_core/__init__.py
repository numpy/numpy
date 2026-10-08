"""
Contains the core of NumPy: ndarray, ufuncs, dtypes, etc.

Please note that this module is private.  All functions and objects
are available in the main ``numpy`` namespace - use that instead.

"""

import os

from numpy.version import version as __version__

# Lazy on Python 3.15+ (PEP 810); ignored by older Pythons.  Names defined in C
# come from the always-loaded multiarray, umath and numerictypes modules.
__lazy_modules__ = [
    "numpy._core",
    "numpy._core.numeric",
    "numpy._core.fromnumeric",
    "numpy._core.arrayprint",
    "numpy._core._ufunc_config",
    "numpy._core._asarray",
    "numpy._core.shape_base",
    "numpy._core.function_base",
    "numpy._core.getlimits",
    "numpy._core.einsumfunc",
    "numpy._core.records",
]

# disables OpenBLAS affinity setting of the main thread that limits
# python threads or processes to one core
env_added = []
for envkey in ['OPENBLAS_MAIN_FREE']:
    if envkey not in os.environ:
        # Note: using `putenv` (and `unsetenv` further down) instead of updating
        # `os.environ` on purpose to avoid a race condition, see gh-30627.
        os.putenv(envkey, '1')
        env_added.append(envkey)

try:
    from . import multiarray
except ImportError as exc:
    import sys

    # Bypass for the module re-initialization opt-out
    if exc.msg == "cannot load module more than once per process":
        raise

    # Basically always, the problem should be that the C module is wrong/missing...
    if (
        isinstance(exc, ModuleNotFoundError)
        and exc.name == "numpy._core._multiarray_umath"
    ):
        import sys
        candidates = []
        for path in __path__:
            candidates.extend(
                f for f in os.listdir(path) if f.startswith("_multiarray_umath"))
        if len(candidates) == 0:
            bad_c_module_info = (
                "We found no compiled module, did NumPy build successfully?\n")
        else:
            candidate_str = '\n  * '.join(candidates)
            # cache_tag is documented to be possibly None, so just use name if it is
            # this guesses at cache_tag being the same as the extension module scheme
            tag = sys.implementation.cache_tag or sys.implementation.name
            bad_c_module_info = (
                f"The following compiled module files exist, but seem incompatible\n"
                f"with either python '{tag}' or the "
                f"platform '{sys.platform}':\n\n  * {candidate_str}\n"
            )
    else:
        bad_c_module_info = ""

    major, minor, *_ = sys.version_info
    msg = f"""

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
{bad_c_module_info}
We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python {major}.{minor} from "{sys.executable}"
  * The NumPy version is: "{__version__}"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: {exc}
"""

    raise ImportError(msg) from exc
finally:
    for envkey in env_added:
        os.unsetenv(envkey)
del envkey
del env_added
del os

from . import umath

# Check that multiarray,umath are pure python modules wrapping
# _multiarray_umath and not either of the old c-extension modules
if not (hasattr(multiarray, '_multiarray_umath') and
        hasattr(umath, '_multiarray_umath')):
    import sys
    path = sys.modules['numpy'].__path__
    msg = ("Something is wrong with the numpy installation. "
        "While importing we detected an older version of "
        "numpy in {}. One method of fixing this is to repeatedly uninstall "
        "numpy until none is found, then reinstall this version.")
    raise ImportError(msg.format(path))

from . import numerictypes as nt
from .numerictypes import sctypeDict, sctypes

multiarray.set_typeDict(nt.sctypeDict)
del nt

# always loaded (C-defined objects)
# eager: `memmap` is also the module name, an import of it would replace a proxy
from .memmap import memmap
from .multiarray import (
    arange,
    array,
    asanyarray,
    asarray,
    ascontiguousarray,
    asfortranarray,
    broadcast,
    busday_count,
    busday_offset,
    busdaycalendar,
    can_cast,
    character,
    complexfloating,
    concatenate,
    copyto,
    datetime_as_string,
    datetime_data,
    dot,
    dtype,
    empty,
    empty_like,
    flatiter,
    flexible,
    floating,
    from_dlpack,
    frombuffer,
    fromfile,
    fromiter,
    fromstring,
    generic,
    inexact,
    inner,
    integer,
    is_busday,
    lexsort,
    may_share_memory,
    min_scalar_type,
    ndarray,
    nditer,
    nested_iters,
    number,
    promote_types,
    putmask,
    result_type,
    shares_memory,
    signedinteger,
    unsignedinteger,
    vdot,
    where,
    zeros,
)
from .numerictypes import (
    ScalarType,
    bool,
    bool_,
    byte,
    bytes_,
    cdouble,
    clongdouble,
    complex64,
    complex128,
    complex256,
    csingle,
    datetime64,
    double,
    float16,
    float32,
    float64,
    float128,
    half,
    int8,
    int16,
    int32,
    int64,
    int_,
    intc,
    intp,
    isdtype,
    issubdtype,
    long,
    longdouble,
    longlong,
    object_,
    short,
    single,
    str_,
    timedelta64,
    typecodes,
    ubyte,
    uint,
    uint8,
    uint16,
    uint32,
    uint64,
    uintc,
    uintp,
    ulong,
    ulonglong,
    ushort,
    void,
)
from .umath import (
    absolute,
    absolute as abs,
    add,
    arccos,
    arccos as acos,
    arccosh,
    arccosh as acosh,
    arcsin,
    arcsin as asin,
    arcsinh,
    arcsinh as asinh,
    arctan,
    arctan as atan,
    arctan2,
    arctan2 as atan2,
    arctanh,
    arctanh as atanh,
    bitwise_and,
    bitwise_count,
    bitwise_or,
    bitwise_xor,
    cbrt,
    ceil,
    conj,
    conjugate,
    copysign,
    cos,
    cosh,
    deg2rad,
    degrees,
    divide,
    divmod,
    e,
    equal,
    euler_gamma,
    exp,
    exp2,
    expm1,
    fabs,
    float_power,
    floor,
    floor_divide,
    fmax,
    fmin,
    fmod,
    frexp,
    frompyfunc,
    gcd,
    greater,
    greater_equal,
    heaviside,
    hypot,
    invert,
    invert as bitwise_invert,
    isfinite,
    isinf,
    isnan,
    isnat,
    lcm,
    ldexp,
    left_shift,
    left_shift as bitwise_left_shift,
    less,
    less_equal,
    log,
    log1p,
    log2,
    log10,
    logaddexp,
    logaddexp2,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    matmul,
    matvec,
    maximum,
    minimum,
    mod,
    modf,
    multiply,
    negative,
    nextafter,
    not_equal,
    pi,
    positive,
    power,
    power as pow,
    rad2deg,
    radians,
    reciprocal,
    remainder,
    right_shift,
    right_shift as bitwise_right_shift,
    rint,
    sign,
    signbit,
    sin,
    sinh,
    spacing,
    sqrt,
    square,
    subtract,
    tan,
    tanh,
    true_divide,
    trunc,
    vecdot,
    vecmat,
)

concat = multiarray.concatenate
ufunc = type(umath.sin)

# lazy on 3.15+; `numeric` first where eager (the others import it, and it
# imports arrayprint at its end)
from . import numeric  # noqa: I001
from . import (
    _asarray,
    _ufunc_config,
    arrayprint,
    einsumfunc,
    fromnumeric,
    function_base,
    getlimits,
    records,
    shape_base,
)
from .numeric import (
    False_,
    True_,
    allclose,
    argwhere,
    array_equal,
    array_equiv,
    astype,
    base_repr,
    binary_repr,
    bitwise_not,
    convolve,
    correlate,
    count_nonzero,
    cross,
    flatnonzero,
    fromfunction,
    full,
    full_like,
    identity,
    indices,
    inf,
    isclose,
    isfortran,
    isscalar,
    little_endian,
    moveaxis,
    newaxis,
    ones,
    ones_like,
    outer,
    roll,
    rollaxis,
    tensordot,
    zeros_like,
)
from .fromnumeric import (
    all,
    amax,
    amin,
    any,
    argmax,
    argmin,
    argpartition,
    argsort,
    around,
    choose,
    clip,
    compress,
    cumprod,
    cumsum,
    cumulative_prod,
    cumulative_sum,
    diagonal,
    matrix_transpose,
    max,
    mean,
    min,
    minmax,
    ndim,
    nonzero,
    partition,
    prod,
    ptp,
    put,
    ravel,
    repeat,
    reshape,
    resize,
    round,
    searchsorted,
    shape,
    size,
    sort,
    squeeze,
    std,
    sum,
    swapaxes,
    take,
    top_k,
    trace,
    transpose,
    var,
)
from .arrayprint import (
    array2string,
    array_repr,
    array_str,
    errstate,
    format_float_positional,
    format_float_scientific,
    get_printoptions,
    printoptions,
    set_printoptions,
)
from ._ufunc_config import (
    getbufsize,
    geterr,
    geterrcall,
    setbufsize,
    seterr,
    seterrcall,
)
from ._asarray import (
    require,
)
from .shape_base import (
    atleast_1d,
    atleast_2d,
    atleast_3d,
    block,
    hstack,
    stack,
    unstack,
    vstack,
)
from .function_base import (
    geomspace,
    linspace,
    logspace,
    nan,
)
from .getlimits import (
    finfo,
    iinfo,
)
from .einsumfunc import (
    einsum,
    einsum_path,
)
from .records import (
    recarray,
    record,
)
from .fromnumeric import transpose as permute_dims

# The public names bound above except the submodules (only names are inspected,
# nothing is imported); checked against the submodules in test_public_api.py.
_not_exported = {
    "multiarray", "umath", "numerictypes", "overrides", "numeric", "fromnumeric",
    "arrayprint", "_ufunc_config", "_asarray", "shape_base", "function_base",
    "getlimits", "einsumfunc", "records", "sctypes",
}
__all__ = [
    name for name in globals()
    if not name.startswith("_") and name not in _not_exported
]
del _not_exported

# side-effect imports (docstrings)
import numpy._core._add_newdocs as _add_newdocs
import numpy._core._add_newdocs_scalars as _add_newdocs_scalars

# add these for module-freeze analysis (like PyInstaller)
from . import _dtype, _dtype_ctypes, _internal, _methods


def _ufunc_reduce(func):
    return func.__name__


def _DType_reconstruct(scalar_type):
    # This is a work-around to allow DType classes to pickle
    return type(multiarray.dtype(scalar_type))


def _DType_reduce(DType):
    # As types/classes, most DTypes can simply be pickled by their name:
    if not DType._legacy or DType.__module__ == "numpy.dtypes":
        return DType.__name__

    # However, user defined legacy dtypes (like rational) do not end up in
    # `numpy.dtypes` as module and do not have a public class at all.
    # For these, we pickle them by reconstructing them from the scalar type:
    scalar_type = DType.type
    return _DType_reconstruct, (scalar_type,)


import copyreg

copyreg.pickle(type(umath.sin), _ufunc_reduce)
copyreg.pickle(type(multiarray.dtype), _DType_reduce, _DType_reconstruct)
# Unclutter namespace (must keep _*_reconstruct for unpickling)
del copyreg, _ufunc_reduce, _DType_reduce

from numpy._pytesttester import PytestTester

test = PytestTester(__name__)
del PytestTester
