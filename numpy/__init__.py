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
`the NumPy homepage <http://www.scipy.org>`_.

We recommend exploring the docstrings using
`IPython <http://ipython.scipy.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.  See below for further
instructions.

The docstring examples assume that `numpy` has been imported as `np`::

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

General-purpose documents like a glossary and help on the basic concepts
of numpy are available under the ``doc`` sub-module::

  >>> from numpy import doc
  >>> help(doc)
  ... # doctest: +SKIP

Available subpackages
---------------------
doc
    Topical documentation on broadcasting, indexing, etc.
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
f2py
    Fortran to Python Interface Generator.
distutils
    Enhancements to distutils with support for
    Fortran compilers support and more.

Utilities
---------
test
    Run numpy unittests
show_config
    Show numpy build configuration
dual
    Overwrite certain functions with high-performance Scipy tools
matlib
    Make everything matrices.
__version__
    NumPy version string

Viewing documentation using IPython
-----------------------------------
Start IPython with the NumPy profile (``ipython -p numpy``), which will
import `numpy` under the alias `np`.  Then, use the ``cpaste`` command to
paste examples into the shell.  To see which functions are available in
`numpy`, type ``np.<TAB>`` (where ``<TAB>`` refers to the TAB key), or use
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
from __future__ import division, absolute_import, print_function

import sys
import warnings

from ._globals import ModuleDeprecationWarning, VisibleDeprecationWarning
from ._globals import _NoValue

# We first need to detect if we're being called as part of the numpy setup
# procedure itself in a reliable manner.
try:
    __NUMPY_SETUP__
except NameError:
    __NUMPY_SETUP__ = False

if __NUMPY_SETUP__:
    sys.stderr.write('Running from numpy source directory.\n')
else:
    try:
        from numpy.__config__ import show as show_config
    except ImportError:
        msg = """Error importing numpy: you should not try to import numpy from
        its source directory; please exit the numpy source tree, and relaunch
        your python interpreter from there."""
        raise ImportError(msg)

    from .version import git_revision as __git_revision__
    from .version import version as __version__

    from ._import_tools import PackageLoader

    def pkgload(*packages, **options):
        loader = PackageLoader(infunc=True)
        return loader(*packages, **options)

    from . import add_newdocs
    __all__ = ['add_newdocs',
               'ModuleDeprecationWarning',
               'VisibleDeprecationWarning']

    pkgload.__doc__ = PackageLoader.__call__.__doc__

    # We don't actually use this ourselves anymore, but I'm not 100% sure that
    # no-one else in the world is using it (though I hope not)
    from .testing import Tester, _numpy_tester
    test = _numpy_tester().test
    bench = _numpy_tester().bench

    # Allow distributors to run custom init code
    from . import _distributor_init

    from . import core
    from .core import (ALLOW_THREADS,
                   AxisError,
                   BUFSIZE,
                   CLIP,
                   ComplexWarning,
                   ERR_CALL,
                   ERR_DEFAULT,
                   ERR_IGNORE,
                   ERR_LOG,
                   ERR_PRINT,
                   ERR_RAISE,
                   ERR_WARN,
                   FLOATING_POINT_SUPPORT,
                   FPE_DIVIDEBYZERO,
                   FPE_INVALID,
                   FPE_OVERFLOW,
                   FPE_UNDERFLOW,
                   False_,
                   Inf,
                   Infinity,
                   MAXDIMS,
                   MAY_SHARE_BOUNDS,
                   MAY_SHARE_EXACT,
                   MachAr,
                   NAN,
                   NINF,
                   NZERO,
                   NaN,
                   PINF,
                   PZERO,
                   RAISE,
                   SHIFT_DIVIDEBYZERO,
                   SHIFT_INVALID,
                   SHIFT_OVERFLOW,
                   SHIFT_UNDERFLOW,
                   ScalarType,
                   TooHardError,
                   True_,
                   UFUNC_BUFSIZE_DEFAULT,
                   UFUNC_PYVALS_NAME,
                   WRAP,
                   abs,
                   absolute,
                   absolute_import,
                   add,
                   alen,
                   all,
                   allclose,
                   alltrue,
                   amax,
                   amin,
                   any,
                   arange,
                   arccos,
                   arccosh,
                   arcsin,
                   arcsinh,
                   arctan,
                   arctan2,
                   arctanh,
                   argmax,
                   argmin,
                   argpartition,
                   argsort,
                   argwhere,
                   around,
                   array,
                   array2string,
                   array_equal,
                   array_equiv,
                   array_repr,
                   array_str,
                   arrayprint,
                   asanyarray,
                   asarray,
                   ascontiguousarray,
                   asfortranarray,
                   atleast_1d,
                   atleast_2d,
                   atleast_3d,
                   base_repr,
                   bench,
                   binary_repr,
                   bitwise_and,
                   bitwise_not,
                   bitwise_or,
                   bitwise_xor,
                   block,
                   bool8,
                   bool_,
                   broadcast,
                   busday_count,
                   busday_offset,
                   busdaycalendar,
                   byte,
                   bytes0,
                   bytes_,
                   can_cast,
                   cast,
                   cbrt,
                   cdouble,
                   ceil,
                   cfloat,
                   char,
                   character,
                   chararray,
                   choose,
                   clip,
                   clongdouble,
                   clongfloat,
                   compare_chararrays,
                   complex128,
                   complex64,
                   complex_,
                   complexfloating,
                   compress,
                   concatenate,
                   conj,
                   conjugate,
                   convolve,
                   copysign,
                   copyto,
                   correlate,
                   cos,
                   cosh,
                   count_nonzero,
                   cross,
                   csingle,
                   cumprod,
                   cumproduct,
                   cumsum,
                   datetime64,
                   datetime_as_string,
                   datetime_data,
                   defchararray,
                   deg2rad,
                   degrees,
                   diagonal,
                   divide,
                   division,
                   divmod,
                   dot,
                   double,
                   dtype,
                   e,
                   einsum,
                   einsum_path,
                   einsumfunc,
                   empty,
                   empty_like,
                   equal,
                   errstate,
                   euler_gamma,
                   exp,
                   exp2,
                   expm1,
                   fabs,
                   fastCopyAndTranspose,
                   find_common_type,
                   finfo,
                   flatiter,
                   flatnonzero,
                   flexible,
                   float16,
                   float32,
                   float64,
                   float_,
                   float_power,
                   floating,
                   floor,
                   floor_divide,
                   fmax,
                   fmin,
                   fmod,
                   format_float_positional,
                   format_float_scientific,
                   format_parser,
                   frexp,
                   frombuffer,
                   fromfile,
                   fromfunction,
                   fromiter,
                   fromnumeric,
                   frompyfunc,
                   fromstring,
                   full,
                   full_like,
                   function_base,
                   gcd,
                   generic,
                   geomspace,
                   get_printoptions,
                   getbufsize,
                   geterr,
                   geterrcall,
                   geterrobj,
                   getlimits,
                   greater,
                   greater_equal,
                   half,
                   heaviside,
                   hstack,
                   hypot,
                   identity,
                   iinfo,
                   indices,
                   inexact,
                   inf,
                   info,
                   infty,
                   inner,
                   int0,
                   int16,
                   int32,
                   int64,
                   int8,
                   int_,
                   int_asbuffer,
                   intc,
                   integer,
                   intp,
                   invert,
                   is_busday,
                   isclose,
                   isfinite,
                   isfortran,
                   isinf,
                   isnan,
                   isnat,
                   isscalar,
                   issctype,
                   issubdtype,
                   lcm,
                   ldexp,
                   left_shift,
                   less,
                   less_equal,
                   lexsort,
                   linspace,
                   little_endian,
                   load,
                   loads,
                   log,
                   log10,
                   log1p,
                   log2,
                   logaddexp,
                   logaddexp2,
                   logical_and,
                   logical_not,
                   logical_or,
                   logical_xor,
                   logspace,
                   long,
                   longcomplex,
                   longdouble,
                   longfloat,
                   longlong,
                   machar,
                   matmul,
                   max,
                   maximum,
                   maximum_sctype,
                   may_share_memory,
                   mean,
                   memmap,
                   min,
                   min_scalar_type,
                   minimum,
                   mod,
                   modf,
                   moveaxis,
                   multiarray,
                   multiply,
                   nan,
                   nbytes,
                   ndarray,
                   ndim,
                   nditer,
                   negative,
                   nested_iters,
                   newaxis,
                   nextafter,
                   nonzero,
                   not_equal,
                   number,
                   numeric,
                   numerictypes,
                   obj2sctype,
                   object0,
                   object_,
                   ones,
                   ones_like,
                   outer,
                   partition,
                   pi,
                   positive,
                   power,
                   print_function,
                   prod,
                   product,
                   promote_types,
                   ptp,
                   put,
                   putmask,
                   rad2deg,
                   radians,
                   rank,
                   ravel,
                   rec,
                   recarray,
                   reciprocal,
                   record,
                   records,
                   remainder,
                   repeat,
                   require,
                   reshape,
                   resize,
                   result_type,
                   right_shift,
                   rint,
                   roll,
                   rollaxis,
                   round,
                   round_,
                   sctype2char,
                   sctypeDict,
                   sctypeNA,
                   sctypes,
                   searchsorted,
                   set_numeric_ops,
                   set_printoptions,
                   set_string_function,
                   setbufsize,
                   seterr,
                   seterrcall,
                   seterrobj,
                   shape,
                   shape_base,
                   shares_memory,
                   short,
                   sign,
                   signbit,
                   signedinteger,
                   sin,
                   single,
                   singlecomplex,
                   sinh,
                   size,
                   sometrue,
                   sort,
                   spacing,
                   sqrt,
                   square,
                   squeeze,
                   stack,
                   std,
                   str0,
                   str_,
                   string_,
                   subtract,
                   sum,
                   swapaxes,
                   take,
                   tan,
                   tanh,
                   tensordot,
                   test,
                   timedelta64,
                   trace,
                   transpose,
                   true_divide,
                   trunc,
                   typeDict,
                   typeNA,
                   typecodes,
                   ubyte,
                   ufunc,
                   uint,
                   uint0,
                   uint16,
                   uint32,
                   uint64,
                   uint8,
                   uintc,
                   uintp,
                   ulonglong,
                   umath,
                   unicode,
                   unicode_,
                   unsignedinteger,
                   ushort,
                   var,
                   vdot,
                   void,
                   void0,
                   vstack,
                   where,
                   zeros,
                   zeros_like)
    from . import compat
    from . import lib
    from .lib import (Arrayterator,
                  DataSource,
                  NumpyVersion,
                  RankWarning,
                  absolute_import,
                  add_docstring,
                  add_newdoc,
                  add_newdoc_ufunc,
                  angle,
                  append,
                  apply_along_axis,
                  apply_over_axes,
                  array_split,
                  arraypad,
                  arraysetops,
                  arrayterator,
                  asarray_chkfinite,
                  asfarray,
                  asscalar,
                  average,
                  bartlett,
                  bench,
                  bincount,
                  blackman,
                  broadcast_arrays,
                  broadcast_to,
                  byte_bounds,
                  c_,
                  column_stack,
                  common_type,
                  copy,
                  corrcoef,
                  cov,
                  delete,
                  deprecate,
                  deprecate_with_doc,
                  diag,
                  diag_indices,
                  diag_indices_from,
                  diagflat,
                  diff,
                  digitize,
                  disp,
                  division,
                  dsplit,
                  dstack,
                  ediff1d,
                  emath,
                  expand_dims,
                  extract,
                  eye,
                  fill_diagonal,
                  financial,
                  fix,
                  flip,
                  fliplr,
                  flipud,
                  format,
                  fromregex,
                  function_base,
                  fv,
                  genfromtxt,
                  get_array_wrap,
                  get_include,
                  gradient,
                  hamming,
                  hanning,
                  histogram,
                  histogram2d,
                  histogramdd,
                  hsplit,
                  i0,
                  imag,
                  in1d,
                  index_exp,
                  index_tricks,
                  info,
                  insert,
                  interp,
                  intersect1d,
                  ipmt,
                  irr,
                  iscomplex,
                  iscomplexobj,
                  isin,
                  isneginf,
                  isposinf,
                  isreal,
                  isrealobj,
                  issubclass_,
                  issubdtype,
                  issubsctype,
                  iterable,
                  ix_,
                  kaiser,
                  kron,
                  load,
                  loads,
                  loadtxt,
                  lookfor,
                  mafromtxt,
                  mask_indices,
                  math,
                  median,
                  meshgrid,
                  mgrid,
                  mintypecode,
                  mirr,
                  mixins,
                  msort,
                  nan_to_num,
                  nanargmax,
                  nanargmin,
                  nancumprod,
                  nancumsum,
                  nanfunctions,
                  nanmax,
                  nanmean,
                  nanmedian,
                  nanmin,
                  nanpercentile,
                  nanprod,
                  nanstd,
                  nansum,
                  nanvar,
                  ndenumerate,
                  ndfromtxt,
                  ndindex,
                  nper,
                  npv,
                  npyio,
                  ogrid,
                  packbits,
                  pad,
                  percentile,
                  piecewise,
                  place,
                  pmt,
                  poly,
                  poly1d,
                  polyadd,
                  polyder,
                  polydiv,
                  polyfit,
                  polyint,
                  polymul,
                  polynomial,
                  polysub,
                  polyval,
                  ppmt,
                  print_function,
                  pv,
                  r_,
                  rate,
                  ravel_multi_index,
                  real,
                  real_if_close,
                  recfromcsv,
                  recfromtxt,
                  roots,
                  rot90,
                  row_stack,
                  s_,
                  safe_eval,
                  save,
                  savetxt,
                  savez,
                  savez_compressed,
                  scimath,
                  select,
                  setdiff1d,
                  setxor1d,
                  shape_base,
                  sinc,
                  sort_complex,
                  source,
                  split,
                  stride_tricks,
                  test,
                  tile,
                  tracemalloc_domain,
                  trapz,
                  tri,
                  tril,
                  tril_indices,
                  tril_indices_from,
                  trim_zeros,
                  triu,
                  triu_indices,
                  triu_indices_from,
                  twodim_base,
                  type_check,
                  typename,
                  ufunclike,
                  union1d,
                  unique,
                  unpackbits,
                  unravel_index,
                  unwrap,
                  utils,
                  vander,
                  vectorize,
                  vsplit,
                  who)
    from . import linalg
    from . import fft
    from . import polynomial
    from . import random
    from . import ctypeslib
    from . import ma
    from . import matrixlib as _mat
    from .matrixlib import (absolute_import,
                        asmatrix,
                        bench,
                        bmat,
                        defmatrix,
                        division,
                        mat,
                        matrix,
                        print_function,
                        test)
    from .compat import long

    # Make these accessible from numpy name-space
    # but not imported in from numpy import *
    if sys.version_info[0] >= 3:
        from builtins import bool, int, float, complex, object, str
        unicode = str
    else:
        from __builtin__ import bool, int, float, complex, object, unicode, str

    from .core import round, abs, max, min

    __all__.extend(['__version__', 'pkgload', 'PackageLoader',
               'show_config'])
    __all__.extend(core.__all__)
    __all__.extend(_mat.__all__)
    __all__.extend(lib.__all__)
    __all__.extend(['linalg', 'fft', 'random', 'ctypeslib', 'ma'])


    # Filter annoying Cython warnings that serve no good purpose.
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    # oldnumeric and numarray were removed in 1.9. In case some packages import
    # but do not use them, we define them here for backward compatibility.
    oldnumeric = 'removed'
    numarray = 'removed'
