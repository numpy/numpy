"""
A NumPy sub-namespace that conforms to the Python array API standard.

This is a proof-of-concept namespace that wraps the corresponding NumPy
functions to give a conforming implementation of the Python array API standard
(https://data-apis.github.io/array-api/latest/). The standard is currently in
an RFC phase and comments on it are both welcome and encouraged. Comments
should be made either at https://github.com/data-apis/array-api or at
https://github.com/data-apis/consortium-feedback/discussions.

This submodule will be accompanied with a NEP (not yet written) proposing its
inclusion in NumPy.

NumPy already follows the proposed spec for the most part, so this module
serves mostly as a thin wrapper around it. However, NumPy also implements a
lot of behavior that is not included in the spec, so this serves as a
restricted subset of the API. Only those functions that are part of the spec
are included in this namespace, and all functions are given with the exact
signature given in the spec, including the use of position-only arguments, and
omitting any extra keyword arguments implemented by NumPy but not part of the
spec. Note that the array object itself is unchanged, as implementing a
restricted subclass of ndarray seems unnecessarily complex for the purposes of
this namespace, so the API of array methods and other behaviors of the array
object will include things that are not part of the spec.

The spec is designed as a "minimal API subset" and explicitly allows libraries
to include behaviors not specified by it. But users of this module that intend
to write portable code should be aware that only those behaviors that are
listed in the spec are guaranteed to be implemented across libraries.

A few notes about the current state of this submodule:

- There is a test suite that tests modules against the array API standard at
  https://github.com/data-apis/array-api-tests. The test suite is still a work
  in progress, but the existing tests pass on this module, with a few
  exceptions:

  - Device support is not yet implemented in NumPy
    (https://data-apis.github.io/array-api/latest/design_topics/device_support.html).
    As a result, the `device` attribute of the array object is missing, and
    array creation functions that take the `device` keyword argument will fail
    with NotImplementedError.

  - DLPack support (see https://github.com/data-apis/array-api/pull/106) is
    not included here, as it requires a full implementation in NumPy proper
    first.

  - np.argmin and np.argmax do not implement the keepdims keyword argument.

  - Some linear algebra functions in the spec are still a work in progress (to
    be added soon). These will be updated once the spec is.

  - Some tests in the test suite are still not fully correct in that they test
    all datatypes whereas certain functions are only defined for a subset of
    datatypes.

  The test suite is yet complete, and even the tests that exist are not
  guaranteed to give a comprehensive coverage of the spec. Therefore, those
  reviewing this submodule should refer to the standard documents themselves.

- All functions include type annotations, corresponding to those given in the
  spec (see _types.py for definitions of the types 'array', 'device', and
  'dtype'). These do not currently fully pass mypy due to some limitations in
  mypy.

- The array object is not modified at all. That means that functions return
  np.ndarray, which has methods and attributes that aren't part of the spec.
  Modifying/subclassing ndarray for the purposes of the array API namespace
  was considered too complex for this initial implementation.

- All functions that would otherwise accept array-like input have been wrapped
  to only accept ndarray (with the exception of methods on the array object,
  which are not modified).

- All places where the implementations in this submodule are known to deviate
  from their corresponding functions in NumPy are marked with "# Note"
  comments. Reviewers should make note of these comments.

"""

__all__ = []

from ._constants import e, inf, nan, pi

__all__ += ['e', 'inf', 'nan', 'pi']

from ._creation_functions import arange, empty, empty_like, eye, full, full_like, linspace, ones, ones_like, zeros, zeros_like

__all__ += ['arange', 'empty', 'empty_like', 'eye', 'full', 'full_like', 'linspace', 'ones', 'ones_like', 'zeros', 'zeros_like']

from ._dtypes import int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, bool

__all__ += ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64', 'bool']

from ._elementwise_functions import abs, acos, acosh, add, asin, asinh, atan, atan2, atanh, bitwise_and, bitwise_left_shift, bitwise_invert, bitwise_or, bitwise_right_shift, bitwise_xor, ceil, cos, cosh, divide, equal, exp, expm1, floor, floor_divide, greater, greater_equal, isfinite, isinf, isnan, less, less_equal, log, log1p, log2, log10, logical_and, logical_not, logical_or, logical_xor, multiply, negative, not_equal, positive, pow, remainder, round, sign, sin, sinh, square, sqrt, subtract, tan, tanh, trunc

__all__ += ['abs', 'acos', 'acosh', 'add', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'bitwise_and', 'bitwise_left_shift', 'bitwise_invert', 'bitwise_or', 'bitwise_right_shift', 'bitwise_xor', 'ceil', 'cos', 'cosh', 'divide', 'equal', 'exp', 'expm1', 'floor', 'floor_divide', 'greater', 'greater_equal', 'isfinite', 'isinf', 'isnan', 'less', 'less_equal', 'log', 'log1p', 'log2', 'log10', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'multiply', 'negative', 'not_equal', 'positive', 'pow', 'remainder', 'round', 'sign', 'sin', 'sinh', 'square', 'sqrt', 'subtract', 'tan', 'tanh', 'trunc']

from ._linear_algebra_functions import cross, det, diagonal, inv, norm, outer, trace, transpose

__all__ += ['cross', 'det', 'diagonal', 'inv', 'norm', 'outer', 'trace', 'transpose']

# from ._linear_algebra_functions import cholesky, cross, det, diagonal, dot, eig, eigvalsh, einsum, inv, lstsq, matmul, matrix_power, matrix_rank, norm, outer, pinv, qr, slogdet, solve, svd, trace, transpose
#
# __all__ += ['cholesky', 'cross', 'det', 'diagonal', 'dot', 'eig', 'eigvalsh', 'einsum', 'inv', 'lstsq', 'matmul', 'matrix_power', 'matrix_rank', 'norm', 'outer', 'pinv', 'qr', 'slogdet', 'solve', 'svd', 'trace', 'transpose']

from ._manipulation_functions import concat, expand_dims, flip, reshape, roll, squeeze, stack

__all__ += ['concat', 'expand_dims', 'flip', 'reshape', 'roll', 'squeeze', 'stack']

from ._searching_functions import argmax, argmin, nonzero, where

__all__ += ['argmax', 'argmin', 'nonzero', 'where']

from ._set_functions import unique

__all__ += ['unique']

from ._sorting_functions import argsort, sort

__all__ += ['argsort', 'sort']

from ._statistical_functions import max, mean, min, prod, std, sum, var

__all__ += ['max', 'mean', 'min', 'prod', 'std', 'sum', 'var']

from ._utility_functions import all, any

__all__ += ['all', 'any']
