"""
A NumPy sub-namespace that conforms to the Python array API standard.

This submodule accompanies NEP 47, which proposes its inclusion in NumPy.

This is a proof-of-concept namespace that wraps the corresponding NumPy
functions to give a conforming implementation of the Python array API standard
(https://data-apis.github.io/array-api/latest/). The standard is currently in
an RFC phase and comments on it are both welcome and encouraged. Comments
should be made either at https://github.com/data-apis/array-api or at
https://github.com/data-apis/consortium-feedback/discussions.

NumPy already follows the proposed spec for the most part, so this module
serves mostly as a thin wrapper around it. However, NumPy also implements a
lot of behavior that is not included in the spec, so this serves as a
restricted subset of the API. Only those functions that are part of the spec
are included in this namespace, and all functions are given with the exact
signature given in the spec, including the use of position-only arguments, and
omitting any extra keyword arguments implemented by NumPy but not part of the
spec. The behavior of some functions is also modified from the NumPy behavior
to conform to the standard. Note that the underlying array object itself is
wrapped in a wrapper Array() class, but is otherwise unchanged. This submodule
is implemented in pure Python with no C extensions.

The array API spec is designed as a "minimal API subset" and explicitly allows
libraries to include behaviors not specified by it. But users of this module
that intend to write portable code should be aware that only those behaviors
that are listed in the spec are guaranteed to be implemented across libraries.
Consequently, the NumPy implementation was chosen to be both conforming and
minimal, so that users can use this implementation of the array API namespace
and be sure that behaviors that it defines will be available in conforming
namespaces from other libraries.

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

  - The linear algebra extension in the spec will be added in a future pull
request.

  The test suite is not yet complete, and even the tests that exist are not
  guaranteed to give a comprehensive coverage of the spec. Therefore, those
  reviewing this submodule should refer to the standard documents themselves.

- There is a custom array object, numpy._array_api.Array, which is returned
  by all functions in this module. All functions in the array API namespace
  implicitly assume that they will only receive this object as input. The only
  way to create instances of this object is to use one of the array creation
  functions. It does not have a public constructor on the object itself. The
  object is a small wrapper Python class around numpy.ndarray. The main
  purpose of it is to restrict the namespace of the array object to only those
  dtypes and only those methods that are required by the spec, as well as to
  limit/change certain behavior that differs in the spec. In particular:

  - The array API namespace does not have scalar objects, only 0-d arrays.
    Operations in on Array that would create a scalar in NumPy create a 0-d
    array.

  - Indexing: Only a subset of indices supported by NumPy are required by the
    spec. The Array object restricts indexing to only allow those types of
    indices that are required by the spec. See the docstring of the
    numpy._array_api.Array._validate_indices helper function for more
    information.

  - Type promotion: Some type promotion rules are different in the spec. In
    particular, the spec does not have any value-based casting. The
    Array._promote_scalar method promotes Python scalars to arrays,
    disallowing cross-type promotions like int -> float64 that are not allowed
    in the spec. Array._normalize_two_args works around some type promotion
    quirks in NumPy, particularly, value-based casting that occurs when one
    argument of an operation is a 0-d array.

- All functions include type annotations, corresponding to those given in the
  spec (see _types.py for definitions of some custom types). These do not
  currently fully pass mypy due to some limitations in mypy.

- Dtype objects are just the NumPy dtype objects, e.g., float64 =
  np.dtype('float64'). The spec does not require any behavior on these dtype
  objects other than that they be accessible by name and be comparable by
  equality, but it was considered too much extra complexity to create custom
  objects to represent dtypes.

- The wrapper functions in this module do not do any type checking for things
  that would be impossible without leaving the _array_api namespace. For
  example, since the array API dtype objects are just the NumPy dtype objects,
  one could pass in a non-spec NumPy dtype into a function.

- All places where the implementations in this submodule are known to deviate
  from their corresponding functions in NumPy are marked with "# Note"
  comments. Reviewers should make note of these comments.

Still TODO in this module are:

- Device support and DLPack support are not yet implemented. These require
  support in NumPy itself first.

- The a non-default value for the `copy` keyword argument is not yet
  implemented on asarray. This requires support in numpy.asarray() first.

- Some functions are not yet fully tested in the array API test suite, and may
  require updates that are not yet known until the tests are written.

"""

__all__ = []

from ._constants import e, inf, nan, pi

__all__ += ['e', 'inf', 'nan', 'pi']

from ._creation_functions import asarray, arange, empty, empty_like, eye, from_dlpack, full, full_like, linspace, meshgrid, ones, ones_like, zeros, zeros_like

__all__ += ['asarray', 'arange', 'empty', 'empty_like', 'eye', 'from_dlpack', 'full', 'full_like', 'linspace', 'meshgrid', 'ones', 'ones_like', 'zeros', 'zeros_like']

from ._data_type_functions import broadcast_arrays, broadcast_to, can_cast, finfo, iinfo, result_type

__all__ += ['broadcast_arrays', 'broadcast_to', 'can_cast', 'finfo', 'iinfo', 'result_type']

from ._dtypes import int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, bool

__all__ += ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64', 'bool']

from ._elementwise_functions import abs, acos, acosh, add, asin, asinh, atan, atan2, atanh, bitwise_and, bitwise_left_shift, bitwise_invert, bitwise_or, bitwise_right_shift, bitwise_xor, ceil, cos, cosh, divide, equal, exp, expm1, floor, floor_divide, greater, greater_equal, isfinite, isinf, isnan, less, less_equal, log, log1p, log2, log10, logaddexp, logical_and, logical_not, logical_or, logical_xor, multiply, negative, not_equal, positive, pow, remainder, round, sign, sin, sinh, square, sqrt, subtract, tan, tanh, trunc

__all__ += ['abs', 'acos', 'acosh', 'add', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'bitwise_and', 'bitwise_left_shift', 'bitwise_invert', 'bitwise_or', 'bitwise_right_shift', 'bitwise_xor', 'ceil', 'cos', 'cosh', 'divide', 'equal', 'exp', 'expm1', 'floor', 'floor_divide', 'greater', 'greater_equal', 'isfinite', 'isinf', 'isnan', 'less', 'less_equal', 'log', 'log1p', 'log2', 'log10', 'logaddexp', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'multiply', 'negative', 'not_equal', 'positive', 'pow', 'remainder', 'round', 'sign', 'sin', 'sinh', 'square', 'sqrt', 'subtract', 'tan', 'tanh', 'trunc']

# einsum is not yet implemented in the array API spec.

# from ._linear_algebra_functions import einsum
# __all__ += ['einsum']

from ._linear_algebra_functions import matmul, tensordot, transpose, vecdot

__all__ += ['matmul', 'tensordot', 'transpose', 'vecdot']

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
