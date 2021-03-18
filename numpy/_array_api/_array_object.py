"""
Wrapper class around the ndarray object for the array API standard.

The array API standard defines some behaviors differently than ndarray, in
particular, type promotion rules are different (the standard has no
value-based casting). The standard also specifies a more limited subset of
array methods and functionalities than are implemented on ndarray. Since the
goal of the array_api namespace is to be a minimal implementation of the array
API standard, we need to define a separate wrapper class for the array_api
namespace.

The standard compliant class is only a wrapper class. It is *not* a subclass
of ndarray.
"""

from __future__ import annotations

import operator
from enum import IntEnum
from ._creation_functions import asarray
from ._dtypes import _boolean_dtypes, _integer_dtypes

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._types import Optional, PyCapsule, Tuple, Union, array

import numpy as np

class ndarray:
    """
    ndarray object for the array API namespace.

    See the docstring of :py:obj:`np.ndarray <numpy.ndarray>` for more
    information.

    This is a wrapper around numpy.ndarray that restricts the usage to only
    those things that are required by the array API namespace. Note,
    attributes on this object that start with a single underscore are not part
    of the API specification and should only be used internally. This object
    should not be constructed directly. Rather, use one of the creation
    functions, such as asarray().

    """
    # Use a custom constructor instead of __init__, as manually initializing
    # this class is not supported API.
    @classmethod
    def _new(cls, x, /):
        """
        This is a private method for initializing the array API ndarray
        object.

        Functions outside of the array_api submodule should not use this
        method. Use one of the creation functions instead, such as
        ``asarray``.

        """
        obj = super().__new__(cls)
        # Note: The spec does not have array scalars, only shape () arrays.
        if isinstance(x, np.generic):
            # Convert the array scalar to a shape () array
            xa = np.empty((), x.dtype)
            xa[()] = x
            x = xa
        obj._array = x
        return obj

    # Prevent ndarray() from working
    def __new__(cls, *args, **kwargs):
        raise TypeError("The array_api ndarray object should not be instantiated directly. Use an array creation function, such as asarray(), instead.")

    # These functions are not required by the spec, but are implemented for
    # the sake of usability.

    def __str__(self: array, /) -> str:
        """
        Performs the operation __str__.
        """
        return self._array.__str__().replace('array', 'ndarray')

    def __repr__(self: array, /) -> str:
        """
        Performs the operation __repr__.
        """
        return self._array.__repr__().replace('array', 'ndarray')

    # Everything below this is required by the spec.

    def __abs__(self: array, /) -> array:
        """
        Performs the operation __abs__.
        """
        res = self._array.__abs__()
        return self.__class__._new(res)

    def __add__(self: array, other: array, /) -> array:
        """
        Performs the operation __add__.
        """
        res = self._array.__add__(asarray(other)._array)
        return self.__class__._new(res)

    def __and__(self: array, other: array, /) -> array:
        """
        Performs the operation __and__.
        """
        res = self._array.__and__(asarray(other)._array)
        return self.__class__._new(res)

    def __array_namespace__(self, /, *, api_version=None):
        if api_version is not None:
            raise ValueError("Unrecognized array API version")
        from numpy import _array_api
        return _array_api

    def __bool__(self: array, /) -> bool:
        """
        Performs the operation __bool__.
        """
        # Note: This is an error here.
        if self._array.shape != ():
            raise TypeError("bool is only allowed on arrays with shape ()")
        res = self._array.__bool__()
        return res

    def __dlpack__(self: array, /, *, stream: Optional[int] = None) -> PyCapsule:
        """
        Performs the operation __dlpack__.
        """
        res = self._array.__dlpack__(stream=None)
        return self.__class__._new(res)

    def __dlpack_device__(self: array, /) -> Tuple[IntEnum, int]:
        """
        Performs the operation __dlpack_device__.
        """
        res = self._array.__dlpack_device__()
        return self.__class__._new(res)

    def __eq__(self: array, other: array, /) -> array:
        """
        Performs the operation __eq__.
        """
        res = self._array.__eq__(asarray(other)._array)
        return self.__class__._new(res)

    def __float__(self: array, /) -> float:
        """
        Performs the operation __float__.
        """
        # Note: This is an error here.
        if self._array.shape != ():
            raise TypeError("bool is only allowed on arrays with shape ()")
        res = self._array.__float__()
        return res

    def __floordiv__(self: array, other: array, /) -> array:
        """
        Performs the operation __floordiv__.
        """
        res = self._array.__floordiv__(asarray(other)._array)
        return self.__class__._new(res)

    def __ge__(self: array, other: array, /) -> array:
        """
        Performs the operation __ge__.
        """
        res = self._array.__ge__(asarray(other)._array)
        return self.__class__._new(res)

    # Note: A large fraction of allowed indices are disallowed here (see the
    # docstring below)
    @staticmethod
    def _validate_index(key, shape):
        """
        Validate an index according to the array API.

        The array API specification only requires a subset of indices that are
        supported by NumPy. This function will reject any index that is
        allowed by NumPy but not required by the array API specification. We
        always raise ``IndexError`` on such indices (the spec does not require
        any specific behavior on them, but this makes the NumPy array API
        namespace a minimal implementation of the spec).

        This function either raises IndexError if the index ``key`` is
        invalid, or a new key to be used in place of ``key`` in indexing. It
        only raises ``IndexError`` on indices that are not already rejected by
        NumPy, as NumPy will already raise the appropriate error on such
        indices. ``shape`` may be None, in which case, only cases that are
        independent of the array shape are checked.

        The following cases are allowed by NumPy, but not specified by the array
        API specification:

        - The start and stop of a slice may not be out of bounds. In
          particular, for a slice ``i:j:k`` on an axis of size ``n``, only the
          following are allowed:

          - ``i`` or ``j`` omitted (``None``).
          - ``-n <= i <= max(0, n - 1)``.
          - For ``k > 0`` or ``k`` omitted (``None``), ``-n <= j <= n``.
          - For ``k < 0``, ``-n - 1 <= j <= max(0, n - 1)``.

        - Boolean array indices are not allowed as part of a larger tuple
          index.

        - Integer array indices are not allowed (with the exception of shape
          () arrays, which are treated the same as scalars).

        Additionally, it should be noted that indices that would return a
        scalar in NumPy will return a shape () array. Array scalars are not allowed
        in the specification, only shape () arrays. This is done in the
        ``ndarray._new`` constructor, not this function.

        """
        if isinstance(key, slice):
            if shape is None:
                return key
            if shape == ():
                return key
            size = shape[0]
            # Ensure invalid slice entries are passed through.
            if key.start is not None:
                try:
                    operator.index(key.start)
                except TypeError:
                    return key
                if not (-size <= key.start <= max(0, size - 1)):
                    raise IndexError("Slices with out-of-bounds start are not allowed in the array API namespace")
            if key.stop is not None:
                try:
                    operator.index(key.stop)
                except TypeError:
                    return key
                step = 1 if key.step is None else key.step
                if (step > 0 and not (-size <= key.stop <= size)
                    or step < 0 and not (-size - 1 <= key.stop <= max(0, size - 1))):
                    raise IndexError("Slices with out-of-bounds stop are not allowed in the array API namespace")
            return key

        elif isinstance(key, tuple):
            key = tuple(ndarray._validate_index(idx, None) for idx in key)

            for idx in key:
                if isinstance(idx, np.ndarray) and idx.dtype in _boolean_dtypes or isinstance(idx, (bool, np.bool_)):
                    if len(key) == 1:
                        return key
                    raise IndexError("Boolean array indices combined with other indices are not allowed in the array API namespace")

            if shape is None:
                return key
            n_ellipsis = key.count(...)
            if n_ellipsis > 1:
                return key
            ellipsis_i = key.index(...) if n_ellipsis else len(key)

            for idx, size in list(zip(key[:ellipsis_i], shape)) + list(zip(key[:ellipsis_i:-1], shape[:ellipsis_i:-1])):
                ndarray._validate_index(idx, (size,))
            return key
        elif isinstance(key, bool):
            return key
        elif isinstance(key, ndarray):
            if key.dtype in _integer_dtypes:
                if key.shape != ():
                    raise IndexError("Integer array indices with shape != () are not allowed in the array API namespace")
            return key._array
        elif key is Ellipsis:
            return key
        elif key is None:
            raise IndexError("newaxis indices are not allowed in the array API namespace")
        try:
            return operator.index(key)
        except TypeError:
            # Note: This also omits boolean arrays that are not already in
            # ndarray() form, like a list of booleans.
            raise IndexError("Only integers, slices (`:`), ellipsis (`...`), and boolean arrays are valid indices in the array API namespace")

    def __getitem__(self: array, key: Union[int, slice, Tuple[Union[int, slice], ...], array], /) -> array:
        """
        Performs the operation __getitem__.
        """
        # Note: Only indices required by the spec are allowed. See the
        # docstring of _validate_index
        key = self._validate_index(key, self.shape)
        res = self._array.__getitem__(key)
        return self.__class__._new(res)

    def __gt__(self: array, other: array, /) -> array:
        """
        Performs the operation __gt__.
        """
        res = self._array.__gt__(asarray(other)._array)
        return self.__class__._new(res)

    def __int__(self: array, /) -> int:
        """
        Performs the operation __int__.
        """
        # Note: This is an error here.
        if self._array.shape != ():
            raise TypeError("bool is only allowed on arrays with shape ()")
        res = self._array.__int__()
        return res

    def __invert__(self: array, /) -> array:
        """
        Performs the operation __invert__.
        """
        res = self._array.__invert__()
        return self.__class__._new(res)

    def __le__(self: array, other: array, /) -> array:
        """
        Performs the operation __le__.
        """
        res = self._array.__le__(asarray(other)._array)
        return self.__class__._new(res)

    def __len__(self, /):
        """
        Performs the operation __len__.
        """
        res = self._array.__len__()
        return self.__class__._new(res)

    def __lshift__(self: array, other: array, /) -> array:
        """
        Performs the operation __lshift__.
        """
        res = self._array.__lshift__(asarray(other)._array)
        return self.__class__._new(res)

    def __lt__(self: array, other: array, /) -> array:
        """
        Performs the operation __lt__.
        """
        res = self._array.__lt__(asarray(other)._array)
        return self.__class__._new(res)

    def __matmul__(self: array, other: array, /) -> array:
        """
        Performs the operation __matmul__.
        """
        res = self._array.__matmul__(asarray(other)._array)
        return self.__class__._new(res)

    def __mod__(self: array, other: array, /) -> array:
        """
        Performs the operation __mod__.
        """
        res = self._array.__mod__(asarray(other)._array)
        return self.__class__._new(res)

    def __mul__(self: array, other: array, /) -> array:
        """
        Performs the operation __mul__.
        """
        res = self._array.__mul__(asarray(other)._array)
        return self.__class__._new(res)

    def __ne__(self: array, other: array, /) -> array:
        """
        Performs the operation __ne__.
        """
        res = self._array.__ne__(asarray(other)._array)
        return self.__class__._new(res)

    def __neg__(self: array, /) -> array:
        """
        Performs the operation __neg__.
        """
        res = self._array.__neg__()
        return self.__class__._new(res)

    def __or__(self: array, other: array, /) -> array:
        """
        Performs the operation __or__.
        """
        res = self._array.__or__(asarray(other)._array)
        return self.__class__._new(res)

    def __pos__(self: array, /) -> array:
        """
        Performs the operation __pos__.
        """
        res = self._array.__pos__()
        return self.__class__._new(res)

    def __pow__(self: array, other: array, /) -> array:
        """
        Performs the operation __pow__.
        """
        res = self._array.__pow__(asarray(other)._array)
        return self.__class__._new(res)

    def __rshift__(self: array, other: array, /) -> array:
        """
        Performs the operation __rshift__.
        """
        res = self._array.__rshift__(asarray(other)._array)
        return self.__class__._new(res)

    def __setitem__(self, key, value, /):
        """
        Performs the operation __setitem__.
        """
        # Note: Only indices required by the spec are allowed. See the
        # docstring of _validate_index
        key = self._validate_index(key, self.shape)
        res = self._array.__setitem__(key, asarray(value)._array)
        return self.__class__._new(res)

    def __sub__(self: array, other: array, /) -> array:
        """
        Performs the operation __sub__.
        """
        res = self._array.__sub__(asarray(other)._array)
        return self.__class__._new(res)

    def __truediv__(self: array, other: array, /) -> array:
        """
        Performs the operation __truediv__.
        """
        res = self._array.__truediv__(asarray(other)._array)
        return self.__class__._new(res)

    def __xor__(self: array, other: array, /) -> array:
        """
        Performs the operation __xor__.
        """
        res = self._array.__xor__(asarray(other)._array)
        return self.__class__._new(res)

    def __iadd__(self: array, other: array, /) -> array:
        """
        Performs the operation __iadd__.
        """
        res = self._array.__iadd__(asarray(other)._array)
        return self.__class__._new(res)

    def __radd__(self: array, other: array, /) -> array:
        """
        Performs the operation __radd__.
        """
        res = self._array.__radd__(asarray(other)._array)
        return self.__class__._new(res)

    def __iand__(self: array, other: array, /) -> array:
        """
        Performs the operation __iand__.
        """
        res = self._array.__iand__(asarray(other)._array)
        return self.__class__._new(res)

    def __rand__(self: array, other: array, /) -> array:
        """
        Performs the operation __rand__.
        """
        res = self._array.__rand__(asarray(other)._array)
        return self.__class__._new(res)

    def __ifloordiv__(self: array, other: array, /) -> array:
        """
        Performs the operation __ifloordiv__.
        """
        res = self._array.__ifloordiv__(asarray(other)._array)
        return self.__class__._new(res)

    def __rfloordiv__(self: array, other: array, /) -> array:
        """
        Performs the operation __rfloordiv__.
        """
        res = self._array.__rfloordiv__(asarray(other)._array)
        return self.__class__._new(res)

    def __ilshift__(self: array, other: array, /) -> array:
        """
        Performs the operation __ilshift__.
        """
        res = self._array.__ilshift__(asarray(other)._array)
        return self.__class__._new(res)

    def __rlshift__(self: array, other: array, /) -> array:
        """
        Performs the operation __rlshift__.
        """
        res = self._array.__rlshift__(asarray(other)._array)
        return self.__class__._new(res)

    def __imatmul__(self: array, other: array, /) -> array:
        """
        Performs the operation __imatmul__.
        """
        res = self._array.__imatmul__(asarray(other)._array)
        return self.__class__._new(res)

    def __rmatmul__(self: array, other: array, /) -> array:
        """
        Performs the operation __rmatmul__.
        """
        res = self._array.__rmatmul__(asarray(other)._array)
        return self.__class__._new(res)

    def __imod__(self: array, other: array, /) -> array:
        """
        Performs the operation __imod__.
        """
        res = self._array.__imod__(asarray(other)._array)
        return self.__class__._new(res)

    def __rmod__(self: array, other: array, /) -> array:
        """
        Performs the operation __rmod__.
        """
        res = self._array.__rmod__(asarray(other)._array)
        return self.__class__._new(res)

    def __imul__(self: array, other: array, /) -> array:
        """
        Performs the operation __imul__.
        """
        res = self._array.__imul__(asarray(other)._array)
        return self.__class__._new(res)

    def __rmul__(self: array, other: array, /) -> array:
        """
        Performs the operation __rmul__.
        """
        res = self._array.__rmul__(asarray(other)._array)
        return self.__class__._new(res)

    def __ior__(self: array, other: array, /) -> array:
        """
        Performs the operation __ior__.
        """
        res = self._array.__ior__(asarray(other)._array)
        return self.__class__._new(res)

    def __ror__(self: array, other: array, /) -> array:
        """
        Performs the operation __ror__.
        """
        res = self._array.__ror__(asarray(other)._array)
        return self.__class__._new(res)

    def __ipow__(self: array, other: array, /) -> array:
        """
        Performs the operation __ipow__.
        """
        res = self._array.__ipow__(asarray(other)._array)
        return self.__class__._new(res)

    def __rpow__(self: array, other: array, /) -> array:
        """
        Performs the operation __rpow__.
        """
        res = self._array.__rpow__(asarray(other)._array)
        return self.__class__._new(res)

    def __irshift__(self: array, other: array, /) -> array:
        """
        Performs the operation __irshift__.
        """
        res = self._array.__irshift__(asarray(other)._array)
        return self.__class__._new(res)

    def __rrshift__(self: array, other: array, /) -> array:
        """
        Performs the operation __rrshift__.
        """
        res = self._array.__rrshift__(asarray(other)._array)
        return self.__class__._new(res)

    def __isub__(self: array, other: array, /) -> array:
        """
        Performs the operation __isub__.
        """
        res = self._array.__isub__(asarray(other)._array)
        return self.__class__._new(res)

    def __rsub__(self: array, other: array, /) -> array:
        """
        Performs the operation __rsub__.
        """
        res = self._array.__rsub__(asarray(other)._array)
        return self.__class__._new(res)

    def __itruediv__(self: array, other: array, /) -> array:
        """
        Performs the operation __itruediv__.
        """
        res = self._array.__itruediv__(asarray(other)._array)
        return self.__class__._new(res)

    def __rtruediv__(self: array, other: array, /) -> array:
        """
        Performs the operation __rtruediv__.
        """
        res = self._array.__rtruediv__(asarray(other)._array)
        return self.__class__._new(res)

    def __ixor__(self: array, other: array, /) -> array:
        """
        Performs the operation __ixor__.
        """
        res = self._array.__ixor__(asarray(other)._array)
        return self.__class__._new(res)

    def __rxor__(self: array, other: array, /) -> array:
        """
        Performs the operation __rxor__.
        """
        res = self._array.__rxor__(asarray(other)._array)
        return self.__class__._new(res)

    @property
    def dtype(self):
        """
        Array API compatible wrapper for :py:meth:`np.ndaray.dtype <numpy.ndarray.dtype>`.

        See its docstring for more information.
        """
        return self._array.dtype

    @property
    def device(self):
        """
        Array API compatible wrapper for :py:meth:`np.ndaray.device <numpy.ndarray.device>`.

        See its docstring for more information.
        """
        return self._array.device

    @property
    def ndim(self):
        """
        Array API compatible wrapper for :py:meth:`np.ndaray.ndim <numpy.ndarray.ndim>`.

        See its docstring for more information.
        """
        return self._array.ndim

    @property
    def shape(self):
        """
        Array API compatible wrapper for :py:meth:`np.ndaray.shape <numpy.ndarray.shape>`.

        See its docstring for more information.
        """
        return self._array.shape

    @property
    def size(self):
        """
        Array API compatible wrapper for :py:meth:`np.ndaray.size <numpy.ndarray.size>`.

        See its docstring for more information.
        """
        return self._array.size

    @property
    def T(self):
        """
        Array API compatible wrapper for :py:meth:`np.ndaray.T <numpy.ndarray.T>`.

        See its docstring for more information.
        """
        return self._array.T
