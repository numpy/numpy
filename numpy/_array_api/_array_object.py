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

from enum import IntEnum
from ._types import Optional, PyCapsule, Tuple, Union, array
from ._creation_functions import asarray

import numpy as np

class ndarray:
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
            # x[...] converts an array scalar to a shape () array.
            x = x[...]
        obj._array = x
        return obj

    # Prevent ndarray() from working
    def __new__(cls, *args, **kwargs):
        raise TypeError("The array_api ndarray object should not be instantiated directly. Use an array creation function, such as asarray(), instead.")

    # These functions are not required by the spec, but are implemented for
    # the sake of usability.

    def __str__(x: array, /) -> str:
        """
        Performs the operation __str__.
        """
        return x._array.__str__().replace('array', 'ndarray')

    def __repr__(x: array, /) -> str:
        """
        Performs the operation __repr__.
        """
        return x._array.__repr__().replace('array', 'ndarray')

    # Everything below this is required by the spec.

    def __abs__(x: array, /) -> array:
        """
        Performs the operation __abs__.
        """
        res = x._array.__abs__()
        return x.__class__._new(res)

    def __add__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __add__.
        """
        res = x1._array.__add__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __and__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __and__.
        """
        res = x1._array.__and__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __bool__(x: array, /) -> bool:
        """
        Performs the operation __bool__.
        """
        # Note: This is an error here.
        if x._array.shape != ():
            raise TypeError("bool is only allowed on arrays with shape ()")
        res = x._array.__bool__()
        return res

    def __dlpack__(x: array, /, *, stream: Optional[int] = None) -> PyCapsule:
        """
        Performs the operation __dlpack__.
        """
        res = x._array.__dlpack__(stream=None)
        return x.__class__._new(res)

    def __dlpack_device__(x: array, /) -> Tuple[IntEnum, int]:
        """
        Performs the operation __dlpack_device__.
        """
        res = x._array.__dlpack_device__()
        return x.__class__._new(res)

    def __eq__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __eq__.
        """
        res = x1._array.__eq__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __float__(x: array, /) -> float:
        """
        Performs the operation __float__.
        """
        # Note: This is an error here.
        if x._array.shape != ():
            raise TypeError("bool is only allowed on arrays with shape ()")
        res = x._array.__float__()
        return res

    def __floordiv__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __floordiv__.
        """
        res = x1._array.__floordiv__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __ge__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __ge__.
        """
        res = x1._array.__ge__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __getitem__(x: array, key: Union[int, slice, Tuple[Union[int, slice], ...], array], /) -> array:
        """
        Performs the operation __getitem__.
        """
        res = x._array.__getitem__(key)
        return x.__class__._new(res)

    def __gt__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __gt__.
        """
        res = x1._array.__gt__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __int__(x: array, /) -> int:
        """
        Performs the operation __int__.
        """
        # Note: This is an error here.
        if x._array.shape != ():
            raise TypeError("bool is only allowed on arrays with shape ()")
        res = x._array.__int__()
        return res

    def __invert__(x: array, /) -> array:
        """
        Performs the operation __invert__.
        """
        res = x._array.__invert__()
        return x.__class__._new(res)

    def __le__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __le__.
        """
        res = x1._array.__le__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __len__(x, /):
        """
        Performs the operation __len__.
        """
        res = x._array.__len__()
        return x.__class__._new(res)

    def __lshift__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __lshift__.
        """
        res = x1._array.__lshift__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __lt__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __lt__.
        """
        res = x1._array.__lt__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __matmul__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __matmul__.
        """
        res = x1._array.__matmul__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __mod__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __mod__.
        """
        res = x1._array.__mod__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __mul__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __mul__.
        """
        res = x1._array.__mul__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __ne__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __ne__.
        """
        res = x1._array.__ne__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __neg__(x: array, /) -> array:
        """
        Performs the operation __neg__.
        """
        res = x._array.__neg__()
        return x.__class__._new(res)

    def __or__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __or__.
        """
        res = x1._array.__or__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __pos__(x: array, /) -> array:
        """
        Performs the operation __pos__.
        """
        res = x._array.__pos__()
        return x.__class__._new(res)

    def __pow__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __pow__.
        """
        res = x1._array.__pow__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __rshift__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rshift__.
        """
        res = x1._array.__rshift__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __setitem__(x, key, value, /):
        """
        Performs the operation __setitem__.
        """
        res = x._array.__setitem__(key, asarray(value)._array)
        return x.__class__._new(res)

    def __sub__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __sub__.
        """
        res = x1._array.__sub__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __truediv__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __truediv__.
        """
        res = x1._array.__truediv__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __xor__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __xor__.
        """
        res = x1._array.__xor__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __iadd__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __iadd__.
        """
        res = x1._array.__iadd__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __radd__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __radd__.
        """
        res = x1._array.__radd__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __iand__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __iand__.
        """
        res = x1._array.__iand__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __rand__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rand__.
        """
        res = x1._array.__rand__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __ifloordiv__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __ifloordiv__.
        """
        res = x1._array.__ifloordiv__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __rfloordiv__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rfloordiv__.
        """
        res = x1._array.__rfloordiv__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __ilshift__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __ilshift__.
        """
        res = x1._array.__ilshift__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __rlshift__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rlshift__.
        """
        res = x1._array.__rlshift__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __imatmul__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __imatmul__.
        """
        res = x1._array.__imatmul__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __rmatmul__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rmatmul__.
        """
        res = x1._array.__rmatmul__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __imod__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __imod__.
        """
        res = x1._array.__imod__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __rmod__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rmod__.
        """
        res = x1._array.__rmod__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __imul__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __imul__.
        """
        res = x1._array.__imul__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __rmul__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rmul__.
        """
        res = x1._array.__rmul__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __ior__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __ior__.
        """
        res = x1._array.__ior__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __ror__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __ror__.
        """
        res = x1._array.__ror__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __ipow__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __ipow__.
        """
        res = x1._array.__ipow__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __rpow__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rpow__.
        """
        res = x1._array.__rpow__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __irshift__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __irshift__.
        """
        res = x1._array.__irshift__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __rrshift__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rrshift__.
        """
        res = x1._array.__rrshift__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __isub__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __isub__.
        """
        res = x1._array.__isub__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __rsub__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rsub__.
        """
        res = x1._array.__rsub__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __itruediv__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __itruediv__.
        """
        res = x1._array.__itruediv__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __rtruediv__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rtruediv__.
        """
        res = x1._array.__rtruediv__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __ixor__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __ixor__.
        """
        res = x1._array.__ixor__(asarray(x2)._array)
        return x1.__class__._new(res)

    def __rxor__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rxor__.
        """
        res = x1._array.__rxor__(asarray(x2)._array)
        return x1.__class__._new(res)

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
