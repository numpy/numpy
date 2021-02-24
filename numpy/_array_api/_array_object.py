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
        obj._array = x
        return obj

    # Prevent ndarray() from working
    def __new__(cls, *args, **kwargs):
        raise TypeError("The array_api ndarray object should not be instantiated directly. Use an array creation function, such as asarray(), instead.")

    def __abs__(x: array, /) -> array:
        """
        Performs the operation __abs__.
        """
        res = x._array.__abs__(x)
        return x.__class__._new(res)

    def __add__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __add__.
        """
        res = x1._array.__add__(x1, x2)
        return x1.__class__._new(res)

    def __and__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __and__.
        """
        res = x1._array.__and__(x1, x2)
        return x1.__class__._new(res)

    def __bool__(x: array, /) -> bool:
        """
        Performs the operation __bool__.
        """
        res = x._array.__bool__(x)
        return x.__class__._new(res)

    def __dlpack__(x: array, /, *, stream: Optional[int] = None) -> PyCapsule:
        """
        Performs the operation __dlpack__.
        """
        res = x._array.__dlpack__(x, stream=None)
        return x.__class__._new(res)

    def __dlpack_device__(x: array, /) -> Tuple[IntEnum, int]:
        """
        Performs the operation __dlpack_device__.
        """
        res = x._array.__dlpack_device__(x)
        return x.__class__._new(res)

    def __eq__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __eq__.
        """
        res = x1._array.__eq__(x1, x2)
        return x1.__class__._new(res)

    def __float__(x: array, /) -> float:
        """
        Performs the operation __float__.
        """
        res = x._array.__float__(x)
        return x.__class__._new(res)

    def __floordiv__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __floordiv__.
        """
        res = x1._array.__floordiv__(x1, x2)
        return x1.__class__._new(res)

    def __ge__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __ge__.
        """
        res = x1._array.__ge__(x1, x2)
        return x1.__class__._new(res)

    def __getitem__(x: array, key: Union[int, slice, Tuple[Union[int, slice], ...], array], /) -> array:
        """
        Performs the operation __getitem__.
        """
        res = x._array.__getitem__(x, key)
        return x.__class__._new(res)

    def __gt__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __gt__.
        """
        res = x1._array.__gt__(x1, x2)
        return x1.__class__._new(res)

    def __int__(x: array, /) -> int:
        """
        Performs the in-place operation __int__.
        """
        x._array.__int__(x)

    def __invert__(x: array, /) -> array:
        """
        Performs the in-place operation __invert__.
        """
        x._array.__invert__(x)

    def __le__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __le__.
        """
        res = x1._array.__le__(x1, x2)
        return x1.__class__._new(res)

    def __len__(x, /):
        """
        Performs the operation __len__.
        """
        res = x._array.__len__(x)
        return x.__class__._new(res)

    def __lshift__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __lshift__.
        """
        res = x1._array.__lshift__(x1, x2)
        return x1.__class__._new(res)

    def __lt__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __lt__.
        """
        res = x1._array.__lt__(x1, x2)
        return x1.__class__._new(res)

    def __matmul__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __matmul__.
        """
        res = x1._array.__matmul__(x1, x2)
        return x1.__class__._new(res)

    def __mod__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __mod__.
        """
        res = x1._array.__mod__(x1, x2)
        return x1.__class__._new(res)

    def __mul__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __mul__.
        """
        res = x1._array.__mul__(x1, x2)
        return x1.__class__._new(res)

    def __ne__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __ne__.
        """
        res = x1._array.__ne__(x1, x2)
        return x1.__class__._new(res)

    def __neg__(x: array, /) -> array:
        """
        Performs the operation __neg__.
        """
        res = x._array.__neg__(x)
        return x.__class__._new(res)

    def __or__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __or__.
        """
        res = x1._array.__or__(x1, x2)
        return x1.__class__._new(res)

    def __pos__(x: array, /) -> array:
        """
        Performs the operation __pos__.
        """
        res = x._array.__pos__(x)
        return x.__class__._new(res)

    def __pow__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __pow__.
        """
        res = x1._array.__pow__(x1, x2)
        return x1.__class__._new(res)

    def __rshift__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rshift__.
        """
        res = x1._array.__rshift__(x1, x2)
        return x1.__class__._new(res)

    def __setitem__(x, key, value, /):
        """
        Performs the operation __setitem__.
        """
        res = x._array.__setitem__(x, key, value)
        return x.__class__._new(res)

    def __sub__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __sub__.
        """
        res = x1._array.__sub__(x1, x2)
        return x1.__class__._new(res)

    def __truediv__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __truediv__.
        """
        res = x1._array.__truediv__(x1, x2)
        return x1.__class__._new(res)

    def __xor__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __xor__.
        """
        res = x1._array.__xor__(x1, x2)
        return x1.__class__._new(res)

    def __iadd__(x1: array, x2: array, /) -> array:
        """
        Performs the in-place operation __iadd__.
        """
        x1._array.__iadd__(x1, x2)

    def __radd__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __radd__.
        """
        res = x1._array.__radd__(x1, x2)
        return x1.__class__._new(res)

    def __iand__(x1: array, x2: array, /) -> array:
        """
        Performs the in-place operation __iand__.
        """
        x1._array.__iand__(x1, x2)

    def __rand__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rand__.
        """
        res = x1._array.__rand__(x1, x2)
        return x1.__class__._new(res)

    def __ifloordiv__(x1: array, x2: array, /) -> array:
        """
        Performs the in-place operation __ifloordiv__.
        """
        x1._array.__ifloordiv__(x1, x2)

    def __rfloordiv__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rfloordiv__.
        """
        res = x1._array.__rfloordiv__(x1, x2)
        return x1.__class__._new(res)

    def __ilshift__(x1: array, x2: array, /) -> array:
        """
        Performs the in-place operation __ilshift__.
        """
        x1._array.__ilshift__(x1, x2)

    def __rlshift__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rlshift__.
        """
        res = x1._array.__rlshift__(x1, x2)
        return x1.__class__._new(res)

    def __imatmul__(x1: array, x2: array, /) -> array:
        """
        Performs the in-place operation __imatmul__.
        """
        x1._array.__imatmul__(x1, x2)

    def __rmatmul__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rmatmul__.
        """
        res = x1._array.__rmatmul__(x1, x2)
        return x1.__class__._new(res)

    def __imod__(x1: array, x2: array, /) -> array:
        """
        Performs the in-place operation __imod__.
        """
        x1._array.__imod__(x1, x2)

    def __rmod__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rmod__.
        """
        res = x1._array.__rmod__(x1, x2)
        return x1.__class__._new(res)

    def __imul__(x1: array, x2: array, /) -> array:
        """
        Performs the in-place operation __imul__.
        """
        x1._array.__imul__(x1, x2)

    def __rmul__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rmul__.
        """
        res = x1._array.__rmul__(x1, x2)
        return x1.__class__._new(res)

    def __ior__(x1: array, x2: array, /) -> array:
        """
        Performs the in-place operation __ior__.
        """
        x1._array.__ior__(x1, x2)

    def __ror__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __ror__.
        """
        res = x1._array.__ror__(x1, x2)
        return x1.__class__._new(res)

    def __ipow__(x1: array, x2: array, /) -> array:
        """
        Performs the in-place operation __ipow__.
        """
        x1._array.__ipow__(x1, x2)

    def __rpow__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rpow__.
        """
        res = x1._array.__rpow__(x1, x2)
        return x1.__class__._new(res)

    def __irshift__(x1: array, x2: array, /) -> array:
        """
        Performs the in-place operation __irshift__.
        """
        x1._array.__irshift__(x1, x2)

    def __rrshift__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rrshift__.
        """
        res = x1._array.__rrshift__(x1, x2)
        return x1.__class__._new(res)

    def __isub__(x1: array, x2: array, /) -> array:
        """
        Performs the in-place operation __isub__.
        """
        x1._array.__isub__(x1, x2)

    def __rsub__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rsub__.
        """
        res = x1._array.__rsub__(x1, x2)
        return x1.__class__._new(res)

    def __itruediv__(x1: array, x2: array, /) -> array:
        """
        Performs the in-place operation __itruediv__.
        """
        x1._array.__itruediv__(x1, x2)

    def __rtruediv__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rtruediv__.
        """
        res = x1._array.__rtruediv__(x1, x2)
        return x1.__class__._new(res)

    def __ixor__(x1: array, x2: array, /) -> array:
        """
        Performs the in-place operation __ixor__.
        """
        x1._array.__ixor__(x1, x2)

    def __rxor__(x1: array, x2: array, /) -> array:
        """
        Performs the operation __rxor__.
        """
        res = x1._array.__rxor__(x1, x2)
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
