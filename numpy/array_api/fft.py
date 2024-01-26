from __future__ import annotations

from typing import TYPE_CHECKING, Union, Optional, Literal

if TYPE_CHECKING:
    from ._typing import Device
    from collections.abc import Sequence

from ._dtypes import (
    _floating_dtypes,
    _real_floating_dtypes,
    _complex_floating_dtypes,
    float32,
    complex64,
)
from ._array_object import Array, CPU_DEVICE
from ._data_type_functions import astype

import numpy as np

def fft(
    x: Array,
    /,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.fft <numpy.fft.fft>`.

    See its docstring for more information.
    """
    if x.dtype not in _complex_floating_dtypes:
        raise TypeError("Only complex floating-point dtypes are allowed in fft")
    res = Array._new(np.fft.fft(x._array, n=n, axis=axis, norm=norm))
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == complex64:
        return astype(res, complex64)
    return res

def ifft(
    x: Array,
    /,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.ifft <numpy.fft.ifft>`.

    See its docstring for more information.
    """
    if x.dtype not in _complex_floating_dtypes:
        raise TypeError("Only complex floating-point dtypes are allowed in ifft")
    res = Array._new(np.fft.ifft(x._array, n=n, axis=axis, norm=norm))
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == complex64:
        return astype(res, complex64)
    return res

def fftn(
    x: Array,
    /,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.fftn <numpy.fft.fftn>`.

    See its docstring for more information.
    """
    if x.dtype not in _complex_floating_dtypes:
        raise TypeError("Only complex floating-point dtypes are allowed in fftn")
    res = Array._new(np.fft.fftn(x._array, s=s, axes=axes, norm=norm))
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == complex64:
        return astype(res, complex64)
    return res

def ifftn(
    x: Array,
    /,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.ifftn <numpy.fft.ifftn>`.

    See its docstring for more information.
    """
    if x.dtype not in _complex_floating_dtypes:
        raise TypeError("Only complex floating-point dtypes are allowed in ifftn")
    res = Array._new(np.fft.ifftn(x._array, s=s, axes=axes, norm=norm))
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == complex64:
        return astype(res, complex64)
    return res

def rfft(
    x: Array,
    /,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.rfft <numpy.fft.rfft>`.

    See its docstring for more information.
    """
    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in rfft")
    res = Array._new(np.fft.rfft(x._array, n=n, axis=axis, norm=norm))
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == float32:
        return astype(res, complex64)
    return res

def irfft(
    x: Array,
    /,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.irfft <numpy.fft.irfft>`.

    See its docstring for more information.
    """
    if x.dtype not in _complex_floating_dtypes:
        raise TypeError("Only complex floating-point dtypes are allowed in irfft")
    res = Array._new(np.fft.irfft(x._array, n=n, axis=axis, norm=norm))
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == complex64:
        return astype(res, float32)
    return res

def rfftn(
    x: Array,
    /,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.rfftn <numpy.fft.rfftn>`.

    See its docstring for more information.
    """
    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in rfftn")
    res = Array._new(np.fft.rfftn(x._array, s=s, axes=axes, norm=norm))
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == float32:
        return astype(res, complex64)
    return res

def irfftn(
    x: Array,
    /,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.irfftn <numpy.fft.irfftn>`.

    See its docstring for more information.
    """
    if x.dtype not in _complex_floating_dtypes:
        raise TypeError("Only complex floating-point dtypes are allowed in irfftn")
    res = Array._new(np.fft.irfftn(x._array, s=s, axes=axes, norm=norm))
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == complex64:
        return astype(res, float32)
    return res

def hfft(
    x: Array,
    /,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.hfft <numpy.fft.hfft>`.

    See its docstring for more information.
    """
    if x.dtype not in _complex_floating_dtypes:
        raise TypeError("Only complex floating-point dtypes are allowed in hfft")
    res = Array._new(np.fft.hfft(x._array, n=n, axis=axis, norm=norm))
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == complex64:
        return astype(res, float32)
    return res

def ihfft(
    x: Array,
    /,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.ihfft <numpy.fft.ihfft>`.

    See its docstring for more information.
    """
    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in ihfft")
    res = Array._new(np.fft.ihfft(x._array, n=n, axis=axis, norm=norm))
    # Note: np.fft functions improperly upcast float32 and complex64 to
    # complex128
    if x.dtype == float32:
        return astype(res, complex64)
    return res

def fftfreq(n: int, /, *, d: float = 1.0, device: Optional[Device] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.fftfreq <numpy.fft.fftfreq>`.

    See its docstring for more information.
    """
    if device not in [CPU_DEVICE, None]:
        raise ValueError(f"Unsupported device {device!r}")
    return Array._new(np.fft.fftfreq(n, d=d))

def rfftfreq(n: int, /, *, d: float = 1.0, device: Optional[Device] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.rfftfreq <numpy.fft.rfftfreq>`.

    See its docstring for more information.
    """
    if device not in [CPU_DEVICE, None]:
        raise ValueError(f"Unsupported device {device!r}")
    return Array._new(np.fft.rfftfreq(n, d=d))

def fftshift(x: Array, /, *, axes: Union[int, Sequence[int]] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.fftshift <numpy.fft.fftshift>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in fftshift")
    return Array._new(np.fft.fftshift(x._array, axes=axes))

def ifftshift(x: Array, /, *, axes: Union[int, Sequence[int]] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.fft.ifftshift <numpy.fft.ifftshift>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in ifftshift")
    return Array._new(np.fft.ifftshift(x._array, axes=axes))

__all__ = [
    "fft",
    "ifft",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
]
