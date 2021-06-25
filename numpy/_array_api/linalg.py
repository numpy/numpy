from __future__ import annotations

from ._dtypes import _floating_dtypes, _numeric_dtypes
from ._array_object import ndarray

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._types import Literal, Optional, Sequence, Tuple, Union, array

from typing import NamedTuple

import numpy.linalg
import numpy as np

class eighresult(NamedTuple):
    u: array
    v: array

class lstsqresult(NamedTuple):
    x: array
    residuals: array
    rank: array
    s: array

class qrresult(NamedTuple):
    q: array
    r: array

class slogdetresult(NamedTuple):
    sign: array
    logabsdet: array

class svdresult(NamedTuple):
    u: array
    s: array
    v: array

# Note: the inclusion of the upper keyword is different from
# np.linalg.cholesky, which does not have it.
def cholesky(x: array, /, *, upper: bool = False) -> array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.cholesky <numpy.linalg.cholesky>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.cholesky.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in cholesky')
    L = np.linalg.cholesky(x._array)
    if upper:
        L = np.moveaxis(L, -1, -2)
    return ndarray._new(L)

def cross(x1: array, x2: array, /, *, axis: int = -1) -> array:
    """
    Array API compatible wrapper for :py:func:`np.cross <numpy.cross>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in cross')
    return ndarray._new(np.cross(x1._array, x2._array, axis=axis))

def det(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.det <numpy.linalg.det>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.det.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in det')
    return ndarray._new(np.linalg.det(x._array))

def diagonal(x: array, /, *, axis1: int = 0, axis2: int = 1, offset: int = 0) -> array:
    """
    Array API compatible wrapper for :py:func:`np.diagonal <numpy.diagonal>`.

    See its docstring for more information.
    """
    return ndarray._new(np.diagonal(x._array, axis1=axis1, axis2=axis2, offset=offset))

# eig() and eigvals() require complex numbers and will be added in a later
# version of the array API specification.

# def eig():
#     """
#     Array API compatible wrapper for :py:func:`np.eig <numpy.eig>`.
#
#     See its docstring for more information.
#     """
#     return np.eig()

# Note: the keyword argument name upper is different from np.linalg.eigh
def eigh(x: array, /, *, upper: bool = False) -> eighresult:
    """
    Array API compatible wrapper for :py:func:`np.eig <numpy.eigh>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.eigh.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in eigh')

    # Note: the return type here is a namedtuple, which is different from
    # np.eigh, which only returns a tuple.
    return eighresult(*map(ndarray._new, np.linalg.eigh(x._array, UPLO='U' if upper else 'L')))

# def eigvals():
#     """
#     Array API compatible wrapper for :py:func:`np.eigvalsh <numpy.eigvals>`.
#
#     See its docstring for more information.
#     """
#     return np.eigvalh()

# Note: the keyword argument name upper is different from np.linalg.eigvalsh
def eigvalsh(x: array, /, *, upper: bool = False) -> array:
    """
    Array API compatible wrapper for :py:func:`np.eigvalsh <numpy.eigvalsh>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.eigvalsh.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in eigvalsh')

    return ndarray._new(np.linalg.eigvalsh(x._array, UPLO='U' if upper else 'L'))

# einsum is not yet implemented in the array API spec.

# def einsum():
#     """
#     Array API compatible wrapper for :py:func:`np.einsum <numpy.einsum>`.
#
#     See its docstring for more information.
#     """
#     return np.einsum()

def inv(x: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.inv <numpy.linalg.inv>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.inv.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in inv')

    return ndarray._new(np.linalg.inv(x._array))

# Note: the keyword argument name rtol is different from np.linalg.lstsq
def lstsq(x1: array, x2: array, /, *, rtol: Optional[Union[float, array]] = None) -> lstsqresult:
    """
    Array API compatible wrapper for :py:func:`np.lstsq <numpy.lstsq>`.

    See its docstring for more information.
    """
    # Note: lstsq is supposed to support stacks of matrices, but
    # np.linalg.lstsq does not yet.

    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.lstsq.
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in lstsq')

    # Note: the default value of rtol is the same as the post-deprecation
    # behavior for np.lstsq() (max(M, N)*eps).

    # Note: the return type here is a namedtuple, which is different from
    # np.lstsq, which only returns a tuple.
    return lstsqresult(*map(ndarray._new, np.linalg.lstsq(x1._array, x2._array, rcond=rtol)))

def matmul(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.matmul <numpy.matmul>`.

    See its docstring for more information.
    """
    # Note: the restriction to numeric dtypes only is different from
    # np.matmul.
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in matmul')

    return ndarray._new(np.matmul(x1._array, x2._array))

def matrix_power(x: array, n: int, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.matrix_power <numpy.matrix_power>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.matrix_power.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in matrix_power')

    # np.matrix_power already checks if n is an integer
    return ndarray._new(np.matrix_power(x._array, n))

# Note: the keyword argument name rtol is different from np.linalg.matrix_rank
def matrix_rank(x: array, /, *, rtol: Optional[Union[float, array]] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.matrix_rank <numpy.matrix_rank>`.

    See its docstring for more information.
    """
    # Note: this is different from np.linalg.matrix_rank, which supports 1
    # dimensional arrays.
    if x.ndim < 2:
        raise np.linalg.LinAlgError("1-dimensional array given. Array must be at least two-dimensional")
    S = np.linalg.svd(x._array, compute_uv=False)
    if rtol is None:
        tol = S.max(axis=-1, keepdims=True) * max(x.shape[-2:]) * np.finfo(S.dtype).eps
    else:
        # Note: this is different from np.linalg.matrix_rank, which does not multiply
        # the tolerance by the largest singular value.
        tol = S.max(axis=-1, keepdims=True)*np.asarray(tol)[..., np.newaxis]
    return ndarray._new(np.count_nonzero(S > tol, axis=-1))

def norm(x: array, /, *, axis: Optional[Union[int, Tuple[int, int]]] = None, keepdims: bool = False, ord: Optional[Union[int, float, Literal[np.inf, -np.inf, 'fro', 'nuc']]] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.norm <numpy.linalg.norm>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.norm.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in norm')

    return ndarray._new(np.linalg.norm(x._array, axis=axis, keepdims=keepdims, ord=ord))

def outer(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.outer <numpy.outer>`.

    See its docstring for more information.
    """
    # Note: the restriction to numeric dtypes only is different from
    # np.outer.
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in outer')

    return ndarray._new(np.outer(x1._array, x2._array))

# Note: the keyword argument name rtol is different from np.linalg.pinv
def pinv(x: array, /, *, rtol: Optional[Union[float, array]] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.pinv <numpy.pinv>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.pinv.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in pinv')

    # Note: this is different from np.linalg.pinv, which does not multiply the
    # default tolerance by max(M, N).
    if rtol is None:
        rtol = max(x.shape[-2:]) * np.finfo(x.dtype).eps
    return ndarray._new(np.pinv(x._array, rcond=rtol))

def qr(x: array, /, *, mode: str = 'reduced') -> qrresult:
    """
    Array API compatible wrapper for :py:func:`np.qr <numpy.qr>`.

    See its docstring for more information.
    """
    # Note: qr is supposed to support stacks of matrices, but
    # np.linalg.qr does not yet.

    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.qr.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in qr')

    # Note: the return type here is a namedtuple, which is different from
    # np.linalg.qr, which only returns a tuple.
    return qrresult(*map(ndarray._new, np.qr(x._array, mode=mode)))

def slogdet(x: array, /) -> slogdetresult:
    """
    Array API compatible wrapper for :py:func:`np.slogdet <numpy.slogdet>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.slogdet.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in slogdet')

    # Note: the return type here is a namedtuple, which is different from
    # np.linalg.slogdet, which only returns a tuple.
    return slogdetresult(*map(ndarray._new, np.slogdet(x._array)))

def solve(x1: array, x2: array, /) -> array:
    """
    Array API compatible wrapper for :py:func:`np.solve <numpy.solve>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.solve.
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in solve')

    return ndarray._new(np.solve(x1._array, x2._array))

def svd(x: array, /, *, full_matrices: bool = True) -> svdresult:
    """
    Array API compatible wrapper for :py:func:`np.svd <numpy.svd>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.svd.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in svd')

    # Note: the return type here is a namedtuple, which is different from
    # np.svd, which only returns a tuple.
    return svdresult(*map(ndarray._new, np.svd(x._array, full_matrices=full_matrices)))

# Note: svdvals is not in NumPy (but it is in SciPy). It is equivalent to
# np.linalg.svd(compute_uv=False).
def svdvals(x: array, /) -> Union[array, Tuple[array, ...]]:
    return ndarray._new(np.linalg.svd(x._array, compute_uv=False))

# Note: axes must be a tuple, unlike np.tensordot where it can be an array.
def tensordot(x1: array, x2: array, /, *, axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2) -> array:
    # Note: the restriction to numeric dtypes only is different from
    # np.tensordot.
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in tensordot')

    return ndarray._new(np.tensordot(x1._array, x2._array, axes=axes))

def trace(x: array, /, *, axis1: int = 0, axis2: int = 1, offset: int = 0) -> array:
    """
    Array API compatible wrapper for :py:func:`np.trace <numpy.trace>`.

    See its docstring for more information.
    """
    return ndarray._new(np.asarray(np.trace(x._array, axis1=axis1, axis2=axis2, offset=offset)))

def transpose(x: array, /, *, axes: Optional[Tuple[int, ...]] = None) -> array:
    """
    Array API compatible wrapper for :py:func:`np.transpose <numpy.transpose>`.

    See its docstring for more information.
    """
    return ndarray._new(np.transpose(x._array, axes=axes))

# Note: vecdot is not in NumPy
def vecdot(x1: array, x2: array, /, *, axis: Optional[int] = None) -> array:
    if axis is None:
        axis = -1
    return tensordot(x1, x2, (axis, axis))

__all__ = ['cholesky', 'cross', 'det', 'diagonal', 'eigh', 'eigvalsh', 'inv', 'lstsq', 'matmul', 'matrix_power', 'matrix_rank', 'norm', 'outer', 'pinv', 'qr', 'slogdet', 'solve', 'svd', 'tensordot', 'svdvals', 'trace', 'transpose', 'vecdot']

# These functions are not yet specified in the spec. eig() and eigvals()
# require complex dtype support, so will not be included until version 2 of
# the spec.

# __all__ = ['eig', 'eigvals', 'einsum']
