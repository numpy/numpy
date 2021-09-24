from __future__ import annotations

from ._dtypes import _floating_dtypes, _numeric_dtypes
from ._array_object import Array

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._typing import Literal, Optional, Sequence, Tuple, Union

from typing import NamedTuple

import numpy.linalg
import numpy as np

class eighresult(NamedTuple):
    u: Array
    v: Array

class lstsqresult(NamedTuple):
    x: Array
    residuals: Array
    rank: Array
    s: Array

class qrresult(NamedTuple):
    q: Array
    r: Array

class slogdetresult(NamedTuple):
    sign: Array
    logabsdet: Array

class svdresult(NamedTuple):
    u: Array
    s: Array
    v: Array

# Note: the inclusion of the upper keyword is different from
# np.linalg.cholesky, which does not have it.
def cholesky(x: Array, /, *, upper: bool = False) -> Array:
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

def cross(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.cross <numpy.cross>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in cross')
    # Note: this is different from np.cross(), which broadcasts
    if x1.shape != x2.shape:
        raise ValueError('x1 and x2 must have the same shape')
    if x1.ndim == 0:
        raise ValueError('cross() requires arrays of dimension at least 1')
    # Note: this is different from np.cross(), which allows dimension 2
    if x1.shape[axis] != 3:
        raise ValueError('cross() dimension must equal 3')
    return ndarray._new(np.cross(x1._array, x2._array, axis=axis))

def det(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.det <numpy.linalg.det>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.det.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in det')
    return ndarray._new(np.linalg.det(x._array))

def diagonal(x: Array, /, *, axis1: int = 0, axis2: int = 1, offset: int = 0) -> Array:
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
def eigh(x: Array, /, *, upper: bool = False) -> eighresult:
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
def eigvalsh(x: Array, /, *, upper: bool = False) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.eigvalsh <numpy.eigvalsh>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.eigvalsh.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in eigvalsh')

    return ndarray._new(np.linalg.eigvalsh(x._array, UPLO='U' if upper else 'L'))

def inv(x: Array, /) -> Array:
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
def lstsq(x1: Array, x2: Array, /, *, rtol: Optional[Union[float, Array]] = None) -> lstsqresult:
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

def matmul(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.matmul <numpy.matmul>`.

    See its docstring for more information.
    """
    # Note: the restriction to numeric dtypes only is different from
    # np.matmul.
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in matmul')

    return ndarray._new(np.matmul(x1._array, x2._array))

def matrix_power(x: Array, n: int, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.matrix_power <numpy.matrix_power>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.matrix_power.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed for the first argument of matrix_power')

    # np.matrix_power already checks if n is an integer
    return ndarray._new(np.linalg.matrix_power(x._array, n))

# Note: the keyword argument name rtol is different from np.linalg.matrix_rank
def matrix_rank(x: Array, /, *, rtol: Optional[Union[float, Array]] = None) -> Array:
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
        if isinstance(rtol, ndarray):
            rtol = rtol._array
        # Note: this is different from np.linalg.matrix_rank, which does not multiply
        # the tolerance by the largest singular value.
        tol = S.max(axis=-1, keepdims=True)*np.asarray(rtol)[..., np.newaxis]
    return ndarray._new(np.count_nonzero(S > tol, axis=-1))

# Note: this function is new in the array API spec. Unlike transpose, it only
# transposes the last two axes.
def matrix_transpose(x: Array, /) -> Array:
    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for matrix_transpose")
    return Array._new(np.swapaxes(x._array, -1, -2))

def norm(x: Array, /, *, axis: Optional[Union[int, Tuple[int, int]]] = None, keepdims: bool = False, ord: Optional[Union[int, float, Literal[np.inf, -np.inf, 'fro', 'nuc']]] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.norm <numpy.linalg.norm>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.norm.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in norm')

    return ndarray._new(np.linalg.norm(x._array, axis=axis, keepdims=keepdims, ord=ord))

def outer(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.outer <numpy.outer>`.

    See its docstring for more information.
    """
    # Note: the restriction to numeric dtypes only is different from
    # np.outer.
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in outer')

    # Note: the restriction to only 1-dim arrays is different from np.outer
    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError('The input arrays to outer must be 1-dimensional')

    return ndarray._new(np.outer(x1._array, x2._array))

# Note: the keyword argument name rtol is different from np.linalg.pinv
def pinv(x: Array, /, *, rtol: Optional[Union[float, Array]] = None) -> Array:
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
    return ndarray._new(np.linalg.pinv(x._array, rcond=rtol))

def qr(x: Array, /, *, mode: str = 'reduced') -> qrresult:
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
    return qrresult(*map(ndarray._new, np.linalg.qr(x._array, mode=mode)))

def slogdet(x: Array, /) -> slogdetresult:
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
    return slogdetresult(*map(ndarray._new, np.linalg.slogdet(x._array)))

def solve(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.solve <numpy.solve>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.solve.
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in solve')

    return ndarray._new(np.linalg.solve(x1._array, x2._array))

def svd(x: Array, /, *, full_matrices: bool = True) -> svdresult:
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
    return svdresult(*map(ndarray._new, np.linalg.svd(x._array, full_matrices=full_matrices)))

# Note: svdvals is not in NumPy (but it is in SciPy). It is equivalent to
# np.linalg.svd(compute_uv=False).
def svdvals(x: Array, /) -> Union[Array, Tuple[Array, ...]]:
    return ndarray._new(np.linalg.svd(x._array, compute_uv=False))

# Note: axes must be a tuple, unlike np.tensordot where it can be an array or array-like.
def tensordot(x1: Array, x2: Array, /, *, axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2) -> Array:
    # Note: the restriction to numeric dtypes only is different from
    # np.tensordot.
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in tensordot')

    return ndarray._new(np.tensordot(x1._array, x2._array, axes=axes))

def trace(x: Array, /, *, axis1: int = 0, axis2: int = 1, offset: int = 0) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.trace <numpy.trace>`.

    See its docstring for more information.
    """
    return ndarray._new(np.asarray(np.trace(x._array, axis1=axis1, axis2=axis2, offset=offset)))

# Note: vecdot is not in NumPy
def vecdot(x1: Array, x2: Array, /, *, axis: Optional[int] = None) -> Array:
    if axis is None:
        axis = -1
    return tensordot(x1, x2, axes=((axis,), (axis,)))

__all__ = ['cholesky', 'cross', 'det', 'diagonal', 'eigh', 'eigvalsh', 'inv', 'lstsq', 'matmul', 'matrix_power', 'matrix_rank', 'matrix_transpose', 'norm', 'outer', 'pinv', 'qr', 'slogdet', 'solve', 'svd', 'tensordot', 'svdvals', 'trace', 'vecdot']

# These functions are not yet specified in the spec. eig() and eigvals()
# require complex dtype support, so will not be included until version 2 of
# the spec.

# __all__ = ['eig', 'eigvals']
