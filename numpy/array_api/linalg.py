from __future__ import annotations

from ._dtypes import _floating_dtypes, _numeric_dtypes
from ._array_object import Array

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._typing import Literal, Optional, Sequence, Tuple, Union

from typing import NamedTuple

import numpy.linalg
import numpy as np

class EIGHResult(NamedTuple):
    eigenvalues: Array
    eigenvectors: Array

class QRResult(NamedTuple):
    q: Array
    r: Array

class SLOGDETResult(NamedTuple):
    sign: Array
    logabsdet: Array

class SVDResult(NamedTuple):
    u: Array
    s: Array
    vh: Array

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
        return Array._new(L).mT
    return Array._new(L)

# Note: cross is the numpy top-level namespace, not np.linalg
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
    return Array._new(np.cross(x1._array, x2._array, axis=axis))

def det(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.det <numpy.linalg.det>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.det.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in det')
    return Array._new(np.linalg.det(x._array))

# Note: diagonal is the numpy top-level namespace, not np.linalg
def diagonal(x: Array, /, *, offset: int = 0) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.diagonal <numpy.diagonal>`.

    See its docstring for more information.
    """
    return Array._new(np.diagonal(x._array, offset=offset))


# Note: the keyword argument name upper is different from np.linalg.eigh
def eigh(x: Array, /) -> EIGHResult:
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
    return EIGHResult(*map(Array._new, np.linalg.eigh(x._array)))


# Note: the keyword argument name upper is different from np.linalg.eigvalsh
def eigvalsh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.eigvalsh <numpy.eigvalsh>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.eigvalsh.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in eigvalsh')

    return Array._new(np.linalg.eigvalsh(x._array))

def inv(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.inv <numpy.linalg.inv>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.inv.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in inv')

    return Array._new(np.linalg.inv(x._array))


# Note: matmul is the numpy top-level namespace but not in np.linalg
def matmul(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.matmul <numpy.matmul>`.

    See its docstring for more information.
    """
    # Note: the restriction to numeric dtypes only is different from
    # np.matmul.
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in matmul')

    return Array._new(np.matmul(x1._array, x2._array))


# Note: the name here is different from norm(). The array API norm is split
# into matrix_norm and vector_norm().
def matrix_norm(x: Array, /, *, axis: Tuple[int, int] = (-2, -1), keepdims: bool = False, ord: Optional[Union[int, float, Literal[np.inf, -np.inf, 'fro', 'nuc']]] = 'fro') -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.norm <numpy.linalg.norm>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.norm.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in norm')

    if not isinstance(axis, tuple) or not len(axis) == 2:
        raise ValueError("axis must be a tuple of 2 integers. Use vector_norm() to compute a vector norm.")
    return Array._new(np.linalg.norm(x._array, axis=axis, keepdims=keepdims, ord=ord))


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
    return Array._new(np.linalg.matrix_power(x._array, n))

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
        if isinstance(rtol, Array):
            rtol = rtol._array
        # Note: this is different from np.linalg.matrix_rank, which does not multiply
        # the tolerance by the largest singular value.
        tol = S.max(axis=-1, keepdims=True)*np.asarray(rtol)[..., np.newaxis]
    return Array._new(np.count_nonzero(S > tol, axis=-1))


# Note: this function is new in the array API spec. Unlike transpose, it only
# transposes the last two axes.
def matrix_transpose(x: Array, /) -> Array:
    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for matrix_transpose")
    return Array._new(np.swapaxes(x._array, -1, -2))

# Note: outer is the numpy top-level namespace, not np.linalg
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

    return Array._new(np.outer(x1._array, x2._array))

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
    return Array._new(np.linalg.pinv(x._array, rcond=rtol))

def qr(x: Array, /, *, mode: Literal['reduced', 'complete'] = 'reduced') -> QRResult:
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
    return QRResult(*map(Array._new, np.linalg.qr(x._array, mode=mode)))

def slogdet(x: Array, /) -> SLOGDETResult:
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
    return SLOGDETResult(*map(Array._new, np.linalg.slogdet(x._array)))

def solve(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.solve <numpy.solve>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.solve.
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in solve')

    return Array._new(np.linalg.solve(x1._array, x2._array))

def svd(x: Array, /, *, full_matrices: bool = True) -> SVDResult:
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
    return SVDResult(*map(Array._new, np.linalg.svd(x._array, full_matrices=full_matrices)))

# Note: svdvals is not in NumPy (but it is in SciPy). It is equivalent to
# np.linalg.svd(compute_uv=False).
def svdvals(x: Array, /) -> Union[Array, Tuple[Array, ...]]:
    return Array._new(np.linalg.svd(x._array, compute_uv=False))

# Note: tensordot is the numpy top-level namespace but not in np.linalg

# Note: axes must be a tuple, unlike np.tensordot where it can be an array or array-like.
def tensordot(x1: Array, x2: Array, /, *, axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2) -> Array:
    # Note: the restriction to numeric dtypes only is different from
    # np.tensordot.
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in tensordot')

    return Array._new(np.tensordot(x1._array, x2._array, axes=axes))

# Note: trace is the numpy top-level namespace, not np.linalg
def trace(x: Array, /, *, offset: int = 0) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.trace <numpy.trace>`.

    See its docstring for more information.
    """
    return Array._new(np.asarray(np.trace(x._array, offset=offset)))

# Note: vecdot is not in NumPy
def vecdot(x1: Array, x2: Array, /, *, axis: Optional[int] = None) -> Array:
    if axis is None:
        axis = -1
    return tensordot(x1, x2, axes=((axis,), (axis,)))


# Note: the name here is different from norm(). The array API norm is split
# into matrix_norm and vector_norm().
def vector_norm(x: Array, /, *, axis: Optional[Union[int, Tuple[int, int]]] = None, keepdims: bool = False, ord: Optional[Union[int, float, Literal[np.inf, -np.inf]]] = 2) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.norm <numpy.linalg.norm>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.norm.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in norm')

    a = x._array
    if axis is None:
        a = a.flatten()
        axis = 0
    elif isinstance(axis, tuple):
        # Note: The axis argument supports any number of axes, whereas norm()
        # only supports a single axis for vector norm.
        rest = tuple(i for i in range(a.ndim) if i not in axis)
        newshape = axis + rest
        a = np.transpose(a, newshape).reshape((np.prod([a.shape[i] for i in axis]), *[a.shape[i] for i in rest]))
        axis = 0
    return Array._new(np.linalg.norm(a, axis=axis, keepdims=keepdims, ord=ord))

__all__ = ['cholesky', 'cross', 'det', 'diagonal', 'eigh', 'eigvalsh', 'inv', 'matmul', 'matrix_norm', 'matrix_power', 'matrix_rank', 'matrix_transpose', 'outer', 'pinv', 'qr', 'slogdet', 'solve', 'svd', 'svdvals', 'tensordot', 'trace', 'vecdot', 'vector_norm']
