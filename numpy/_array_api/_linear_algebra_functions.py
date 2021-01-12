import numpy as np

# def cholesky():
#     """
#     Array API compatible wrapper for :py:func:`np.cholesky <numpy.cholesky>`.
#
#     See its docstring for more information.
#     """
#     return np.cholesky()

def cross(x1, x2, /, *, axis=-1):
    """
    Array API compatible wrapper for :py:func:`np.cross <numpy.cross>`.

    See its docstring for more information.
    """
    return np.cross(x1, x2, axis=axis)

def det(x, /):
    """
    Array API compatible wrapper for :py:func:`np.linalg.det <numpy.linalg.det>`.

    See its docstring for more information.
    """
    # Note: this function is being imported from a nondefault namespace
    return np.linalg.det(x)

def diagonal(x, /, *, axis1=0, axis2=1, offset=0):
    """
    Array API compatible wrapper for :py:func:`np.diagonal <numpy.diagonal>`.

    See its docstring for more information.
    """
    return np.diagonal(x, axis1=axis1, axis2=axis2, offset=offset)

# def dot():
#     """
#     Array API compatible wrapper for :py:func:`np.dot <numpy.dot>`.
#
#     See its docstring for more information.
#     """
#     return np.dot()
#
# def eig():
#     """
#     Array API compatible wrapper for :py:func:`np.eig <numpy.eig>`.
#
#     See its docstring for more information.
#     """
#     return np.eig()
#
# def eigvalsh():
#     """
#     Array API compatible wrapper for :py:func:`np.eigvalsh <numpy.eigvalsh>`.
#
#     See its docstring for more information.
#     """
#     return np.eigvalsh()
#
# def einsum():
#     """
#     Array API compatible wrapper for :py:func:`np.einsum <numpy.einsum>`.
#
#     See its docstring for more information.
#     """
#     return np.einsum()

def inv(x):
    """
    Array API compatible wrapper for :py:func:`np.linalg.inv <numpy.linalg.inv>`.

    See its docstring for more information.
    """
    # Note: this function is being imported from a nondefault namespace
    return np.linalg.inv(x)

# def lstsq():
#     """
#     Array API compatible wrapper for :py:func:`np.lstsq <numpy.lstsq>`.
#
#     See its docstring for more information.
#     """
#     return np.lstsq()
#
# def matmul():
#     """
#     Array API compatible wrapper for :py:func:`np.matmul <numpy.matmul>`.
#
#     See its docstring for more information.
#     """
#     return np.matmul()
#
# def matrix_power():
#     """
#     Array API compatible wrapper for :py:func:`np.matrix_power <numpy.matrix_power>`.
#
#     See its docstring for more information.
#     """
#     return np.matrix_power()
#
# def matrix_rank():
#     """
#     Array API compatible wrapper for :py:func:`np.matrix_rank <numpy.matrix_rank>`.
#
#     See its docstring for more information.
#     """
#     return np.matrix_rank()

def norm(x, /, *, axis=None, keepdims=False, ord=None):
    """
    Array API compatible wrapper for :py:func:`np.linalg.norm <numpy.linalg.norm>`.

    See its docstring for more information.
    """
    # Note: this is different from the default behavior
    if axis == None and x.ndim > 2:
        x = x.flatten()
    # Note: this function is being imported from a nondefault namespace
    return np.linalg.norm(x, axis=axis, keepdims=keepdims, ord=ord)

def outer(x1, x2, /):
    """
    Array API compatible wrapper for :py:func:`np.outer <numpy.outer>`.

    See its docstring for more information.
    """
    return np.outer(x1, x2)

# def pinv():
#     """
#     Array API compatible wrapper for :py:func:`np.pinv <numpy.pinv>`.
#
#     See its docstring for more information.
#     """
#     return np.pinv()
#
# def qr():
#     """
#     Array API compatible wrapper for :py:func:`np.qr <numpy.qr>`.
#
#     See its docstring for more information.
#     """
#     return np.qr()
#
# def slogdet():
#     """
#     Array API compatible wrapper for :py:func:`np.slogdet <numpy.slogdet>`.
#
#     See its docstring for more information.
#     """
#     return np.slogdet()
#
# def solve():
#     """
#     Array API compatible wrapper for :py:func:`np.solve <numpy.solve>`.
#
#     See its docstring for more information.
#     """
#     return np.solve()
#
# def svd():
#     """
#     Array API compatible wrapper for :py:func:`np.svd <numpy.svd>`.
#
#     See its docstring for more information.
#     """
#     return np.svd()

def trace(x, /, *, axis1=0, axis2=1, offset=0):
    """
    Array API compatible wrapper for :py:func:`np.trace <numpy.trace>`.

    See its docstring for more information.
    """
    return np.trace(x, axis1=axis1, axis2=axis2, offset=offset)

def transpose(x, /, *, axes=None):
    """
    Array API compatible wrapper for :py:func:`np.transpose <numpy.transpose>`.

    See its docstring for more information.
    """
    return np.transpose(x, axes=axes)
