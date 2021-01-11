# def cholesky():
#     from .. import cholesky
#     return cholesky()

def cross(x1, x2, /, *, axis=-1):
    from .. import cross
    return cross(x1, x2, axis=axis)

def det(x, /):
    # Note: this function is being imported from a nondefault namespace
    from ..linalg import det
    return det(x)

def diagonal(x, /, *, axis1=0, axis2=1, offset=0):
    from .. import diagonal
    return diagonal(x, axis1=axis1, axis2=axis2, offset=offset)

# def dot():
#     from .. import dot
#     return dot()
#
# def eig():
#     from .. import eig
#     return eig()
#
# def eigvalsh():
#     from .. import eigvalsh
#     return eigvalsh()
#
# def einsum():
#     from .. import einsum
#     return einsum()

def inv(x):
    # Note: this function is being imported from a nondefault namespace
    from ..linalg import inv
    return inv(x)

# def lstsq():
#     from .. import lstsq
#     return lstsq()
#
# def matmul():
#     from .. import matmul
#     return matmul()
#
# def matrix_power():
#     from .. import matrix_power
#     return matrix_power()
#
# def matrix_rank():
#     from .. import matrix_rank
#     return matrix_rank()

def norm(x, /, *, axis=None, keepdims=False, ord=None):
    # Note: this function is being imported from a nondefault namespace
    from ..linalg import norm
    return norm(x, axis=axis, keepdims=keepdims, ord=ord)

def outer(x1, x2, /):
    from .. import outer
    return outer(x1, x2)

# def pinv():
#     from .. import pinv
#     return pinv()
#
# def qr():
#     from .. import qr
#     return qr()
#
# def slogdet():
#     from .. import slogdet
#     return slogdet()
#
# def solve():
#     from .. import solve
#     return solve()
#
# def svd():
#     from .. import svd
#     return svd()

def trace(x, /, *, axis1=0, axis2=1, offset=0):
    from .. import trace
    return trace(x, axis1=axis1, axis2=axis2, offset=offset)

def transpose(x, /, *, axes=None):
    from .. import transpose
    return transpose(x, axes=axes)

# __all__ = ['cholesky', 'cross', 'det', 'diagonal', 'dot', 'eig', 'eigvalsh', 'einsum', 'inv', 'lstsq', 'matmul', 'matrix_power', 'matrix_rank', 'norm', 'outer', 'pinv', 'qr', 'slogdet', 'solve', 'svd', 'trace', 'transpose']

__all__ = ['cross', 'det', 'diagonal', 'inv', 'norm', 'outer', 'trace', 'transpose']
