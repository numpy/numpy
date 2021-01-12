import numpy as np

# def cholesky():
#     return np.cholesky()

def cross(x1, x2, /, *, axis=-1):
    return np.cross(x1, x2, axis=axis)

def det(x, /):
    # Note: this function is being imported from a nondefault namespace
    return np.linalg.det(x)

def diagonal(x, /, *, axis1=0, axis2=1, offset=0):
    return np.diagonal(x, axis1=axis1, axis2=axis2, offset=offset)

# def dot():
#     return np.dot()
#
# def eig():
#     return np.eig()
#
# def eigvalsh():
#     return np.eigvalsh()
#
# def einsum():
#     return np.einsum()

def inv(x):
    # Note: this function is being imported from a nondefault namespace
    return np.linalg.inv(x)

# def lstsq():
#     return np.lstsq()
#
# def matmul():
#     return np.matmul()
#
# def matrix_power():
#     return np.matrix_power()
#
# def matrix_rank():
#     return np.matrix_rank()

def norm(x, /, *, axis=None, keepdims=False, ord=None):
    # Note: this function is being imported from a nondefault namespace
    # Note: this is different from the default behavior
    if axis == None and x.ndim > 2:
        x = x.flatten()
    return np.linalg.norm(x, axis=axis, keepdims=keepdims, ord=ord)

def outer(x1, x2, /):
    return np.outer(x1, x2)

# def pinv():
#     return np.pinv()
#
# def qr():
#     return np.qr()
#
# def slogdet():
#     return np.slogdet()
#
# def solve():
#     return np.solve()
#
# def svd():
#     return np.svd()

def trace(x, /, *, axis1=0, axis2=1, offset=0):
    return np.trace(x, axis1=axis1, axis2=axis2, offset=offset)

def transpose(x, /, *, axes=None):
    return np.transpose(x, axes=axes)
