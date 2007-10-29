
__all__ = ['Matrix']

from numpy import matrix as _matrix

def Matrix(data, typecode=None, copy=1, savespace=0):
    return _matrix(data, typecode, copy=copy)
