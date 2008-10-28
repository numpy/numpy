import numpy as np
from numpy.core.defmatrix import matrix, asmatrix
# need * as we're copying the numpy namespace
from numpy import *

__version__ = np.__version__

__all__ = np.__all__[:] # copy numpy namespace
__all__ += ['rand', 'randn', 'repmat']

def empty(shape, dtype=None, order='C'):
    """return an empty matrix of the given shape
    """
    return ndarray.__new__(matrix, shape, dtype, order=order)

def ones(shape, dtype=None, order='C'):
    """
    Matrix of ones.

    Return a matrix of given shape and type, filled with ones.

    Parameters
    ----------
    shape : {sequence of ints, int}
        Shape of the matrix
    dtype : data-type, optional
        The desired data-type for the matrix, default is np.float64.
    order : {'C', 'F'}, optional
        Whether to store matrix in C- or Fortran-contiguous order,
        default is 'C'.

    Returns
    -------
    out : matrix
        Matrix of ones of given shape, dtype, and order.

    See Also
    --------
    ones : Array of ones.
    matlib.zeros : Zero matrix.

    Notes
    -----
    If `shape` has length one i.e. ``(N,)``, or is a scalar ``N``,
    `out` becomes a single row matrix of shape ``(1,N)``.

    Examples
    --------
    >>> np.matlib.ones((2,3))
    matrix([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])

    >>> np.matlib.ones(2)
    matrix([[ 1.,  1.]]

    """
    a = ndarray.__new__(matrix, shape, dtype, order=order)
    a.fill(1)
    return a

def zeros(shape, dtype=None, order='C'):
    """
    Zero matrix.

    Return a matrix of given shape and type, filled with zeros

    Parameters
    ----------
    shape : {sequence of ints, int}
        Shape of the matrix
    dtype : data-type, optional
        The desired data-type for the matrix, default is np.float64.
    order : {'C', 'F'}, optional
        Whether to store the result in C- or Fortran-contiguous order,
        default is 'C'.

    Returns
    -------
    out : matrix
        Zero matrix of given shape, dtype, and order.

    See Also
    --------
    zeros : Zero array.
    matlib.ones : Matrix of ones.

    Notes
    -----
    If `shape` has length one i.e. ``(N,)``, or is a scalar ``N``,
    `out` becomes a single row matrix of shape ``(1,N)``.

    Examples
    --------
    >>> np.matlib.zeros((2,3))
    matrix([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])

    >>> np.matlib.zeros(2)
    matrix([[ 0.,  0.]]

    """
    a = ndarray.__new__(matrix, shape, dtype, order=order)
    a.fill(0)
    return a

def identity(n,dtype=None):
    """
    Returns the square identity matrix of given size.

    Parameters
    ----------
    n : int
        Size of identity matrix

    dtype : data-type, optional
        Data-type of the output. Defaults to ``float``.

    Returns
    -------
    out : matrix
        `n` x `n` matrix with its main diagonal set to one,
        and all other elements zero.

    See Also
    --------
    identity : Equivalent array function.
    matlib.eye : More general matrix identity function.

    Notes
    -----
    For more detailed documentation, see the docstring of the equivalent
    array function ``np.identity``

    """
    a = array([1]+n*[0],dtype=dtype)
    b = empty((n,n),dtype=dtype)
    b.flat = a
    return b

def eye(n,M=None, k=0, dtype=float):
    """
    Return a matrix with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    n : int
        Number of rows in the output.
    M : int, optional
        Number of columns in the output, defaults to n.
    k : int, optional
        Index of the diagonal: 0 refers to the main diagonal,
        a positive value refers to an upper diagonal,
        and a negative value to a lower diagonal.
    dtype : dtype, optional
        Data-type of the returned matrix.

    Returns
    -------
    I : matrix
        A `n` x `M` matrix where all elements are equal to zero,
        except for the k-th diagonal, whose values are equal to one.

    See Also
    --------
    eye : Equivalent array function
    matlib.identity : Square identity matrix

    Notes
    -----
    For more detailed docuemtation, see the docstring of the equivalent
    array function ``np.eye``.

    """
    return asmatrix(np.eye(n,M,k,dtype))

def rand(*args):
    if isinstance(args[0], tuple):
        args = args[0]
    return asmatrix(np.random.rand(*args))

def randn(*args):
    if isinstance(args[0], tuple):
        args = args[0]
    return asmatrix(np.random.randn(*args))

def repmat(a, m, n):
    """Repeat a 0-d to 2-d array mxn times
    """
    a = asanyarray(a)
    ndim = a.ndim
    if ndim == 0:
        origrows, origcols = (1,1)
    elif ndim == 1:
        origrows, origcols = (1, a.shape[0])
    else:
        origrows, origcols = a.shape
    rows = origrows * m
    cols = origcols * n
    c = a.reshape(1,a.size).repeat(m, 0).reshape(rows, origcols).repeat(n,0)
    return c.reshape(rows, cols)
