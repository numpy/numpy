""" Basic functions for manipulating 2d arrays

"""

__all__ = ['diag','diagflat','eye','fliplr','flipud','rot90','tri','triu',
           'tril','vander','histogram2d']

from numpy.core.numeric import asanyarray, equal, subtract, arange, \
     zeros, arange, greater_equal, multiply, ones, asarray

def fliplr(m):
    """
    Left-right flip.

    Flip the entries in each row in the left/right direction.
    Columns are preserved, but appear in a different order than before.

    Parameters
    ----------
    m : array_like
        Input array.

    Returns
    -------
    f : ndarray
        A view of `m` with the columns reversed.  Since a view
        is returned, this operation is :math:`\\mathcal O(1)`.

    See Also
    --------
    flipud : Flip array in the up/down direction.
    rot90 : Rotate array counterclockwise.

    Notes
    -----
    Equivalent to A[::-1,...]. Does not require the array to be
    two-dimensional.

    Examples
    --------
    >>> A = np.diag([1.,2.,3.])
    >>> A
    array([[ 1.,  0.,  0.],
           [ 0.,  2.,  0.],
           [ 0.,  0.,  3.]])
    >>> np.fliplr(A)
    array([[ 0.,  0.,  1.],
           [ 0.,  2.,  0.],
           [ 3.,  0.,  0.]])

    >>> A = np.random.randn(2,3,5)
    >>> np.all(numpy.fliplr(A)==A[:,::-1,...])
    True

    """
    m = asanyarray(m)
    if m.ndim < 2:
        raise ValueError, "Input must be >= 2-d."
    return m[:, ::-1]

def flipud(m):
    """
    Up-down flip.

    Flip the entries in each column in the up/down direction.
    Rows are preserved, but appear in a different order than before.

    Parameters
    ----------
    m : array_like
        Input array.

    Returns
    -------
    out : array_like
        A view of `m` with the rows reversed.  Since a view is
        returned, this operation is :math:`\\mathcal O(1)`.

    Notes
    -----
    Equivalent to ``A[::-1,...]``.
    Does not require the array to be two-dimensional.

    Examples
    --------
    >>> A = np.diag([1.0, 2, 3])
    >>> A
    array([[ 1.,  0.,  0.],
           [ 0.,  2.,  0.],
           [ 0.,  0.,  3.]])
    >>> np.flipud(A)
    array([[ 0.,  0.,  3.],
           [ 0.,  2.,  0.],
           [ 1.,  0.,  0.]])

    >>> A = np.random.randn(2,3,5)
    >>> np.all(np.flipud(A)==A[::-1,...])
    True

    >>> np.flipud([1,2])
    array([2, 1])

    """
    m = asanyarray(m)
    if m.ndim < 1:
        raise ValueError, "Input must be >= 1-d."
    return m[::-1,...]

def rot90(m, k=1):
    """
    Rotate an array by 90 degrees in the counter-clockwise direction.

    The first two dimensions are rotated; therefore, the array must be at
    least 2-D.

    Parameters
    ----------
    m : array_like
        Array of two or more dimensions.
    k : integer
        Number of times the array is rotated by 90 degrees.

    Returns
    -------
    y : ndarray
        Rotated array.

    See Also
    --------
    fliplr : Flip an array horizontally.
    flipud : Flip an array vertically.

    Examples
    --------
    >>> m = np.array([[1,2],[3,4]], int)
    >>> m
    array([[1, 2],
           [3, 4]])
    >>> np.rot90(m)
    array([[2, 4],
           [1, 3]])
    >>> np.rot90(m, 2)
    array([[4, 3],
           [2, 1]])

    """
    m = asanyarray(m)
    if m.ndim < 2:
        raise ValueError, "Input must >= 2-d."
    k = k % 4
    if k == 0: return m
    elif k == 1: return fliplr(m).swapaxes(0,1)
    elif k == 2: return fliplr(flipud(m))
    else: return fliplr(m.swapaxes(0,1))  # k==3

def eye(N, M=None, k=0, dtype=float):
    """
    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
      Number of rows in the output.
    M : int, optional
      Number of columns in the output. If None, defaults to `N`.
    k : int, optional
      Index of the diagonal: 0 refers to the main diagonal, a positive value
      refers to an upper diagonal and a negative value to a lower diagonal.
    dtype : dtype, optional
      Data-type of the returned array.

    Returns
    -------
    I : ndarray (N,M)
      An array where all elements are equal to zero, except for the k'th
      diagonal, whose values are equal to one.

    See Also
    --------
    diag : Return a diagonal 2-D array using a 1-D array specified by the user.

    Examples
    --------
    >>> np.eye(2, dtype=int)
    array([[1, 0],
           [0, 1]])
    >>> np.eye(3, k=1)
    array([[ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  0.,  0.]])

    """
    if M is None: M = N
    m = equal(subtract.outer(arange(N), arange(M)),-k)
    if m.dtype != dtype:
        m = m.astype(dtype)
    return m

def diag(v, k=0):
    """
    Extract a diagonal or construct a diagonal array.

    Parameters
    ----------
    v : array_like
        If `v` is a 2-dimensional array, return a copy of
        its `k`-th diagonal. If `v` is a 1-dimensional array,
        return a 2-dimensional array with `v` on the `k`-th diagonal.
    k : int, optional
        Diagonal in question.  The defaults is 0.

    Examples
    --------
    >>> x = np.arange(9).reshape((3,3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    >>> np.diag(x)
    array([0, 4, 8])

    >>> np.diag(np.diag(x))
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 8]])

    """
    v = asarray(v)
    s = v.shape
    if len(s)==1:
        n = s[0]+abs(k)
        res = zeros((n,n), v.dtype)
        if (k>=0):
            i = arange(0,n-k)
            fi = i+k+i*n
        else:
            i = arange(0,n+k)
            fi = i+(i-k)*n
        res.flat[fi] = v
        return res
    elif len(s)==2:
        N1,N2 = s
        if k >= 0:
            M = min(N1,N2-k)
            i = arange(0,M)
            fi = i+k+i*N2
        else:
            M = min(N1+k,N2)
            i = arange(0,M)
            fi = i + (i-k)*N2
        return v.flat[fi]
    else:
        raise ValueError, "Input must be 1- or 2-d."

def diagflat(v,k=0):
    """
    Create a 2-dimensional array with the flattened input as a diagonal.

    Parameters
    ----------
    v : array_like
        Input data, which is flattened and set as the `k`-th
        diagonal of the output.
    k : int, optional
        Diagonal to set.  The default is 0.

    Examples
    --------
    >>> np.diagflat([[1,2],[3,4]])
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])

    >>> np.diagflat([1,2], 1)
    array([[0, 1, 0],
           [0, 0, 2],
           [0, 0, 0]])

    """
    try:
        wrap = v.__array_wrap__
    except AttributeError:
        wrap = None
    v = asarray(v).ravel()
    s = len(v)
    n = s + abs(k)
    res = zeros((n,n), v.dtype)
    if (k>=0):
        i = arange(0,n-k)
        fi = i+k+i*n
    else:
        i = arange(0,n+k)
        fi = i+(i-k)*n
    res.flat[fi] = v
    if not wrap:
        return res
    return wrap(res)

def tri(N, M=None, k=0, dtype=float):
    """
    Construct an array filled with ones at and below the given diagonal.

    Parameters
    ----------
    N : int
        Number of rows in the array.
    M : int, optional
        Number of columns in the array.
        By default, `M` is taken to equal to `N`.
    k : int, optional
        The sub-diagonal below which the array is filled.
        ``k = 0`` is the main diagonal, while ``k < 0`` is below it,
        and ``k > 0`` is above.  The default is 0.
    dtype : dtype, optional
        Data type of the returned array.  The default is `float`.

    Returns
    -------
    T : (N,M) ndarray
        Array with a lower triangle filled with ones, in other words
        ``T[i,j] == 1`` for ``i <= j + k``.

    Examples
    --------
    >>> np.tri(3, 5, 2, dtype=int)
    array([[1, 1, 1, 0, 0],
           [1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1]])

    >>> np.tri(3, 5, -1)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  0.,  0.,  0.]])

    """
    if M is None: M = N
    m = greater_equal(subtract.outer(arange(N), arange(M)),-k)
    return m.astype(dtype)

def tril(m, k=0):
    """
    Lower triangular.

    Return a copy of an array with elements above the k-th diagonal zeroed.

    Parameters
    ----------
    m : array-like, shape (M, N)
        Input array.
    k : int
        Diagonal above which to zero elements.
        `k = 0` is the main diagonal, `k < 0` is below it and `k > 0` is above.

    Returns
    -------
    L : ndarray, shape (M, N)
        Lower triangle of `m`, of same shape and data-type as `m`.

    See Also
    --------
    triu

    Examples
    --------
    >>> np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
    array([[ 0,  0,  0],
           [ 4,  0,  0],
           [ 7,  8,  0],
           [10, 11, 12]])

    """
    m = asanyarray(m)
    out = multiply(tri(m.shape[0], m.shape[1], k=k, dtype=int),m)
    return out

def triu(m, k=0):
    """
    Upper triangular.

    Construct a copy of a matrix with elements below the k-th diagonal zeroed.

    Please refer to the documentation for `tril`.

    See Also
    --------
    tril

    Examples
    --------
    >>> np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 0,  8,  9],
           [ 0,  0, 12]])

    """
    m = asanyarray(m)
    out = multiply((1-tri(m.shape[0], m.shape[1], k-1, int)),m)
    return out

# borrowed from John Hunter and matplotlib
def vander(x, N=None):
    """
    Generate a Van der Monde matrix.

    The columns of the output matrix are decreasing powers of the input
    vector.  Specifically, the i-th output column is the input vector to
    the power of ``N - i - 1``.

    Parameters
    ----------
    x : array_like
        Input array.
    N : int, optional
        Order of (number of columns in) the output.

    Returns
    -------
    out : ndarray
        Van der Monde matrix of order `N`.  The first column is ``x^(N-1)``,
        the second ``x^(N-2)`` and so forth.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 5])
    >>> N = 3
    >>> np.vander(x, N)
    array([[ 1,  1,  1],
           [ 4,  2,  1],
           [ 9,  3,  1],
           [25,  5,  1]])

    >>> np.column_stack([x**(N-1-i) for i in range(N)])
    array([[ 1,  1,  1],
           [ 4,  2,  1],
           [ 9,  3,  1],
           [25,  5,  1]])

    """
    x = asarray(x)
    if N is None: N=len(x)
    X = ones( (len(x),N), x.dtype)
    for i in range(N-1):
        X[:,i] = x**(N-i-1)
    return X


def histogram2d(x,y, bins=10, range=None, normed=False, weights=None):
    """
    Compute the bidimensional histogram of two data samples.

    Parameters
    ----------
    x : array-like (N,)
      A sequence of values to be histogrammed along the first dimension.
    y : array-like (N,)
      A sequence of values to be histogrammed along the second dimension.
    bins : int or [int, int] or array-like or [array, array], optional
      The bin specification:

        * the number of bins for the two dimensions (nx=ny=bins),
        * the number of bins in each dimension (nx, ny = bins),
        * the bin edges for the two dimensions (x_edges=y_edges=bins),
        * the bin edges in each dimension (x_edges, y_edges = bins).

    range : array-like, (2,2), optional
      The leftmost and rightmost edges of the bins along each dimension
      (if not specified explicitly in the `bins` parameters):
      [[xmin, xmax], [ymin, ymax]]. All values outside of this range will be
      considered outliers and not tallied in the histogram.
    normed : boolean, optional
      If False, returns the number of samples in each bin. If True, returns
      the bin density, ie, the bin count divided by the bin area.
    weights : array-like (N,), optional
      An array of values `w_i` weighing each sample `(x_i, y_i)`. Weights are
      normalized to 1 if normed is True. If normed is False, the values of the
      returned histogram are equal to the sum of the weights belonging to the
      samples falling into each bin.

    Returns
    -------
    H : array (nx, ny)
      The bidimensional histogram of samples x and y. Values in x are
      histogrammed along the first dimension and values in y are histogrammed
      along the second dimension.
    xedges : array (nx,)
      The bin edges along the first dimension.
    yedges : array (ny,)
      The bin edges along the second dimension.

    See Also
    --------
    histogram: 1D histogram
    histogramdd: Multidimensional histogram

    Notes
    -----
    When normed is True, then the returned histogram is the sample density,
    defined such that:

      .. math::
        \\sum_{i=0}^{nx-1} \\sum_{j=0}^{ny-1} H_{i,j} \\Delta x_i \\Delta y_j = 1

    where :math:`H` is the histogram array and :math:`\\Delta x_i \\Delta y_i`
    the area of bin :math:`{i,j}`.

    Please note that the histogram does not follow the cartesian convention
    where `x` values are on the abcissa and `y` values on the ordinate axis.
    Rather, `x` is histogrammed along the first dimension of the array
    (vertical), and `y` along the second dimension of the array (horizontal).
    This ensures compatibility with `histogrammdd`.

    Examples
    --------
    >>> x,y = np.random.randn(2,100)
    >>> H, xedges, yedges = np.histogram2d(x, y, bins = (5, 8))
    >>> H.shape, xedges.shape, yedges.shape
    ((5,8), (6,), (9,))

    """
    from numpy import histogramdd

    try:
        N = len(bins)
    except TypeError:
        N = 1

    if N != 1 and N != 2:
        xedges = yedges = asarray(bins, float)
        bins = [xedges, yedges]
    hist, edges = histogramdd([x,y], bins, range, normed, weights)
    return hist, edges[0], edges[1]
