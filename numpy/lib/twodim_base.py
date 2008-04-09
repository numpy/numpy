""" Basic functions for manipulating 2d arrays

"""

__all__ = ['diag','diagflat','eye','fliplr','flipud','rot90','tri','triu',
           'tril','vander','histogram2d']

from numpy.core.numeric import asanyarray, equal, subtract, arange, \
     zeros, arange, greater_equal, multiply, ones, asarray

def fliplr(m):
    """ returns an array m with the rows preserved and columns flipped
        in the left/right direction.  Works on the first two dimensions of m.
    """
    m = asanyarray(m)
    if m.ndim < 2:
        raise ValueError, "Input must be >= 2-d."
    return m[:, ::-1]

def flipud(m):
    """ returns an array with the columns preserved and rows flipped in
        the up/down direction.  Works on the first dimension of m.
    """
    m = asanyarray(m)
    if m.ndim < 1:
        raise ValueError, "Input must be >= 1-d."
    return m[::-1,...]

def rot90(m, k=1):
    """ returns the array found by rotating m by k*90
    degrees in the counterclockwise direction.  Works on the first two
    dimensions of m.
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
    """ eye returns a N-by-M 2-d array where the  k-th diagonal is all ones,
        and everything else is zeros.
    """
    if M is None: M = N
    m = equal(subtract.outer(arange(N), arange(M)),-k)
    if m.dtype != dtype:
        m = m.astype(dtype)
    return m

def diag(v, k=0):
    """ returns a copy of the the k-th diagonal if v is a 2-d array
        or returns a 2-d array with v as the k-th diagonal if v is a
        1-d array.
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
    """Return a 2D array whose k'th diagonal is a flattened v and all other 
    elements are zero. 
    
    Examples
    --------
      >>> diagflat([[1,2],[3,4]]])
      array([[1, 0, 0, 0],
             [0, 2, 0, 0],
             [0, 0, 3, 0],
             [0, 0, 0, 4]])
      
      >>> diagflat([1,2], 1)
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
    """ returns a N-by-M array where all the diagonals starting from
        lower left corner up to the k-th are all ones.
    """
    if M is None: M = N
    m = greater_equal(subtract.outer(arange(N), arange(M)),-k)
    return m.astype(dtype)

def tril(m, k=0):
    """ returns the elements on and below the k-th diagonal of m.  k=0 is the
        main diagonal, k > 0 is above and k < 0 is below the main diagonal.
    """
    m = asanyarray(m)
    out = multiply(tri(m.shape[0], m.shape[1], k=k, dtype=int),m)
    return out

def triu(m, k=0):
    """ returns the elements on and above the k-th diagonal of m.  k=0 is the
        main diagonal, k > 0 is above and k < 0 is below the main diagonal.
    """
    m = asanyarray(m)
    out = multiply((1-tri(m.shape[0], m.shape[1], k-1, int)),m)
    return out

# borrowed from John Hunter and matplotlib
def vander(x, N=None):
    """
    Generate the Vandermonde matrix of vector x.

    The i-th column of X is the the (N-i)-1-th power of x.  N is the
    maximum power to compute; if N is None it defaults to len(x).

    """
    x = asarray(x)
    if N is None: N=len(x)
    X = ones( (len(x),N), x.dtype)
    for i in range(N-1):
        X[:,i] = x**(N-i-1)
    return X


def histogram2d(x,y, bins=10, range=None, normed=False, weights=None):
    """histogram2d(x,y, bins=10, range=None, normed=False) -> H, xedges, yedges

    Compute the 2D histogram from samples x,y.

    :Parameters:
      - `x,y` : Sample arrays (1D).
      - `bins` : Number of bins -or- [nbin x, nbin y] -or-
             [bin edges] -or- [x bin edges, y bin edges].
      - `range` : A sequence of lower and upper bin edges (default: [min, max]).
      - `normed` : Boolean, if False, return the number of samples in each bin,
                if True, returns the density.
      - `weights` : An array of weights. The weights are normed only if normed
                is True. Should weights.sum() not equal N, the total bin count \
                will not be equal to the number of samples.

    :Return:
      - `hist` :    Histogram array.
      - `xedges, yedges` : Arrays defining the bin edges.

    Example:
      >>> x = random.randn(100,2)
      >>> hist2d, xedges, yedges = histogram2d(x, bins = (6, 7))

    :SeeAlso: histogramdd
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
