""" Basic functions for manipulating 2d arrays

"""

__all__ = ['diag','diagflat','eye','fliplr','flipud','rot90','tri','triu',
           'tril','vander','histogram2d']

from numpy.core.numeric import asanyarray, int_, equal, subtract, arange, \
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
    elif k == 1: return fliplr(m).transpose()
    elif k == 2: return fliplr(flipud(m))
    else: return fliplr(m.transpose())  # k==3

def eye(N, M=None, k=0, dtype=float):
    """ eye returns a N-by-M 2-d array where the  k-th diagonal is all ones,
        and everything else is zeros.
    """
    if M is None: M = N
    m = equal(subtract.outer(arange(N), arange(M)),-k)
    if m.dtype != dtype:
        return m.astype(dtype)

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
    if m.dtype != dtype:
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
    X = vander(x,N=None)

    The Vandermonde matrix of vector x.  The i-th column of X is the
    the i-th power of x.  N is the maximum power to compute; if N is
    None it defaults to len(x).

    """
    x = asarray(x)
    if N is None: N=len(x)
    X = ones( (len(x),N), x.dtype)
    for i in range(N-1):
        X[:,i] = x**(N-i-1)
    return X

def  histogram2d(x,y, bins=10, range=None, normed=False):
    """histogram2d(x,y, bins=10, range=None, normed=False) -> H, xedges, yedges
    
    Compute the 2D histogram from samples x,y. 

    Parameters
    ----------
    x,y: 1D data series. Both arrays must have the same length.
    bins: Number of bins -or- [nbin x, nbin y] -or- 
         [bin edges] -or- [x bin edges, y bin edges].
    range:  A sequence of lower and upper bin edges (default: [min, max]).
    normed: True or False. 
    
    The histogram array is a count of the number of samples in each 
    two dimensional bin. 
    Setting normed to True returns a density rather than a bin count. 
    Data falling outside of the edges are not counted.
    """
    import numpy as np
    try:
        N = len(bins)
    except TypeError:
        N = 1
        bins = [bins]
    x = asarray(x)
    y = asarray(y)
    if range is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax = range[0]
        ymin, ymax = range[1]
    if N == 2:
        if np.isscalar(bins[0]):
            xnbin = bins[0]
            xedges = np.linspace(xmin, xmax, xnbin+1)
        else:
            xedges = asarray(bins[0], float)
            xnbin = len(xedges)-1
        if np.isscalar(bins[1]):
            ynbin = bins[1]
            yedges = np.linspace(ymin, ymax, ynbin+1)
        else:
            yedges = asarray(bins[1], float)
            ynbin = len(yedges)-1
    elif N == 1:
        ynbin = xnbin = bins[0]
        xedges = np.linspace(xmin, xmax, xnbin+1)
        yedges = np.linspace(ymin, ymax, ynbin+1)
    else:
        yedges = asarray(bins, float)
        xedges = yedges.copy()
        ynbin = len(yedges)-1
        xnbin = len(xedges)-1
    
    dxedges = np.diff(xedges)
    dyedges = np.diff(yedges)
    
    # Flattened histogram matrix (1D)
    hist = np.zeros((xnbin)*(ynbin), int)

    # Count the number of sample in each bin (1D)
    xbin = np.digitize(x,xedges) 
    ybin = np.digitize(y,yedges) 
 
    # Values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right 
    # edge to be counted in the last bin, and not as an outlier.
    xdecimal = int(-np.log10(dxedges.min()))+6
    ydecimal = int(-np.log10(dyedges.min()))+6
    on_edge_x = np.where(np.around(x,xdecimal) == np.around(xedges[-1], xdecimal))[0]
    on_edge_y = np.where(np.around(y,ydecimal) == np.around(yedges[-1], ydecimal))[0]
    xbin[on_edge_x] -= 1
    ybin[on_edge_y] -= 1
    # Remove the true outliers
    outliers = (xbin==0) | (xbin==xnbin+1) | (ybin==0) | (ybin == ynbin+1)
    xbin = xbin[outliers==False] - 1
    ybin = ybin[outliers==False] - 1
    
    # Compute the sample indices in the flattened histogram matrix.
    if xnbin >= ynbin:
        xy = ybin*(xnbin) + xbin
        
    else:
        xy = xbin*(ynbin) + ybin
        
       
    # Compute the number of repetitions in xy and assign it to the flattened
    #  histogram matrix.

    flatcount = np.bincount(xy)
    indices = np.arange(len(flatcount))
    hist[indices] = flatcount

    shape = np.sort([xnbin, ynbin])
    # Shape into a proper matrix
    histmat = hist.reshape(shape)
    if (shape == (ynbin, xnbin)).all():
        histmat = histmat.T
    if normed:
        diff2 = np.outer(dxedges, dyedges)
        histmat = histmat / diff2 / histmat.sum()
    return histmat, xedges, yedges
