import numpy as np

def sliding_window_view(arr, shape):
    """
    This function was written by fnjn on github and posted to numpy/numpy.
    It takes in an array and a shape, and then gives back an array that is
    essentially moving around a planchette of the described shape over the
    array.
    """
    n = arr.shape
    o = n - shape + 1 # output shape
    strides = arr.strides
    new_shape = np.concatenate((o, shape), axis=0)
    new_strides = np.concatenate((strides, strides), axis=0)
    return np.lib.stride_tricks.as_strided(arr ,new_shape, new_strides)

def pascal(n):
    """returns the nth line of a pascal's triangle"""
    return np.array([1]) if n == 1 else np.array([0,*pascal(n-1)])+np.array([*pascal(n-1),0])

def triangle(n):
    return np.array([*range(1,int((n+2)/2))]+[*range(int((n+1)/2),0,-1)])

def quadratic(n):
    return triangle(n)**2

def fnjn_mvgavg(a, n, axis=0, weights=False):
    shape = a.shape-(n-1)*np.eye(a.ndim,dtype='int')[axis]
    window = sliding_window_view(a, shape)
    if weights:
        if weights == 'pascal': weights = pascal(n)
        elif weights == 'triangle': weights = triangle(n)
        elif weights == 'quadratic': weights = quadratic(n)
        window = window.swapaxes(0,axis)
        assert len(window)==len(weights)
        window=np.array([window[n]*weights[n] for n in range(len(weights))])/np.sum(weights)
        window = window.swapaxes(0,axis)
    return np.mean(window,axis=axis).reshape(shape)


def cumsum_mvgavg(a, n, axis=0):
    table = np.cumsum(a.swapaxes(0,axis),axis=0)/n
    try:
        table = np.vstack([[0*table[0]],table])
    except:
        table = np.array([0,*table])
    return np.swapaxes(table[n:]-table[:-n],0,axis)


def binning_mvgavg(a,n,axis=0):
    a=a.swapaxes(0,axis)
    return a[:int(len(a)/n)*n].reshape(int(len(a)/n),n,*a.shape[1:]).mean(axis=1).swapaxes(0,axis)    
