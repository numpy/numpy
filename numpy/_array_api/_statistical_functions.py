import numpy as np

def max(x, /, *, axis=None, keepdims=False):
    return np.max(x, axis=axis, keepdims=keepdims)

def mean(x, /, *, axis=None, keepdims=False):
    return np.mean(x, axis=axis, keepdims=keepdims)

def min(x, /, *, axis=None, keepdims=False):
    return np.min(x, axis=axis, keepdims=keepdims)

def prod(x, /, *, axis=None, keepdims=False):
    return np.prod(x, axis=axis, keepdims=keepdims)

def std(x, /, *, axis=None, correction=0.0, keepdims=False):
    # Note: the keyword argument correction is different here
    return np.std(x, axis=axis, ddof=correction, keepdims=keepdims)

def sum(x, /, *, axis=None, keepdims=False):
    return np.sum(x, axis=axis, keepdims=keepdims)

def var(x, /, *, axis=None, correction=0.0, keepdims=False):
    # Note: the keyword argument correction is different here
    return np.var(x, axis=axis, ddof=correction, keepdims=keepdims)
