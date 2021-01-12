import numpy as np

def all(x, /, *, axis=None, keepdims=False):
    return np.all(x, axis=axis, keepdims=keepdims)

def any(x, /, *, axis=None, keepdims=False):
    return np.any(x, axis=axis, keepdims=keepdims)
