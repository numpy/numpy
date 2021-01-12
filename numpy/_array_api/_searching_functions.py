import numpy as np

def argmax(x, /, *, axis=None, keepdims=False):
    return np.argmax(x, axis=axis, keepdims=keepdims)

def argmin(x, /, *, axis=None, keepdims=False):
    return np.argmin(x, axis=axis, keepdims=keepdims)

def nonzero(x, /):
    return np.nonzero(x)

def where(condition, x1, x2, /):
    return np.where(condition, x1, x2)
