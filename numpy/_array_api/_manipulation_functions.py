import numpy as np

def concat(arrays, /, *, axis=0):
    # Note: the function name is different here
    return np.concatenate(arrays, axis=axis)

def expand_dims(x, axis, /):
    return np.expand_dims(x, axis)

def flip(x, /, *, axis=None):
    return np.flip(x, axis=axis)

def reshape(x, shape, /):
    return np.reshape(x, shape)

def roll(x, shift, /, *, axis=None):
    return np.roll(x, shift, axis=axis)

def squeeze(x, /, *, axis=None):
    return np.squeeze(x, axis=axis)

def stack(arrays, /, *, axis=0):
    return np.stack(arrays, axis=axis)
