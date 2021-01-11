def concat(arrays, /, *, axis=0):
    # Note: the function name is different here
    from .. import concatenate
    return concatenate(arrays, axis=axis)

def expand_dims(x, axis, /):
    from .. import expand_dims
    return expand_dims(x, axis)

def flip(x, /, *, axis=None):
    from .. import flip
    return flip(x, axis=axis)

def reshape(x, shape, /):
    from .. import reshape
    return reshape(x, shape)

def roll(x, shift, /, *, axis=None):
    from .. import roll
    return roll(x, shift, axis=axis)

def squeeze(x, /, *, axis=None):
    from .. import squeeze
    return squeeze(x, axis=axis)

def stack(arrays, /, *, axis=0):
    from .. import stack
    return stack(arrays, axis=axis)

__all__ = ['concat', 'expand_dims', 'flip', 'reshape', 'roll', 'squeeze', 'stack']
