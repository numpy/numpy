from numpy._core import numeric

# these must import without warning or error from numpy.core.numeric to
# support old pickle files
for item in ['_frombuffer']:
    globals()[item] = getattr(numeric, item)

def __getattr__(attr_name):
    from numpy._core import numeric

    from ._utils import _raise_warning

    sentinel = object()
    ret = getattr(numeric, attr_name, sentinel)
    if ret is sentinel:
        raise AttributeError(
            f"module 'numpy.core.numeric' has no attribute {attr_name}")
    _raise_warning(attr_name, "numeric")
    return ret


del numeric, item
