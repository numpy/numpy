from contextlib import contextmanager

@contextmanager
def keep_0D():
    from numpy.core._multiarray_umath import PyArray_SetKeep0D
    PyArray_SetKeep0D(1)
    try:
        yield
    finally:
        PyArray_SetKeep0D(0)