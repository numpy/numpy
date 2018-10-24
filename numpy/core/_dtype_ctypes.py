"""
Conversion from ctypes to dtype.

In an ideal world, we could acheive this through the PEP3118 buffer protocol,
something like::

    def dtype_from_ctypes_type(t):
        # needed to ensure that the shape of `t` is within memoryview.format
        class DummyStruct(ctypes.Structure):
            _fields_ = [('a', t)]

        # empty to avoid memory allocation
        ctype_0 = (DummyStruct * 0)()
        mv = memoryview(ctype_0)

        # convert the struct, and slice back out the field
        return _dtype_from_pep3118(mv.format)['a']

Unfortunately, this fails because:

* ctypes cannot handle length-0 arrays with PEP3118 (bpo-32782)
* PEP3118 cannot represent unions, but both numpy and ctypes can
* ctypes cannot handle big-endian structs with PEP3118 (bpo-32780)
"""
import _ctypes
import ctypes

import numpy as np


def _from_ctypes_array(t):
    return np.dtype((dtype_from_ctypes_type(t._type_), (t._length_,)))


def _from_ctypes_structure(t):
    # TODO: gh-10533, gh-10532
    fields = []
    for item in t._fields_:
        if len(item) > 2:
            raise TypeError(
                "ctypes bitfields have no dtype equivalent")
        fname, ftyp = item
        fields.append((fname, dtype_from_ctypes_type(ftyp)))

    # by default, ctypes structs are aligned
    return np.dtype(fields, align=True)


def dtype_from_ctypes_type(t):
    """
    Construct a dtype object from a ctypes type
    """
    if issubclass(t, _ctypes.Array):
        return _from_ctypes_array(t)
    elif issubclass(t, _ctypes._Pointer):
        raise TypeError("ctypes pointers have no dtype equivalent")
    elif issubclass(t, _ctypes.Structure):
        return _from_ctypes_structure(t)
    elif issubclass(t, _ctypes.Union):
        # TODO
        raise NotImplementedError(
            "conversion from ctypes.Union types like {} to dtype"
            .format(t.__name__))
    elif isinstance(t._type_, str):
        return np.dtype(t._type_)
    else:
        raise NotImplementedError(
            "Unknown ctypes type {}".format(t.__name__))
