from info import __doc__

from numpy.core.multiarray import basearray
from numpy.core.multiarray import wrap, copy, view, astype, sort, repr
from numpy.core.multiarray import getitem, setitem
from numpy.core.multiarray import getshape, setshape, getstrides, setstrides
from numpy.core.multiarray import getdtype, setdtype, getreal, setreal
from numpy.core.multiarray import getimag, setimag, getflags


def make_module():
    import numpy.core.umath
    
    def _wrap(func):
        # XXX transfer name and __doc__ and whatever....
        def helper(*args, **kwargs):
            return func(*args, **kwargs)
        return helper
        
    g = globals()
    for name in g:
        if name in ["make_module", "basearray"]:
            continue
        obj = g[name]
        if callable(obj) and name[:3] not in ('get', 'set'):
            g[name] = _wrap(obj)
        
    g['getitem'] = _wrap(g['getitem'])
    g['setitem'] = _wrap(g['setitem'])

        
    for name in dir(numpy.core.umath):
        obj = getattr(numpy.core.umath, name)
        if callable(obj):
            g[name] = _wrap(obj)

make_module()

def reverse_op(op):
    def revop(self, other):
        return op(other, self)
    return revop

def inplace_op(op):
    def iop(self, other):
        return op(self, other, self)[()]
    return iop

import numpy as _numpy

def _ndwrapper(obj, inst=_numpy.array([0])):
    return wrap(inst, obj)
def _wrapper(obj, inst=basearray([])):
    return wrap(inst, obj)

def str(obj):
    return _ndwrapper(obj).__str__()
    
def repr(obj):
    return _ndwrapper(obj).__repr__()
    
def fromobj(subtype, obj, dtype=None, order="C"):
    if order not in ["C", "FORTRAN"]:
        raise ValueError("Order must be either 'C' or 'FORTRAN', not %r" % order)
    nda = _numpy.array(obj, dtype, order=order)
    return basearray.__new__(subtype, nda.shape, nda.dtype, nda.data, order=order)

del make_module
