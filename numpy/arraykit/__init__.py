from info import __doc__

from numpy.core.multiarray import basearray
from numpy.core.multiarray import wrap, copy, view, astype, sort
from numpy.core.multiarray import getitem, setitem
from numpy.core.multiarray import getshape, setshape, getstrides, setstrides
from numpy.core.multiarray import getdtype, setdtype, getreal, setreal
from numpy.core.multiarray import getimag, setimag, getflags, newfromobj

def unary_op(op):
    def unop(self):
        return op(self)
    return unop

def binary_op(op):
    def binop(self, other):
        return op(self, other)
    return binop

def reverse_op(op):
    def revop(self, other):
        return op(other, self)
    return revop

def inplace_op(op):
    def iop(self, other):
        return op(self, other, self)[()]
    return iop

def str(obj):
    import numpy
    return wrap(numpy.array([0]), obj).__str__()
    
def repr(obj):
    import numpy
    return wrap(numpy.array([0]), obj).__repr__()
    


