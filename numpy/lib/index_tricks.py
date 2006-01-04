## Automatically adapted for numpy Sep 19, 2005 by convertcode.py

__all__ = ['mgrid','ogrid','r_', 'c_', 'index_exp', 'ix_','ndenumerate']

import sys
import types
import numeric as _nx
from numeric import asarray

from type_check import ScalarType
import function_base
import twodim_base as matrix_base
import matrix
makemat = matrix.matrix

def ix_(*args):
    """ Construct an open mesh from multiple sequences.

    This function takes n 1-d sequences and returns n outputs with n
    dimensions each such that the shape is 1 in all but one dimension and
    the dimension with the non-unit shape value cycles through all n
    dimensions.

    Using ix_() one can quickly construct index arrays that will index
    the cross product.

    a[ix_([1,3,7],[2,5,8])]  returns the array

    a[1,2]  a[1,5]  a[1,8]
    a[3,2]  a[3,5]  a[3,8]
    a[7,2]  a[7,5]  a[7,8]
    """
    out = []
    nd = len(args)
    baseshape = [1]*nd
    for k in range(nd):
        new = _nx.array(args[k])
        if (new.ndim <> 1):
            raise ValueError, "Cross index must be 1 dimensional"
        baseshape[k] = len(new)
        new.shape = tuple(baseshape)
        out.append(new)
        baseshape[k] = 1
    return tuple(out)

class nd_grid(object):
    """ Construct a "meshgrid" in N-dimensions.

        grid = nd_grid() creates an instance which will return a mesh-grid
        when indexed.  The dimension and number of the output arrays are equal
        to the number of indexing dimensions.  If the step length is not a
        complex number, then the stop is not inclusive.

        However, if the step length is a COMPLEX NUMBER (e.g. 5j), then the
        integer part of it's magnitude is interpreted as specifying the
        number of points to create between the start and stop values, where
        the stop value IS INCLUSIVE.

        If instantiated with an argument of 1, the mesh-grid is open or not
        fleshed out so that only one-dimension of each returned argument is
        greater than 1

        Example:

           >>> mgrid = nd_grid()
           >>> mgrid[0:5,0:5]
           array([[[0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2],
                   [3, 3, 3, 3, 3],
                   [4, 4, 4, 4, 4]],
                  [[0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4]]])
           >>> mgrid[-1:1:5j]
           array([-1. , -0.5,  0. ,  0.5,  1. ])

           >>> ogrid = nd_grid(1)
           >>> ogrid[0:5,0:5]
           [array([[0],[1],[2],[3],[4]]), array([[0, 1, 2, 3, 4]])] 
    """
    def __init__(self, sparse=False):
        self.sparse = sparse
    def __getitem__(self,key):
        try:
            size = []
            typecode = _nx.Int
            for k in range(len(key)):
                step = key[k].step
                start = key[k].start
                if start is None: start=0
                if step is None: step=1
                if type(step) is type(1j):
                    size.append(int(abs(step)))
                    typecode = _nx.Float
                else:
                    size.append(int((key[k].stop - start)/(step*1.0)))
                if isinstance(step,types.FloatType) or \
                   isinstance(start, types.FloatType) or \
                   isinstance(key[k].stop, types.FloatType):
                       typecode = _nx.Float
            if self.sparse:
                nn = map(lambda x,t: _nx.arange(x,dtype=t),size,(typecode,)*len(size))
            else:
                nn = _nx.indices(size,typecode)
            for k in range(len(size)):
                step = key[k].step
                start = key[k].start
                if start is None: start=0
                if step is None: step=1
                if type(step) is type(1j):
                    step = int(abs(step))
                    step = (key[k].stop - start)/float(step-1)
                nn[k] = (nn[k]*step+start)
            if self.sparse:
                slobj = [_nx.NewAxis]*len(size)
                for k in range(len(size)):
                    slobj[k] = slice(None,None)
                    nn[k] = nn[k][slobj]
                    slobj[k] = _nx.NewAxis
            return nn
        except (IndexError, TypeError):
            step = key.step
            stop = key.stop
            start = key.start
            if start is None: start = 0
            if type(step) is type(1j):
                step = abs(step)
                length = int(step)
                step = (key.stop-start)/float(step-1)
                stop = key.stop+step
                return _nx.arange(0,length,1,_nx.Float)*step + start
            else:
                return _nx.arange(start, stop, step)

    def __getslice__(self,i,j):
        return _nx.arange(i,j)

    def __len__(self):
        return 0

mgrid = nd_grid()
ogrid = nd_grid(1)

class concatenator(object):
    """ Translates slice objects to concatenation along an axis.
    """
    def _retval(self, res):
        if self.matrix:
            oldndim = res.ndim
            res = makemat(res)
            if oldndim == 1 and self.col:
                res = res.T
        self.axis=self._axis
        self.matrix=self._matrix
        self.col=0
        return res

    def __init__(self, axis=0, matrix=False):
        self._axis = axis
        self._matrix = matrix
        self.axis = axis
        self.matrix = matrix
        self.col = 0

    def __getitem__(self,key):
        if isinstance(key,types.StringType):
            frame = sys._getframe().f_back
            mymat = matrix.bmat(key,frame.f_globals,frame.f_locals)
            return mymat
        if type(key) is not types.TupleType:
            key = (key,)
        objs = []
        for k in range(len(key)):
            if type(key[k]) is types.SliceType:
                step = key[k].step
                start = key[k].start
                stop = key[k].stop
                if start is None: start = 0
                if step is None:
                    step = 1
                if type(step) is type(1j):
                    size = int(abs(step))
                    newobj = function_base.linspace(start, stop, num=size)
                else:
                    newobj = _nx.arange(start, stop, step)
            elif type(key[k]) is types.StringType:
                if (key[k] in 'rc'):
                    self.matrix = True
                    self.col = (key[k] == 'c')
                    continue
                try:
                    self.axis = int(key[k])
                    continue
                except:
                    raise ValueError, "Unknown special directive."
            elif type(key[k]) in ScalarType:
                newobj = asarray([key[k]])
            else:
                newobj = key[k]
            objs.append(newobj)
        res = _nx.concatenate(tuple(objs),axis=self.axis)
        return self._retval(res)

    def __getslice__(self,i,j):
        res = _nx.arange(i,j)
        return self._retval(res)

    def __len__(self):
        return 0

r_=concatenator(0)
c_=concatenator(-1)
#row = concatenator(0,1)
#col = concatenator(-1,1)


# A simple nd index iterator over an array:

class ndenumerate(object):
    def __init__(self, arr):
        arr = asarray(arr)
        self.iter = enumerate(arr.flat)
        self.ashape = arr.shape
        self.nd = arr.ndim
        self.factors = [None]*(self.nd-1)
        val = self.ashape[-1]
        for i in range(self.nd-1,0,-1):
            self.factors[i-1] = val
            val *= self.ashape[i-1]

    def next(self):
        res = self.iter.next()
        indxs = [None]*self.nd
        val = res[0]
        for i in range(self.nd-1):
            indxs[i] = val / self.factors[i]
            val = val % self.factors[i]
        indxs[self.nd-1] = val
        return tuple(indxs), res[1]

    def __iter__(self):
        return self



# A nicer way to build up index tuples for arrays.
#
# You can do all this with slice() plus a few special objects,
# but there's a lot to remember. This version is simpler because
# it uses the standard array indexing syntax.
#
# Written by Konrad Hinsen <hinsen@cnrs-orleans.fr>
# last revision: 1999-7-23
#
# Cosmetic changes by T. Oliphant 2001
#
#
# This module provides a convenient method for constructing
# array indices algorithmically. It provides one importable object,
# 'index_expression'.
#
# For any index combination, including slicing and axis insertion,
# 'a[indices]' is the same as 'a[index_expression[indices]]' for any
# array 'a'. However, 'index_expression[indices]' can be used anywhere
# in Python code and returns a tuple of slice objects that can be
# used in the construction of complex index expressions.

class _index_expression_class(object):
    maxint = sys.maxint

    def __getitem__(self, item):
        if type(item) != type(()):
            return (item,)
        else:
            return item

    def __len__(self):
        return self.maxint

    def __getslice__(self, start, stop):
        if stop == self.maxint:
            stop = None
        return self[start:stop:None]

index_exp = _index_expression_class()

# End contribution from Konrad.

