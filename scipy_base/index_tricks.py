import types
import Numeric
__all__ = ['mgrid','ogrid','r_', 'row', 'c_', 'col', 'index_exp']

from type_check import ScalarType, asarray
import function_base
import matrix_base
makemat = matrix_base.Matrix.Matrix

class nd_grid:
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
    def __init__(self, sparse=0):
        self.sparse = sparse
    def __getitem__(self,key):
        try:
	    size = []
            typecode = Numeric.Int
	    for k in range(len(key)):
	        step = key[k].step
                start = key[k].start
                if start is None: start=0
                if step is None: step=1
                if type(step) is type(1j):
                    size.append(int(abs(step)))
                    typecode = Numeric.Float
                else:
                    size.append(int((key[k].stop - start)/(step*1.0)))
                if isinstance(step,types.FloatType) or \
                   isinstance(start, types.FloatType) or \
                   isinstance(key[k].stop, types.FloatType):
                       typecode = Numeric.Float
            if self.sparse:
                nn = map(lambda x,t: Numeric.arange(x,typecode=t),size,(typecode,)*len(size))
            else:
                nn = Numeric.indices(size,typecode)
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
                slobj = [Numeric.NewAxis]*len(size)
                for k in range(len(size)):
                    slobj[k] = slice(None,None)
                    nn[k] = nn[k][slobj]
                    slobj[k] = Numeric.NewAxis
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
                return Numeric.arange(0,length,1,Numeric.Float)*step + start
            else:
                return Numeric.arange(start, stop, step)
	    
    def __getslice__(self,i,j):
        return Numeric.arange(i,j)

    def __len__(self):
        return 0

mgrid = nd_grid()
ogrid = nd_grid(1)

import sys
class concatenator:
    """ Translates slice objects to concatenation along an axis.
    """
    def _retval(self, res):
        if not self.matrix:
            return res
        else:
            if self.axis == 0:
                return makemat(res)
            else:
                return makemat(res).T        
        
    def __init__(self, axis=0, matrix=0):
        self.axis = axis
        self.matrix = matrix
    def __getitem__(self,key):
        if isinstance(key,types.StringType):
            frame = sys._getframe().f_back
            mymat = matrix_base.bmat(key,frame.f_globals,frame.f_locals)
            if self.matrix:
                return mymat
            else:
                return asarray(mymat)
        if type(key) is not types.TupleType:
            key = (key,)
        objs = []
        for k in range(len(key)):
            if type(key[k]) is types.SliceType:
                typecode = Numeric.Int
	        step = key[k].step
                start = key[k].start
                stop = key[k].stop
                if start is None: start = 0
                if step is None:
                    step = 1
                if type(step) is type(1j):
                    size = int(abs(step))
                    typecode = Numeric.Float
                    newobj = function_base.linspace(start, stop, num=size)
                else:
                    newobj = Numeric.arange(start, stop, step)
            elif type(key[k]) in ScalarType:
                newobj = asarray([key[k]])
            else:
                newobj = key[k]
            objs.append(newobj)
        res = Numeric.concatenate(tuple(objs),axis=self.axis)
        return self._retval(res)
         
    def __getslice__(self,i,j):
        res = Numeric.arange(i,j)
        return self._retval(res)

    def __len__(self):
        return 0

r_=concatenator(0)
c_=concatenator(-1)
row = concatenator(0,1)
col = concatenator(-1,1)

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

class _index_expression_class:
    import sys
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

