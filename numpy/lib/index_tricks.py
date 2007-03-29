## Automatically adapted for numpy Sep 19, 2005 by convertcode.py

__all__ = ['unravel_index',
           'mgrid',
           'ogrid',
           'r_', 'c_', 's_',
           'index_exp', 'ix_',
           'ndenumerate','ndindex']

import sys
import numpy.core.numeric as _nx
from numpy.core.numeric import asarray, ScalarType, array
import math

import function_base
import numpy.core.defmatrix as matrix
makemat = matrix.matrix

# contributed by Stefan van der Walt
def unravel_index(x,dims):
    """Convert a flat index into an index tuple for an array of given shape.

    e.g. for a 2x2 array, unravel_index(2,(2,2)) returns (1,0).

    Example usage:
      p = x.argmax()
      idx = unravel_index(p,x.shape)
      x[idx] == x.max()

    Note:  x.flat[p] == x.max()

      Thus, it may be easier to use flattened indexing than to re-map
      the index to a tuple.
    """
    if x > _nx.prod(dims)-1 or x < 0:
        raise ValueError("Invalid index, must be 0 <= x <= number of elements.")

    idx = _nx.empty_like(dims)

    # Take dimensions
    # [a,b,c,d]
    # Reverse and drop first element
    # [d,c,b]
    # Prepend [1]
    # [1,d,c,b]
    # Calculate cumulative product
    # [1,d,dc,dcb]
    # Reverse
    # [dcb,dc,d,1]
    dim_prod = _nx.cumprod([1] + list(dims)[:0:-1])[::-1]
    # Indeces become [x/dcb % a, x/dc % b, x/d % c, x/1 % d]
    return tuple(x/dim_prod % dims)

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
        new = _nx.asarray(args[k])
        if (new.ndim != 1):
            raise ValueError, "Cross index must be 1 dimensional"
        if issubclass(new.dtype.type, _nx.bool_):
            new = new.nonzero()[0]
        baseshape[k] = len(new)
        new = new.reshape(tuple(baseshape))
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

        If instantiated with an argument of sparse=True, the mesh-grid is
        open (or not fleshed out) so that only one-dimension of each returned
        argument is greater than 1

        Example:

           >>> mgrid = nd_grid()
           >>> mgrid[0:5,0:5]
           array([[[0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2],
                   [3, 3, 3, 3, 3],
                   [4, 4, 4, 4, 4]],
           <BLANKLINE>
                  [[0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4]]])
           >>> mgrid[-1:1:5j]
           array([-1. , -0.5,  0. ,  0.5,  1. ])

           >>> ogrid = nd_grid(sparse=True)
           >>> ogrid[0:5,0:5]
           [array([[0],
                  [1],
                  [2],
                  [3],
                  [4]]), array([[0, 1, 2, 3, 4]])]
           
    """
    def __init__(self, sparse=False):
        self.sparse = sparse
    def __getitem__(self,key):
        try:
            size = []
            typ = int
            for k in range(len(key)):
                step = key[k].step
                start = key[k].start
                if start is None: start=0
                if step is None: step=1
                if isinstance(step, complex):
                    size.append(int(abs(step)))
                    typ = float
                else:
                    size.append(math.ceil((key[k].stop - start)/(step*1.0)))
                if isinstance(step, float) or \
                    isinstance(start, float) or \
                    isinstance(key[k].stop, float):
                    typ = float
            if self.sparse:
                nn = map(lambda x,t: _nx.arange(x, dtype=t), size, \
                                     (typ,)*len(size))
            else:
                nn = _nx.indices(size, typ)
            for k in range(len(size)):
                step = key[k].step
                start = key[k].start
                if start is None: start=0
                if step is None: step=1
                if isinstance(step, complex):
                    step = int(abs(step))
                    if step != 1:
                        step = (key[k].stop - start)/float(step-1)
                nn[k] = (nn[k]*step+start)
            if self.sparse:
                slobj = [_nx.newaxis]*len(size)
                for k in range(len(size)):
                    slobj[k] = slice(None,None)
                    nn[k] = nn[k][slobj]
                    slobj[k] = _nx.newaxis
            return nn
        except (IndexError, TypeError):
            step = key.step
            stop = key.stop
            start = key.start
            if start is None: start = 0
            if isinstance(step, complex):
                step = abs(step)
                length = int(step)
                if step != 1:
                    step = (key.stop-start)/float(step-1)
                stop = key.stop+step
                return _nx.arange(0, length,1, float)*step + start
            else:
                return _nx.arange(start, stop, step)

    def __getslice__(self,i,j):
        return _nx.arange(i,j)

    def __len__(self):
        return 0

mgrid = nd_grid(sparse=False)
ogrid = nd_grid(sparse=True)

class concatenator(object):
    """Translates slice objects to concatenation along an axis.
    """
    def _retval(self, res):
        if self.matrix:
            oldndim = res.ndim
            res = makemat(res)
            if oldndim == 1 and self.col:
                res = res.T
        self.axis = self._axis
        self.matrix = self._matrix
        self.col = 0
        return res

    def __init__(self, axis=0, matrix=False, ndmin=1, trans1d=-1):
        self._axis = axis
        self._matrix = matrix
        self.axis = axis
        self.matrix = matrix
        self.col = 0
        self.trans1d = trans1d
        self.ndmin = ndmin

    def __getitem__(self,key):
        trans1d = self.trans1d
        ndmin = self.ndmin
        if isinstance(key, str):
            frame = sys._getframe().f_back
            mymat = matrix.bmat(key,frame.f_globals,frame.f_locals)
            return mymat
        if type(key) is not tuple:
            key = (key,)
        objs = []
        scalars = []
        final_dtypedescr = None
        for k in range(len(key)):
            scalar = False
            if type(key[k]) is slice:
                step = key[k].step
                start = key[k].start
                stop = key[k].stop
                if start is None: start = 0
                if step is None:
                    step = 1
                if isinstance(step, complex):
                    size = int(abs(step))
                    newobj = function_base.linspace(start, stop, num=size)
                else:
                    newobj = _nx.arange(start, stop, step)
                if ndmin > 1:
                    newobj = array(newobj,copy=False,ndmin=ndmin)
                    if trans1d != -1:
                        newobj = newobj.swapaxes(-1,trans1d)
            elif isinstance(key[k],str):
                if k != 0:
                    raise ValueError, "special directives must be the"\
                          "first entry."
                key0 = key[0]
                if key0 in 'rc':
                    self.matrix = True
                    self.col = (key0 == 'c')
                    continue
                if ',' in key0:
                    vec = key0.split(',')
                    try:
                        self.axis, ndmin = \
                                   [int(x) for x in vec[:2]]
                        if len(vec) == 3:
                            trans1d = int(vec[2])
                        continue
                    except:
                        raise ValueError, "unknown special directive"
                try:
                    self.axis = int(key[k])
                    continue
                except (ValueError, TypeError):
                    raise ValueError, "unknown special directive"
            elif type(key[k]) in ScalarType:
                newobj = array(key[k],ndmin=ndmin)
                scalars.append(k)
                scalar = True
            else:
                newobj = key[k]
                if ndmin > 1:
                    tempobj = array(newobj, copy=False, subok=True)
                    newobj = array(newobj, copy=False, subok=True,
                                   ndmin=ndmin)
                    if trans1d != -1 and tempobj.ndim < ndmin:
                        k2 = ndmin-tempobj.ndim
                        if (trans1d < 0):
                            trans1d += k2 + 1
                        defaxes = range(ndmin)
                        k1 = trans1d
                        axes = defaxes[:k1] + defaxes[k2:] + \
                               defaxes[k1:k2]
                        newobj = newobj.transpose(axes)
                    del tempobj
            objs.append(newobj)
            if isinstance(newobj, _nx.ndarray) and not scalar:
                if final_dtypedescr is None:
                    final_dtypedescr = newobj.dtype
                elif newobj.dtype > final_dtypedescr:
                    final_dtypedescr = newobj.dtype
        if final_dtypedescr is not None:
            for k in scalars:
                objs[k] = objs[k].astype(final_dtypedescr)
        res = _nx.concatenate(tuple(objs),axis=self.axis)
        return self._retval(res)

    def __getslice__(self,i,j):
        res = _nx.arange(i,j)
        return self._retval(res)

    def __len__(self):
        return 0

# separate classes are used here instead of just making r_ = concatentor(0),
# etc. because otherwise we couldn't get the doc string to come out right
# in help(r_)

class r_class(concatenator):
    """Translates slice objects to concatenation along the first axis.

        For example:
        >>> r_[array([1,2,3]), 0, 0, array([4,5,6])]
        array([1, 2, 3, 0, 0, 4, 5, 6])
        
    """
    def __init__(self):
        concatenator.__init__(self, 0)

r_ = r_class()

class c_class(concatenator):
    """Translates slice objects to concatenation along the second axis.
    """
    def __init__(self):
        concatenator.__init__(self, -1, ndmin=2, trans1d=0)

c_ = c_class()

class ndenumerate(object):
    """
    A simple nd index iterator over an array.

    Example:
    >>> a = array([[1,2],[3,4]])
    >>> for index, x in ndenumerate(a):
    ...     print index, x
    (0, 0) 1
    (0, 1) 2
    (1, 0) 3
    (1, 1) 4
    """
    def __init__(self, arr):
        self.iter = asarray(arr).flat

    def next(self):
        return self.iter.coords, self.iter.next()

    def __iter__(self):
        return self


class ndindex(object):
    """Pass in a sequence of integers corresponding
    to the number of dimensions in the counter.  This iterator
    will then return an N-dimensional counter.

    Example:
    >>> for index in ndindex(3,2,1):
    ...     print index
    (0, 0, 0)
    (0, 1, 0)
    (1, 0, 0)
    (1, 1, 0)
    (2, 0, 0)
    (2, 1, 0)

    """

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        self.nd = len(args)
        self.ind = [0]*self.nd
        self.index = 0
        self.maxvals = args
        tot = 1
        for k in range(self.nd):
            tot *= args[k]
        self.total = tot

    def _incrementone(self, axis):
        if (axis < 0):  # base case
            return
        if (self.ind[axis] < self.maxvals[axis]-1):
            self.ind[axis] += 1
        else:
            self.ind[axis] = 0
            self._incrementone(axis-1)

    def ndincr(self):
        self._incrementone(self.nd-1)

    def next(self):
        if (self.index >= self.total):
            raise StopIteration
        val = tuple(self.ind)
        self.index += 1
        self.ndincr()
        return val

    def __iter__(self):
        return self




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

class _index_expression_class(object):
    """
    A nicer way to build up index tuples for arrays.

    For any index combination, including slicing and axis insertion,
    'a[indices]' is the same as 'a[index_exp[indices]]' for any
    array 'a'. However, 'index_exp[indices]' can be used anywhere
    in Python code and returns a tuple of slice objects that can be
    used in the construction of complex index expressions.
    """
    maxint = sys.maxint
    def __init__(self, maketuple):
        self.maketuple = maketuple

    def __getitem__(self, item):
        if self.maketuple and type(item) != type(()):
            return (item,)
        else:
            return item

    def __len__(self):
        return self.maxint

    def __getslice__(self, start, stop):
        if stop == self.maxint:
            stop = None
        return self[start:stop:None]

index_exp = _index_expression_class(1)
s_ = _index_expression_class(0)

# End contribution from Konrad.
