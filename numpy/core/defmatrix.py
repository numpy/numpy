__all__ = ['matrix', 'bmat', 'mat', 'asmatrix']

import numeric as N
from numeric import concatenate, isscalar, binary_repr
import string as str_
import sys

# make translation table
_table = [None]*256
for k in range(256):
    _table[k] = chr(k)
_table = ''.join(_table)

_numchars = str_.digits + ".-+jeEL"
del str_
_todelete = []
for k in _table:
    if k not in _numchars:
        _todelete.append(k)
_todelete = ''.join(_todelete)
del k

def _eval(astr):
    return eval(astr.translate(_table,_todelete))

def _convert_from_string(data):
    rows = data.split(';')
    newdata = []
    count = 0
    for row in rows:
        trow = row.split(',')
        newrow = []
        for col in trow:
            temp = col.split()
            newrow.extend(map(_eval,temp))
        if count == 0:
            Ncols = len(newrow)
        elif len(newrow) != Ncols:
            raise ValueError, "Rows not the same size."
        count += 1
        newdata.append(newrow)
    return newdata

def asmatrix(data, dtype=None):
    """ Returns 'data' as a matrix.  Unlike matrix(), no copy is performed
    if 'data' is already a matrix or array.  Equivalent to:
    matrix(data, copy=False)
    """
    return matrix(data, dtype=dtype, copy=False)



class matrix(N.ndarray):
    __array_priority__ = 10.0
    def __new__(subtype, data, dtype=None, copy=True):
        if isinstance(data, matrix):
            dtype2 = data.dtype
            if (dtype is None):
                dtype = dtype2
            if (dtype2 == dtype) and (not copy):
                return data
            return data.astype(dtype)

        if isinstance(data, N.ndarray):
            if dtype is None:
                intype = data.dtype
            else:
                intype = N.dtype(dtype)
            new = data.view(subtype)
            if intype != data.dtype:
                return new.astype(intype)
            if copy: return new.copy()
            else: return new

        if isinstance(data, str):
            data = _convert_from_string(data)

        # now convert data to an array
        arr = N.array(data, dtype=dtype, copy=copy)
        ndim = arr.ndim
        shape = arr.shape
        if (ndim > 2):
            raise ValueError, "matrix must be 2-dimensional"
        elif ndim == 0:
            shape = (1,1)
        elif ndim == 1:
            shape = (1,shape[0])

        order = False
        if (ndim == 2) and arr.flags.fortran:
            order = True

        if not (order or arr.flags.contiguous):
            arr = arr.copy()

        ret = N.ndarray.__new__(subtype, shape, arr.dtype,
                                buffer=arr,
                                order=order)
        return ret

    def __array_finalize__(self, obj):
        self._getitem = False
        if (isinstance(obj, matrix) and obj._getitem): return
        ndim = self.ndim
        if (ndim == 2):
            return
        if (ndim > 2):
            newshape = tuple([x for x in self.shape if x > 1])
            ndim = len(newshape)
            if ndim == 2:
                self.shape = newshape
                return
            elif (ndim > 2):
                raise ValueError, "shape too large to be a matrix."
        else:
            newshape = self.shape
        if ndim == 0:
            self.shape = (1,1)
        elif ndim == 1:
            self.shape = (1,newshape[0])
        return

    def __getitem__(self, index):
        self._getitem = True
        try:
            out = N.ndarray.__getitem__(self, index)
        finally:
            self._getitem = False

        if not isinstance(out, N.ndarray):
            return out

        if out.ndim == 0:
            return out[()]
        if out.ndim == 1:
            sh = out.shape[0]
            # Determine when we should have a column array
            try:
                n = len(index)
            except:
                n = 0
            if n > 1 and isscalar(index[1]):
                out.shape = (sh,1)
            else:
                out.shape = (1,sh)
        return out

    def _get_truendim(self):
        shp = self.shape
        truend = 0
        for val in shp:
            if (val > 1): truend += 1
        return truend


    def __mul__(self, other):
        if isinstance(other,(N.ndarray, list, tuple)) :
            # This promotes 1-D vectors to row vectors
            return N.dot(self, asmatrix(other))
        if N.isscalar(other) or not hasattr(other, '__rmul__') :
            return N.dot(self, other)
        return NotImplemented

    def __rmul__(self, other):
        return N.dot(other, self)

    def __imul__(self, other):
        self[:] = self * other
        return self

    def __pow__(self, other):
        shape = self.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise TypeError, "matrix is not square"
        if type(other) in (type(1), type(1L)):
            if other==0:
                return matrix(N.identity(shape[0]))
            if other<0:
                x = self.I
                other=-other
            else:
                x=self
            result = x
            if other <= 3:
                while(other>1):
                    result=result*x
                    other=other-1
                return result
            # binary decomposition to reduce the number of Matrix
            #  Multiplies for other > 3.
            beta = binary_repr(other)
            t = len(beta)
            Z,q = x.copy(),0
            while beta[t-q-1] == '0':
                Z *= Z
                q += 1
            result = Z.copy()
            for k in range(q+1,t):
                Z *= Z
                if beta[t-k-1] == '1':
                    result *= Z
            return result
        else:
            raise TypeError, "exponent must be an integer"

    def __rpow__(self, other):
        return NotImplemented

    def __repr__(self):
        s = repr(self.__array__()).replace('array', 'matrix')
        # now, 'matrix' has 6 letters, and 'array' 5, so the columns don't
        # line up anymore. We need to add a space.
        l = s.splitlines()
        for i in range(1, len(l)):
            if l[i]:
                l[i] = ' ' + l[i]
        return '\n'.join(l)

    def __str__(self):
        return str(self.__array__())

    def _align(self, axis):
        """A convenience function for operations that need to preserve axis
        orientation.
        """
        if axis is None:
            return self[0,0]
        elif axis==0:
            return self
        elif axis==1:
            return self.transpose()
        else:
            raise ValueError, "unsupported axis"

    # To preserve orientation of result...
    def sum(self, axis=None, dtype=None, out=None):
        """Sum the matrix over the given axis.  If the axis is None, sum
        over all dimensions.  This preserves the orientation of the
        result as a row or column.
        """
        return N.ndarray.sum(self, axis, dtype, out)._align(axis)

    def mean(self, axis=None, out=None):
        return N.ndarray.mean(self, axis, out)._align(axis)

    def std(self, axis=None, dtype=None, out=None):
        return N.ndarray.std(self, axis, dtype, out)._align(axis)

    def var(self, axis=None, dtype=None, out=None):
        return N.ndarray.var(self, axis, dtype, out)._align(axis)

    def prod(self, axis=None, dtype=None, out=None):
        return N.ndarray.prod(self, axis, dtype, out)._align(axis)

    def any(self, axis=None, out=None):
        return N.ndarray.any(self, axis, out)._align(axis)

    def all(self, axis=None, out=None):
        return N.ndarray.all(self, axis, out)._align(axis)

    def max(self, axis=None, out=None):
        return N.ndarray.max(self, axis, out)._align(axis)

    def argmax(self, axis=None, out=None):
        return N.ndarray.argmax(self, axis, out)._align(axis)

    def min(self, axis=None, out=None):
        return N.ndarray.min(self, axis, out)._align(axis)

    def argmin(self, axis=None, out=None):
        return N.ndarray.argmin(self, axis, out)._align(axis)

    def ptp(self, axis=None, out=None):
        return N.ndarray.ptp(self, axis, out)._align(axis)

    # Needed becase tolist method expects a[i]
    #  to have dimension a.ndim-1
    def tolist(self):
        return self.__array__().tolist()

    def getI(self):
        M,N = self.shape
        if M == N:
            from numpy.dual import inv as func
        else:
            from numpy.dual import pinv as func
        return asmatrix(func(self))

    def getA(self):
        return self.__array__()

    def getA1(self):
        return self.__array__().ravel()

    def getT(self):
        return self.transpose()

    def getH(self):
        if issubclass(self.dtype.type, N.complexfloating):
            return self.transpose().conjugate()
        else:
            return self.transpose()

    T = property(getT, None, doc="transpose")
    A = property(getA, None, doc="base array")
    A1 = property(getA1, None, doc="1-d base array")
    H = property(getH, None, doc="hermitian (conjugate) transpose")
    I = property(getI, None, doc="inverse")

def _from_string(str,gdict,ldict):
    rows = str.split(';')
    rowtup = []
    for row in rows:
        trow = row.split(',')
        newrow = []
        for x in trow:
            newrow.extend(x.split())
        trow = newrow
        coltup = []
        for col in trow:
            col = col.strip()
            try:
                thismat = ldict[col]
            except KeyError:
                try:
                    thismat = gdict[col]
                except KeyError:
                    raise KeyError, "%s not found" % (col,)

            coltup.append(thismat)
        rowtup.append(concatenate(coltup,axis=-1))
    return concatenate(rowtup,axis=0)


def bmat(obj, ldict=None, gdict=None):
    """Build a matrix object from string, nested sequence, or array.

    Ex:  F = bmat('A, B; C, D')
         F = bmat([[A,B],[C,D]])
         F = bmat(r_[c_[A,B],c_[C,D]])

        all produce the same Matrix Object    [ A  B ]
                                              [ C  D ]

        if A, B, C, and D are appropriately shaped 2-d arrays.
    """
    if isinstance(obj, str):
        if gdict is None:
            # get previous frame
            frame = sys._getframe().f_back
            glob_dict = frame.f_globals
            loc_dict = frame.f_locals
        else:
            glob_dict = gdict
            loc_dict = ldict

        return matrix(_from_string(obj, glob_dict, loc_dict))

    if isinstance(obj, (tuple, list)):
        # [[A,B],[C,D]]
        arr_rows = []
        for row in obj:
            if isinstance(row, N.ndarray):  # not 2-d
                return matrix(concatenate(obj,axis=-1))
            else:
                arr_rows.append(concatenate(row,axis=-1))
        return matrix(concatenate(arr_rows,axis=0))
    if isinstance(obj, N.ndarray):
        return matrix(obj)

mat = asmatrix
