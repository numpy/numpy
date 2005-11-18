
__all__ = ['matrix', 'bmat', 'mat']

import numeric as N
from numeric import ArrayType, concatenate, integer, multiply, power
from type_check import isscalar
from function_base import binary_repr
import types
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
            raise ValueError, "rows not the same size"
        count += 1
        newdata.append(newrow)
    return newdata


class matrix(object):
    __array_priority__ = 10.0
    def __init__(self, data, dtype=None, copy=True):
        if isinstance(data, matrix):
            swapped = data.flags.swapped
            dtype2 = data.dtype
            if (dtype is None):
                dtype = dtype2
            if (dtype2 is dtype) and (not copy):
                return data
            return data.astype(dtype)
        elif isinstance(data, N.ndarray):
            swapped = data.flags.swapped
            if dtype is None:
                dtype = data.dtype
        else:
            swapped = False
        intype = N.obj2dtype(dtype)

        if isinstance(data, types.StringType):
            data = _convert_from_string(data)

        # now convert data to an array
        arr = N.array(data, dtype=intype, copy=copy)
        arr.flags.swapped = swapped
        ndim = arr.ndim
        shape = arr.shape
        if (ndim > 2):
            raise ValueError, "matrix must be 2-dimensional"
        elif ndim == 0:
            arr = arr.reshape((1,1))
        elif ndim == 1:
            shape = (1,shape[0])
        arr.shape = shape
        self.arr = arr


    def __array_finalize__(self, obj):
        ndim = self.arr.ndim
        if ndim == 0:
            arr.shape = (1, 1)
        elif ndim == 1:
            arr.shape = (1, self.arr.shape[0])
        return

    def __setitem__(self, index, value):
        out = self.arr.__setitem__(index, value)

    def __getitem__(self, index):
        out = self.arr.__getitem__(index)
        # Need to swap if slice is on first index
        retscal = False
        try:
            n = len(index)
            if (n==2):
                if isinstance(index[0], types.SliceType):
                    if (isscalar(index[1])):
                        sh = out.shape
                        out.shape = (sh[1], sh[0])
                else:
                    if (isscalar(index[0])) and (isscalar(index[1])):
                        retscal = True
        except TypeError:
            pass
        if retscal and out.shape == (1,1): # convert scalars
            return out.A[0,0]
        # Return array if the output is 1-d, or matrix if the output is 2-d
        if out.ndim == 2:
            return matrix(out)
        else:
            return out

    def copy(self):
        return matrix(self.arr.copy())

    def __copy__(self):
        return matrix(self.arr.copy())

    def __add__(self, other):
        return matrix(self.arr + other)

    def __radd__(self, other):
        return matrix(other + self.arr)

    def __sub__(self, other):
        return matrix(self.arr - other)

    def __rsub__(self, other):
        return matrix(other - self.arr)

    def __mul__(self, other):
        if (isinstance(other, N.ndarray) or isinstance(other, matrix)) \
                and other.ndim == 0:
            return matrix(N.multiply(self.arr, other))
        else:
            return matrix(N.dot(self.arr, other))

    def __rmul__(self, other):
        if (isinstance(other, N.ndarray) or isinstance(other, matrix)) \
                and other.ndim == 0:
            return matrix(N.multiply(other, self.arr))
        else:
            return matrix(N.dot(other, self.arr))

    def __div__(self, other):
        try:
            if other.ndim == 0:
                return matrix(N.divide(self.arr, other))
            else:
                raise NotImplementedError, "matrix division not yet implemented"
        except AttributeError: 
            return matrix(N.divide(self.arr, other))

    def __rdiv__(self, other):
        try:
            if other.ndim == 0:
                return matrix(N.divide(other, self.arr))
            else:
                raise NotImplementedError, "matrix division not yet implemented"
        except AttributeError: 
            return matrix(N.divide(other, self.arr))

    def __iadd__(self, other):
        new = self.arr + other
        try:
            self.arr[:] = new
        except TypeError:
            self.arr = new
        return self

    def __isub__(self, other):
        new = self.arr - other
        try:
            self.arr[:] = new
        except TypeError:
            self.arr = new
        return self

    def __imul__(self, other):
        new = (self * other).arr
        try:
            self.arr[:] = new
        except TypeError:
            self.arr = new
        return self

    def __idiv__(self, other):
        new = (self / other).arr
        try:
            self.arr[:] = new
        except TypeError:
            self.arr = new
        return self

    def __int__(self):
        return int(self.arr)

    def __float__(self):
        return float(self.arr)

    def __long__(self):
        return long(self.arr)

    def __complex__(self):
        return complex(self.arr)

    def __oct__(self):
        return oct(self.arr)

    def __hex__(self):
        return hex(self.arr)

    def __len__(self):
        return len(self.arr)

    def __contains__(self, item):
        return self.arr.__contains__(item)

    def __nonzero__(self):
        return self.arr.__nonzero__()

    def __lt__(self, item):
        return self.arr.__lt__(item)

    def __le__(self, item):
        return self.arr.__le__(item)

    def __gt__(self, item):
        return self.arr.__gt__(item)

    def __ge__(self, item):
        return self.arr.__ge__(item)

    def __eq__(self, item):
        return self.arr.__eq__(item)

    def __ne__(self, item):
        return self.arr.__ne__(item)

    def __pos__(self):
        return self.arr.__pos__()

    def __neg__(self):
        return self.arr.__neg__()

    def __abs__(self):
        return self.arr.__abs__()

    def __getattr__(self, obj):
        return self.arr.__getattribute__(obj)

    def __setattr__(self, obj, value):
        if obj in ('shape', 'arr'):
            object.__setattr__(self, obj, value)
        else:
            self.arr.__setattr__(obj, value)

    def __pow__(self, other):
        shape = self.arr.shape
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
            if other <= 3:
                result = x.copy()
                while(other>1):
                    result *= x
                    other -= 1
                return result
            # binary decomposition to reduce the number of matrix
            # multiplications for 'other' > 3.
            beta = binary_repr(other)
            t = len(beta)
            Z, q = x.copy(), 0
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
        raise NotImplementedError

    def __repr__(self):
        return repr(self.arr).replace('array','matrix')

    def __str__(self):
        return str(self.arr)

    # Needed becase tolist method expects a[i] 
    #  to have dimension a.ndim-1
    #def tolist(self):
    #    return self.__array__().tolist()

    def getA(self):
        return self.arr

    def getT(self):
        return matrix(self.arr.transpose())

    def getH(self):
        if issubclass(self.arr.dtype, N.complexfloating):
            return matrix(self.arr.transpose().conjugate())
        else:
            return matrix(self.arr.transpose())

    def getI(self):
        from scipy import linalg
        return matrix(linalg.inv(self))

    A = property(getA, None, doc="base array")
    T = property(getT, None, doc="transpose")
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


def bmat(obj,ldict=None, gdict=None):
    """Build a matrix object from string, nested sequence, or array.

    Ex:  F = bmat('A, B; C, D') 
         F = bmat([[A,B],[C,D]])
         F = bmat(r_[c_[A,B],c_[C,D]])

        all produce the same Matrix Object    [ A  B ]
                                              [ C  D ]

        if A, B, C, and D are appropriately shaped 2-d arrays.
    """
    if isinstance(obj, types.StringType):
        if gdict is None:
            # get previous frame
            frame = sys._getframe().f_back
            glob_dict = frame.f_globals
            loc_dict = frame.f_locals
        else:
            glob_dict = gdict
            loc_dict = ldict

        return matrix(_from_string(obj, glob_dict, loc_dict))

    if isinstance(obj, (types.TupleType, types.ListType)):
        # [[A,B],[C,D]]
        arr_rows = []
        for row in obj:
            if isinstance(row, ArrayType):  # not 2-d
                return matrix(concatenate(obj,axis=-1))
            else:
                arr_rows.append(concatenate(row,axis=-1))
        return matrix(concatenate(arr_rows,axis=0))
    if isinstance(obj, ArrayType):
        return matrix(obj)

mat = matrix

