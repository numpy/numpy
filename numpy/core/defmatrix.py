
__all__ = ['matrix', 'bmat', 'mat', 'asmatrix']

import numeric as N
from numeric import ArrayType, concatenate, isscalar, binary_repr
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
            new = data.view(matrix)
            if intype != data.dtype:
                return new.astype(intype)
            if copy: return new.copy()
            else: return new

        if isinstance(data, types.StringType):
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
        if ndim == 0:
            self.shape = (1,1)
        elif ndim == 1:
            self.shape = (1,self.shape[0])
        return

    def __getitem__(self, index):
        out = N.ndarray.__getitem__(self, index)
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
        return out

    def _get_truendim(self):
        shp = self.shape
        truend = 0
        for val in shp:
            if (val > 1): truend += 1
        return truend

    def __mul__(self, other):
        if isinstance(other, N.ndarray) or N.isscalar(other) or \
               not hasattr(other, '__rmul__'):
            return N.dot(self, other)
        else:
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
        return repr(self.__array__()).replace('array','matrix')

    def __str__(self):
        return str(self.__array__())

    def sum(self, axis=None, dtype=None):
        """Sum the matrix over the given axis.  If the axis is None, sum
        over all dimensions.  This preserves the orientation of the
        result as a row or column.
        """
        s = N.ndarray.sum(self, axis, dtype)
        if axis==1:
            return s.transpose()
        else:
            return s

    # Needed becase tolist method expects a[i]
    #  to have dimension a.ndim-1
    def tolist(self):
        return self.__array__().tolist()

    def getA(self):
        return self.__array__()

    def getT(self):
        return self.transpose()

    def getH(self):
        if issubclass(self.dtype.type, N.complexfloating):
            return self.transpose().conjugate()
        else:
            return self.transpose()

    def getI(self):
        from numpy.dual import inv
        return matrix(inv(self))

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


def bmat(obj, ldict=None, gdict=None):
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
