
__all__ = ['matrix', 'bmat', 'mat', 'asmatrix']

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
            if (dtype2 is dtype) and (not copy):
                return data
            return data.astype(dtype)

        if dtype is None:
            if isinstance(data, N.ndarray):
                dtype = data.dtype
        intype = N.obj2dtype(dtype)

        if isinstance(data, types.StringType):
            data = _convert_from_string(data)

        # now convert data to an array
        arr = N.array(data, dtype=intype, copy=copy)
        ndim = arr.ndim
        shape = arr.shape
        if (ndim > 2):
            raise ValueError, "matrix must be 2-dimensional"
        elif ndim == 0:
            shape = (1,1)
        elif ndim == 1:
            shape = (1,shape[0])

        fortran = False
        if (ndim == 2) and arr.flags.fortran:
            fortran = True
            
        if not (fortran or arr.flags.contiguous):
            arr = arr.copy()

        ret = N.ndarray.__new__(subtype, shape, arr.descr,
                                buffer=arr,
                                fortran=fortran,
                                swap=arr.flags.swapped)
        return ret

    def __array_finalize__(self, obj):
        ndim = self.ndim
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

    def __mul__(self, other):
        if isinstance(other, N.ndarray) and other.ndim == 0:
            return N.multiply(self, other)
        else:
            return N.dot(self, other)

    def __rmul__(self, other):
        if isinstance(other, N.ndarray) and other.ndim == 0:
            return N.multiply(other, self)
        else:
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
        raise NotImplementedError

    def __repr__(self):
        return repr(self.__array__()).replace('array','matrix')

    def __str__(self):
        return str(self.__array__())

    # Needed becase tolist method expects a[i] 
    #  to have dimension a.ndim-1
    def tolist(self):
        return self.__array__().tolist()

    def getA(self):
        return self.__array__()

    def getT(self):
        return self.transpose()

    def getH(self):
        if issubclass(self.dtype, N.complexfloating):
            return self.transpose().conjugate()
        else:
            return self.transpose()

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
