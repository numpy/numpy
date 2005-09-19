
import numeric as N
import types
import string as str_

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

_lkup = {'0':'000',
         '1':'001',
         '2':'010',
         '3':'011',
         '4':'100',
         '5':'101',
         '6':'110',
         '7':'111'}

def _binary(num):
    ostr = oct(num)
    bin = ''
    for ch in ostr[1:]:
        bin += _lkup[ch]
    ind = 0
    while bin[ind] == '0':
        ind += 1
    return bin[ind:]


class matrix(N.ndarray):
    __array_priority__ = 10.0
    def __new__(self, data, dtype=None, copy=0):
        if isinstance(data, matrix):
            dtype2 = data.dtype
            if (dtype is None):
                dtype = dtype2
            if (dtype2 is dtype) and (not copy):
                return data
            return data.astype(dtype)

        if dtype is None:
            dtype = N.intp
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

        fortran = False;
        if (ndim == 2) and arr.flags['FORTRAN']:
            fortran = True

        if not (fortran or arr.flags['CONTIGUOUS']):
            arr = arr.copy()

        ret = N.ndarray.__new__(matrix, shape, intype, buffer=arr,
                                fortran=fortran,
                                swap=(not arr.flags['NOTSWAPPED']))
        return ret; 
            
    def __array_wrap__(self, obj):
        try:
            ret = matrix(obj,dtype=obj.dtype)
        except:
            ret = obj
        return ret

    def _update_meta(self, obj):
        ndim = self.ndim
        if ndim == 0:
            self.shape = (1,1)
        elif ndim == 1:
            self.shape = (1, self.shape[0])
        return

    def __getitem__(self, index):
        out = N.ndarray.__getitem__(self, index)
        # Need to swap if slice is on first index
        try:
            n = len(index)
            if (n > 1) and isinstance(index[0], types.SliceType) \
               and (isinstance(index[1], types.IntType) or
                    isinstance(index[1], types.LongType)):
                sh = out.shape
                out.shape = (sh[1], sh[0])
        except TypeError:
            pass
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

    def __pow__(self, other):
        if len(shape)!=2 or shape[0]!=shape[1]:
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
            beta = _binary(other)
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
        return repr(self.A).replace('array','matrix')

    def __str__(self):
        return str(self.A)

    def tolist(self):
        return self.A.tolist()    

    def getA(self):
        arr = self
        fortran = False;
        if (self.ndim == 2) and arr.flags['FORTRAN']:
            fortran = True

        if not (fortran or arr.flags['CONTIGUOUS']):
            arr = arr.copy()

        return N.ndarray.__new__(N.ndarray, self.shape, self.dtype, buffer=arr,
                                fortran=fortran,
                                swap=(not arr.flags['NOTSWAPPED']))
        
    def getT(self):
        return self.transpose()

    def getH(self):
        if issubclass(self.dtype, N.complexfloating):
            return self.transpose(self.conjugate())
        else:
            return self.transpose()

    # inverse doesn't work yet....
    def getI(self):
        return self

    A = property(getA, None, doc="Get base array")
    T = property(getT, None, doc="transpose")    
    H = property(getH, None, doc="hermitian (conjugate) transpose")
    

    
        
