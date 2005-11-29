from numerictypes import character, string, unicode_, obj2dtype, integer
from numeric import ndarray, multiter, empty

# special sub-class for character arrays (string and unicode_)
# This adds equality testing and methods of str and unicode types
#  which operate on an element-by-element basis

class ndchararray(ndarray):
    def __new__(subtype, shape, itemlen=1, unicode=False, buffer=None,
                offset=0, strides=None, swap=0, fortran=0):

        if unicode:
            dtype = 'U%d' % itemlen
        else:
            dtype = 'U%d' % itemlen
            swap = 0


        if buffer is None:
            self = ndarray.__new__(subtype, shape, dtype, fortran=fortran)
        else:
            self = ndarray.__new__(subtype, shape, dtype, buffer=buffer,
                                   offset=offset, strides=strides,
                                   swap=swap, fortran=fortran)
        return self


    def __reduce__(self):
        pass

    # these should be moved to C
    def __eq__(self, other):
        b = multiter(self, other)
        result = empty(b.shape, dtype=bool)
        res = result.flat
        for k, val in enumerate(b):
            res[k] = (val[0] == val[1])
        return result

    def __ne__(self, other):
        b = multiter(self, other)
        result = empty(b.shape, dtype=bool)
        res = result.flat
        for k, val in enumerate(b):
            res[k] = (val[0] != val[1])
        return result

    def __ge__(self, other):
        b = multiter(self, other)
        result = empty(b.shape, dtype=bool)
        res = result.flat
        for k, val in enumerate(b):
            res[k] = (val[0] >= val[1])
        return result
        
    def __le__(self, other):
        b = multiter(self, other)
        result = empty(b.shape, dtype=bool)
        res = result.flat
        for k, val in enumerate(b):
            res[k] = (val[0] <= val[1])
        return result        

    def __gt__(self, other):
        b = multiter(self, other)
        result = empty(b.shape, dtype=bool)
        res = result.flat
        for k, val in enumerate(b):
            res[k] = (val[0] > val[1])
        return result
    
    def __lt__(self, other):
        b = multiter(self, other)
        result = empty(b.shape, dtype=bool)
        res = result.flat
        for k, val in enumerate(b):
            res[k] = (val[0] < val[1])
        return result        
        
    def __add__(self, other):
        b = multiter(self, other)
        arr = b.iters[1].base
        outitem = self.itemsize + arr.itemsize
        dtype = self.dtypestr[1:2] + str(outitem)
        result = empty(b.shape, dtype=dtype)
        res = result.flat
        for k, val in enumerate(b):
            res[k] = (val[0] + val[1])
        return result 

    def __radd__(self, other):
        b = multiter(other, self)
        outitem = b.iters[0].base.itemsize + \
                  b.iters[1].base.itemsize
        dtype = self.dtypestr[1:2] + str(outitem)
        result = empty(b.shape, dtype=dtype)
        res = result.flat
        for k, val in enumerate(b):
            res[k] = (val[0] + val[1])
        return result 

    def __mul__(self, other):
        b = multiter(self, other)
        arr = b.iters[1].base
        if not issubclass(arr.dtype, integer):
            raise ValueError, "Can only multiply by integers"
        outitem = b.iters[0].base.itemsize * arr.max()
        dtype = self.dtypestr[1:2] + str(outitem)
        result = empty(b.shape, dtype=dtype)
        res = result.flat
        for k, val in enumerate(b):
            res[k] = val[0]*val[1]
        return result

    def __rmul__(self, other):
        b = multiter(self, other)
        arr = b.iters[1].base
        if not issubclass(arr.dtype, integer):
            raise ValueError, "Can only multiply by integers"
        outitem = b.iters[0].base.itemsize * arr.max()
        dtype = self.dtypestr[1:2] + str(outitem)
        result = empty(b.shape, dtype=dtype)
        res = result.flat
        for k, val in enumerate(b):
            res[k] = val[0]*val[1]
        return result

    def __mod__(self, other):
        return NotImplemented

    def __rmod__(self, other):
        return NotImplemented

    def capitalize(self):
        pass

    def center(self):
        pass

    def count(self):
        pass

    def decode(self):
        pass

    def encode(self):
        pass

    def endswith(self):
        pass

    def expandtabs(self):
        pass

    def find(self):
        pass

    def index(self):
        pass

    def isalnum(self):
        pass

    def isalpha(self):
        pass

    def isdigit(self):
        pass

    def islower(self):
        pass

    def isspace(self):
        pass

    def istitle(self):
        pass

    def isupper(self):
        pass

    def join(self):
        pass

    def ljust(self):
        pass

    def lower(self):
        pass

    def lstrip(self):
        pass

    def replace(self):
        pass

    def rfind(self):
        pass

    def rindex(self):
        pass

    def rjust(self):
        pass

    def rsplit(self):
        pass

    def rstrip(self):
        pass

    def split(self):
        pass

    def splitlines(self):
        pass

    def startswith(self):
        pass

    def strip(self):
        pass

    def swapcase(self):
        pass

    def title(self):
        pass

    def translate(self):
        pass

    def upper(self):
        pass

    def zfill(self):
        pass

                
def chararray(obj, itemlen=7, copy=True, unicode=False, fortran=False):

    if isinstance(obj, charndarray):
        if copy or (itemlen != obj.itemlen) \
           or (not unicode and obj.dtype == unicode_) \
           or (unicode and obj.dtype == string):
            return obj.astype(obj.dtypestr[1:])
        else:
            return obj

        
    if isinstance(obj, ndarray) and (obj.dtype in [unicode_, string]):
        copied = 0
        
        if unicode:
            dtype = 'U%d' % obj.itemlen
            if obj.dtype == string:
                obj = obj.astype(dtype)
                copied = 1
        else:
            dtype = 'S%d' % obj.itemlen
            if obj.dtype == unicode_:
                obj = obj.astype(dtype)
                copied = 1

        if copy and not copied:
            obj = obj.copy()

        return ndarray.__new__(chararray, obj.shape)

    if unicode:
        dtype = "U%d" % itemlen
    else:
        dtype = "S%d" % itemlen

    val = asarray(obj).astype(dtype)
    
    return ndchararray(val.shape, itemlen, unicode, buffer=val,
                       strides=val.strides, fortran=fortran)

    
def aschararray(obj):
    return chararray(obj, copy=False)
