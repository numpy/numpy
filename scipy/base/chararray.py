from numerictypes import character, string, unicode_, obj2dtype
from numeric import ndarray

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
        return NotImplemented

    def __ne__(self, other):
        return NotImplemented

    def __ge__(self, other):
        return NotImplemented

    def __le__(self, other):
        return NotImplemented

    def __gt__(self, other):
        return NotImplemented

    def __lt__(self, other):
        return NotImplemented

    def __add__(self, other):
        return NotImplemented

    def __radd__(self, other):
        return NotImplemented

    def __mul__(self, other):
        return NotImplemented

    def __rmul__(self, other):
        return NotImplemented

    def __mod__(self, other):
        return NotImplemented

    def __rmod__(self, other):
        return NotImplemented


    def _strmethod(self, *args, **kwds):
        name = args[0]
        args = args[1:]
        
        for obj in self.flat:
            getattr(obj, name)(*args, **kwds)

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

    if isinstance(obj, chararray):
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
    
    
    val = asarray(obj).astype(dtype)
    
    return chararray(val.shape, itemlen, unicode, buffer=val,
                     strides=val.strides, fortran=fortran)

    
def aschararray(obj):
    return chararray(obj, copy=False)
