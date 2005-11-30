from numerictypes import character, string, unicode_, obj2dtype, integer
from numeric import ndarray, broadcast, empty
import sys

# special sub-class for character arrays (string and unicode_)
# This adds equality testing and methods of str and unicode types
#  which operate on an element-by-element basis


class ndchararray(ndarray):
    def __new__(subtype, shape, itemlen=1, unicode=False, buffer=None,
                offset=0, strides=None, swap=0, fortran=0):

        if unicode:
            dtype = unicode_
        else:
            dtype = string
            swap = 0

        if buffer is None:
            self = ndarray.__new__(subtype, shape, dtype, itemlen, fortran=fortran)
        else:
            self = ndarray.__new__(subtype, shape, dtype, itemlen, buffer=buffer,
                                   offset=offset, strides=strides,
                                   swap=swap, fortran=fortran)
        return self

    def __reduce__(self):
        pass

    def _richcmpfunc(self, other, op):
        b = broadcast(self, other)
        result = empty(b.shape, dtype=bool)
        res = result.flat
        for k, val in enumerate(b):
            r1 = val[0].strip('\x00')
	    r2 = val[1]
	    res[k] = eval("r1 %s r2" % op, {'r1':r1,'r2':r2})
        return result
        
    # these should probably be moved to C
    def __eq__(self, other):
        return self._richcmpfunc(other, '==')

    def __ne__(self, other):
        return self._richcmpfunc(other, '!=')

    def __ge__(self, other):
        return self._richcmpfunc(other, '>=')

    def __le__(self, other):
        return self._richcmpfunc(other, '<=')

    def __gt__(self, other):
        return self._richcmpfunc(other, '>')
    
    def __lt__(self, other):
        return self._richcmpfunc(other, '<')
        
    def __add__(self, other):
        b = broadcast(self, other)
        arr = b.iters[1].base
        outitem = self.itemsize + arr.itemsize
        result = ndchararray(b.shape, outitem, self.dtype is unicode_)
        res = result.flat
        for k, val in enumerate(b):
            res[k] = (val[0] + val[1])
        return result 

    def __radd__(self, other):
        b = broadcast(other, self)
        outitem = b.iters[0].base.itemsize + \
                  b.iters[1].base.itemsize
        result = ndchararray(b.shape, outitem, self.dtype is unicode_)        
        res = result.flat
        for k, val in enumerate(b):
            res[k] = (val[0] + val[1])
        return result 

    def __mul__(self, other):
        b = broadcast(self, other)
        arr = b.iters[1].base
        if not issubclass(arr.dtype, integer):
            raise ValueError, "Can only multiply by integers"
        outitem = b.iters[0].base.itemsize * arr.max()
        result = ndchararray(b.shape, outitem, self.dtype is unicode_)
        res = result.flat
        for k, val in enumerate(b):
            res[k] = val[0]*val[1]
        return result

    def __rmul__(self, other):
        b = broadcast(self, other)
        arr = b.iters[1].base
        if not issubclass(arr.dtype, integer):
            raise ValueError, "Can only multiply by integers"
        outitem = b.iters[0].base.itemsize * arr.max()
        result = ndchararray(b.shape, outitem, self.dtype is unicode_)        
        res = result.flat
        for k, val in enumerate(b):
            res[k] = val[0]*val[1]
        return result

    def __mod__(self, other):
        b = broadcast(self, other)
        res = [None]*b.size
        maxsize = -1
        for k,val in enumerate(b):
            newval = val[0] % val[1]
            maxsize = max(len(newval), maxsize)
            res[k] = newval
        newarr = ndchararray(b.shape, maxsize, self.dtype is unicode_)
        nearr[:] = res
        return newarr
    
    def __rmod__(self, other):
        return NotImplemented

    def _generalmethod(self, name, myiter):
        res = [None]*myiter.size
        maxsize = -1
        for k, val in enumerate(myiter):
            newval = []
            for chk in val[1:]:
                if chk is None:
                    break
                newval.append(chk)
            newitem = getattr(val[0],name)(*newval)
            maxsize = max(len(newitem), maxsize)
            res[k] = newval
        newarr = ndchararray(myiter.shape, maxsize, self.dtype is unicode_)
        newarr[:] = res
        return newarr

    def _typedmethod(self, name, myiter, dtype):
        result = empty(myiter.shape, dtype=dtype)
        res = result.flat
        for k, val in enumerate(myiter):
            newval = []
            for chk in val[1:]:
                if chk is None:
                    break
                newval.append(chk)
            newitem = getattr(val[0],name)(*newval)
            res[k] = newval
        return res

    def _samemethod(self, name):
        result = self.copy()
        res = result.flat
        for k, val in enumerate(self.flat):
            res[k] = getattr(val, name)()
        return result

    def capitalize(self):
        return self._samemethod('capitalize')

    if sys.version[:3] >= '2.4':
        def center(self, width, fillchar=' '):
            return self._generalmethod('center', broadcast(self, width, fillchar))
        def ljust(self, width, fillchar=' '):
            return self._generalmethod('ljust', broadcast(self, width, fillchar))
        def rjust(self, width, fillchar=' '):
            return self._generalmethod('rjust', broadcast(self, width, fillchar))        
        def rsplit(self, sep=None, maxsplit=None):
            return self._generalmethod2('rsplit', broadcast(self, sep, maxsplit))
    else:
        def ljust(self, width):
            return self._generalmethod('ljust', broadcast(self, width))
        def rjust(self, width):
            return self._generalmethod('rjust', broadcast(self, width))        
        def center(self, width):
            return self._generalmethod('center', broadcast(self, width))            

    def count(self, sub, start=None, end=None):
        return self._typedmethod('count', broadcast(self, sub, start, end), int)

    def decode(self,encoding=None,errors=None):
        return self._generalmethod('decode', broadcast(self, encoding, errors))
    
    def encode(self,encoding=None,errors=None):
        return self._generalmethod('encode', broadcast(self, encoding, errors))
    
    def endswith(self, suffix, start=None, end=None):
        return self._typedmethod('endswith', broadcast(self, suffix, start, end), bool)
    
    def expandtabs(self, tabsize=None):
        return self._generalmethod('endswith', broadcast(self, tabsize))

    def find(self, sub, start=None, end=None):
        return self._typedmethod('find', broadcast(self, sub, start, end), int)

    def index(self, sub, start=None, end=None):
        return self._typedmethod('index', broadcast(self, sub, start, end), int)

    def _ismethod(self, name):        
        result = empty(self.shape, dtype=bool)
        res = result.flat
        for k, val in enumerate(self.flat):
            res[k] = getattr(val, name)()
        return result

    def isalnum(self):
        return self._ismethod('isalnum')

    def isalpha(self):
        return self._ismethod('isalpha')

    def isdigit(self):
        return self._ismethod('isdigit')

    def islower(self):
        return self._ismethod('islower')
    
    def isspace(self):
        return self._ismethod('isspace')

    def istitle(self):
        return self._ismethod('istitle')

    def isupper(self):
        return self._ismethod('isupper')

    def join(self, seq):
        return self._generalmethod('join', broadcast(self, seq))
        
    def lower(self):
        return self._samemethod('lower')

    def lstrip(self, chars):
        return self._generalmethod('lstrip', broadcast(self, chars))

    def replace(self, old, new, count=None):
        return self._generalmethod('replace', broadcast(self, old, new, count))

    def rfind(self, sub, start=None, end=None):
        return self._typedmethod('rfind', broadcast(self, sub, start, end), int)        

    def rindex(self, sub, start=None, end=None):
        return self._typedmethod('rindex', broadcast(self, sub, start, end), int)

    def rstrip(self, chars=None):
        return self._generalmethod('rstrip', broadcast(self, chars))        

    def split(self, sep=None, maxsplit=None):
        return self._typedmethod('split', broadcast(self, sep, maxsplit), object)

    def splitlines(self, keepends=None):
        return self._typedmethod('splitlines', broadcast(self, keepends), object)
    
    def startswith(self, prefix, start=None, end=None):
        return self._typedmethod('startswith', broadcast(self, prefix, start, end), bool)

    def strip(self, chars=None):
        return self._generalmethod('strip', broadcast(self, chars))        

    def swapcase(self):
        return self._samemethod('swapcase')

    def title(self):
        return self._samemethod('title')

    #deletechars not accepted for unicode objects
    def translate(self, table, deletechars=None):
        if self.dtype is unicode_:
            return self._generalmethod('translate', broadcast(self, table))
        else:
            return self._generalmethod('translate', broadcast(self, table, deletechars))
        
    def upper(self):
        return self._samemethod('upper')

    def zfill(self, width):
        return self._generalmethod('zfill', broadcast(self, width))

                
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
