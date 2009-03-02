import sys
from numerictypes import string_, unicode_, integer, object_
from numeric import ndarray, broadcast, empty, compare_chararrays
from numeric import array as narray

__all__ = ['chararray']

_globalvar = 0
_unicode = unicode

# special sub-class for character arrays (string_ and unicode_)
# This adds + and * operations and methods of str and unicode types
#  which operate on an element-by-element basis

# It also strips white-space on element retrieval and on
#   comparisons

class chararray(ndarray):
    """
    chararray(shape, itemsize=1, unicode=False, buffer=None, offset=0,
              strides=None, order=None)

    A character array of string or unicode type.

    The array items will be `itemsize` characters long.

    Create the array using buffer (with offset and strides) if it is
    not None. If buffer is None, then construct a new array with strides
    in Fortran order if len(shape) >=2 and order is 'Fortran' (otherwise
    the strides will be in 'C' order).

    """
    def __new__(subtype, shape, itemsize=1, unicode=False, buffer=None,
                offset=0, strides=None, order='C'):
        global _globalvar

        if unicode:
            dtype = unicode_
        else:
            dtype = string_

        _globalvar = 1
        if buffer is None:
            self = ndarray.__new__(subtype, shape, (dtype, itemsize),
                                   order=order)
        else:
            self = ndarray.__new__(subtype, shape, (dtype, itemsize),
                                   buffer=buffer,
                                   offset=offset, strides=strides,
                                   order=order)
        _globalvar = 0
        return self

    def __array_finalize__(self, obj):
        # The b is a special case because it is used for reconstructing.
        if not _globalvar and self.dtype.char not in 'SUbc':
            raise ValueError, "Can only create a chararray from string data."

    def __getitem__(self, obj):
        val = ndarray.__getitem__(self, obj)
        if isinstance(val, (string_, unicode_)):
            temp = val.rstrip()
            if len(temp) == 0:
                val = ''
            else:
                val = temp
        return val

    def __eq__(self, other):
        return compare_chararrays(self, other, '==', True)

    def __ne__(self, other):
        return compare_chararrays(self, other, '!=', True)

    def __ge__(self, other):
        return compare_chararrays(self, other, '>=', True)

    def __le__(self, other):
        return compare_chararrays(self, other, '<=', True)

    def __gt__(self, other):
        return compare_chararrays(self, other, '>', True)

    def __lt__(self, other):
        return compare_chararrays(self, other, '<', True)

    def __add__(self, other):
        b = broadcast(self, other)
        arr = b.iters[1].base
        outitem = self.itemsize + arr.itemsize
        result = chararray(b.shape, outitem, self.dtype is unicode_)
        res = result.flat
        for k, val in enumerate(b):
            res[k] = (val[0] + val[1])
        return result

    def __radd__(self, other):
        b = broadcast(other, self)
        outitem = b.iters[0].base.itemsize + \
                  b.iters[1].base.itemsize
        result = chararray(b.shape, outitem, self.dtype is unicode_)
        res = result.flat
        for k, val in enumerate(b):
            res[k] = (val[0] + val[1])
        return result

    def __mul__(self, other):
        b = broadcast(self, other)
        arr = b.iters[1].base
        if not issubclass(arr.dtype.type, integer):
            raise ValueError, "Can only multiply by integers"
        outitem = b.iters[0].base.itemsize * arr.max()
        result = chararray(b.shape, outitem, self.dtype is unicode_)
        res = result.flat
        for k, val in enumerate(b):
            res[k] = val[0]*val[1]
        return result

    def __rmul__(self, other):
        b = broadcast(self, other)
        arr = b.iters[1].base
        if not issubclass(arr.dtype.type, integer):
            raise ValueError, "Can only multiply by integers"
        outitem = b.iters[0].base.itemsize * arr.max()
        result = chararray(b.shape, outitem, self.dtype is unicode_)
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
        newarr = chararray(b.shape, maxsize, self.dtype is unicode_)
        newarr[:] = res
        return newarr

    def __rmod__(self, other):
        return NotImplemented

    def argsort(self, axis=-1, kind='quicksort', order=None):
        return self.__array__().argsort(axis, kind, order)

    def _generalmethod(self, name, myiter):
        res = [None]*myiter.size
        maxsize = -1
        for k, val in enumerate(myiter):
            newval = []
            for chk in val[1:]:
                if not chk or (chk.dtype is object_ and chk.item() is None):
                    break
                newval.append(chk)
            newitem = getattr(val[0],name)(*newval)
            maxsize = max(len(newitem), maxsize)
            res[k] = newitem
        newarr = chararray(myiter.shape, maxsize, self.dtype is unicode_)
        newarr[:] = res
        return newarr

    def _typedmethod(self, name, myiter, dtype):
        result = empty(myiter.shape, dtype=dtype)
        res = result.flat
        for k, val in enumerate(myiter):
            newval = []
            for chk in val[1:]:
                if not chk or (chk.dtype is object_ and chk.item() is None):
                    break
                newval.append(chk)
            this_str = val[0].rstrip('\x00')
            newitem = getattr(this_str,name)(*newval)
            res[k] = newitem
        return result

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
            return self._generalmethod('center',
                                       broadcast(self, width, fillchar))
        def ljust(self, width, fillchar=' '):
            return self._generalmethod('ljust',
                                       broadcast(self, width, fillchar))
        def rjust(self, width, fillchar=' '):
            return self._generalmethod('rjust',
                                       broadcast(self, width, fillchar))
        def rsplit(self, sep=None, maxsplit=None):
            return self._typedmethod('rsplit', broadcast(self, sep, maxsplit),
                                     object)
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
            item = val.rstrip('\x00')
            res[k] = getattr(item, name)()
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

    def translate(self, table, deletechars=None):
        if self.dtype is unicode_:
            return self._generalmethod('translate', broadcast(self, table))
        else:
            return self._generalmethod('translate', broadcast(self, table, deletechars))

    def upper(self):
        return self._samemethod('upper')

    def zfill(self, width):
        return self._generalmethod('zfill', broadcast(self, width))


def array(obj, itemsize=None, copy=True, unicode=False, order=None):

    if isinstance(obj, chararray):
        if itemsize is None:
            itemsize = obj.itemsize
        if copy or (itemsize != obj.itemsize) \
           or (not unicode and obj.dtype == unicode_) \
           or (unicode and obj.dtype == string_):
            return obj.astype("%s%d" % (obj.dtype.char, itemsize))
        else:
            return obj

    if isinstance(obj, ndarray) and (obj.dtype in [unicode_, string_]):
        new = obj.view(chararray)
        if unicode and obj.dtype == string_:
            return new.astype((unicode_, obj.itemsize))
        elif obj.dtype == unicode_:
            return new.astype((string_, obj.itemsize))

        if copy: return new.copy()
        else: return new

    if unicode: dtype = "U"
    else: dtype = "S"

    if itemsize is not None:
        dtype += str(itemsize)

    if isinstance(obj, (str, _unicode)):
        if itemsize is None:
            itemsize = len(obj)
        shape = len(obj) / itemsize
        return chararray(shape, itemsize=itemsize, unicode=unicode,
                         buffer=obj)

    # default
    val = narray(obj, dtype=dtype, order=order, subok=1)

    return val.view(chararray)

def asarray(obj, itemsize=None, unicode=False, order=None):
    return array(obj, itemsize, copy=False,
                 unicode=unicode, order=order)
