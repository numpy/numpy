from numerictypes import character, string, unicode_
from numeric import ndarray, issubclass_

class chararray(ndarray):
    def __new__(subtype, shape, dtype=string, buffer=None, offset=0,
                strides=None, swap=0, fortran=0, itemsize=1):
        if not issubclass_(dtype, character):
            raise ValueError, "dtype must be string or unicode_"

        


class CharArrayType:
    def __init__(self, itemsize):
        self.itemsize = itemsize
        self.name = "CharArrayType(%d)" % (self.itemsize,)

    def __repr__(self):
        return self.name

CharArrayTypeCache = {}

def NewCharArrayType(itemsize):
    """NewCharArrayType creates CharArrayTypes on demand, but checks to see
    if they already exist in the cache first.  This makes type equivalence
    the same as object identity.
    """
    if itemsize not in CharArrayTypeCache.keys():
        CharArrayTypeCache[itemsize] = CharArrayType(itemsize)
    return CharArrayTypeCache[itemsize]


class PrecisionWarning(UserWarning):
    pass


class RawCharArray(_na.NDArray):
    """RawCharArray(buffer=None, shape=None, byteoffset=0, bytestride=None)

      RawCharArray is a fixed length array of characters based on
      ndarray.NDArray with no automatic stripping or padding.

      itemsize specifies the length of all strings in the array.
    """
    _is_raw = 1
    def __init__(self, buffer=None, itemsize=None, shape=None, byteoffset=0,
                 bytestride=None, aligned=1, type=None, padc=" "):

        if isinstance(shape, types.IntType):
            shape = (shape,)

        if type is not None:
            if itemsize is not None:
                raise ValueError("Specify type *or* itemsize, not both.")
            itemsize = type.itemsize

        if not (padc, types.StringType) or len(padc) <> 1:
            raise ValueError("padc must be a string of length 1.")

        if buffer is None:
            if shape is None or itemsize is None:
                raise ValueError("Must define both shape & itemsize if buffer is None")
        else:
            if shape is None and itemsize is None:
                raise ValueError("Must specify shape, itemsize, or both.")
            ni = _bufferItems( buffer, byteoffset, bytestride, itemsize)
            if shape and itemsize == None:
                itemsize = ni/_na.product(shape)
            if itemsize and shape == None:
                shape = (ni,)
            if itemsize == 0:  # Another hack for 0 length strings.
                bytestride=0

        if not _nda.is_buffer(buffer) and buffer is not None:
            raise TypeError("buffer must either support the C buffer protocol or return something that does from its __buffer__() method");
        
        _na.NDArray.__init__(self, shape=shape, itemsize=itemsize,
                                 buffer=buffer, byteoffset=byteoffset,
                                 bytestride=bytestride, aligned=aligned)
        
        self._flags |= _gen._UPDATEDICT

        if type is None:
            type = NewCharArrayType(itemsize)

        self._type = type
        self._padc = padc
        
        if buffer is None:
            self.fill(" ")

    def __get_array_typestr__(self):
        return '|S%d' % self._itemsize

    __array_typestr__ = property(__get_array_typestr__, None, "")

    def tolist(self):
        """returns array as a (nested) list of strings."""
        if len(self._shape) == 1:
            if self._shape[0] > 0:
                return [ x for x in self ]
            else:
                return []
        else:
            return [ ni.tolist() for ni in self ]

    def __getstate__(self):
        """returns RawCharArray state dictionary for pickling"""
        state = _na.NDArray.__getstate__(self)
        state["_type"] = self._type.itemsize
        return state

    def __setstate__(self, state):
        """sets RawCharArray dictionary from pickled state"""
        _na.NDArray.__setstate__(self, state)
        self._type = NewCharArrayType(state["_type"])

    def isbyteswapped(self):
        """CharArray's are *never* byteswapped.  returns 0."""
        return 0

    def isaligned(self):
        """CharArray's are *always* aligned.  returns 1."""
        return 1

    def sinfo(self):
        "returns string describing a CharArray"
        s = _na.NDArray.sinfo(self)
        s += "type: " + repr(self._type) + "\n"
        return s

    def _getitem(self, offset):
        """_getitem(self, offset) returns  the "stripped" fixed length
        string from 'self' at 'offset'.
        """
        if isinstance(self._data, memory.MemoryType):
            s = buffer(self._data)[offset:offset+self._itemsize]
            return self.strip(s)
        else:
            return self.strip(str(self._data[offset:offset+self._itemsize]))

    def _setitem(self, offset, value):
        """_setitem(self, offset) sets 'offset' to result of "padding" 'value'.

        _setitem silently truncates inputs which are too long.

        >>> s=array([""])
        >>> s[0] = "this"
        >>> s
        CharArray([''])
        >>> s=array(["this","that"])
        >>> s[0] = "othe"
        >>> s
        CharArray(['othe', 'that'])
        >>> s[1] = "v"
        >>> s
        CharArray(['othe', 'v'])
        >>> s = array("")
        >>> s[0] = "this"
        >>> s
        CharArray([''])
        """
        bo = offset
        where = memory.writeable_buffer(self._data)
        where[bo:bo+self._itemsize] = self.pad(value)[0:self._itemsize]

    def _byteView(self):
        """_byteView(self) returns a view of self as an array of bytes.
        A _byteView cannot be taken from a chararray with itemsize==0.
        """
        if self._itemsize == 0:
            raise ValueError("_byteView doesn't work for zero length items.")
        b = _na.NumArray(buffer=self._data,
                              shape=self._shape+(self._itemsize,),
                              type=_na.UInt8,
                              byteoffset=self._byteoffset,
                              bytestride=self._bytestride)
        b._strides = self._strides + (1,)
        return b;

    def _copyFrom(self, arr):
        """
        >>> c = array([""])
        >>> c[:] = array(["this"])
        >>> c
        CharArray([''])
        >>> c = array(["this","that"])
        >>> c[:] = array(["a really long line","a"])
        >>> c
        CharArray(['a re', 'a'])
        >>> c[:] = ["more","money"]
        >>> c
        CharArray(['more', 'mone'])
        >>> c[:] = array(["x"])
        >>> c
        CharArray(['x', 'x'])
        """
        if self._itemsize == 0:
            return

        # Convert lists and strings to chararray.
        arr = asarray(arr, itemsize=self._itemsize,
                      padc=self._padc, kind=self.__class__)
        
        # Ensure shapes match.
        arr = self._broadcast( arr )
        if arr._itemsize == 0: return
        
        # Get views of both source and destination as UInt8 numarray.
        it = arr._byteView()
        me = self._byteView()
        if self._itemsize <= arr._itemsize:
            me[:] = it[..., :self._itemsize]
        else:
            me[...,:it._shape[-1]] = it
            # zero fill the part of subarr *not* covered by arr
            me[...,it._shape[-1]:] = 0

    def copy(self):
        """Return a new array with the same shape and type, but a copy
        of the data

        >>> c = fromlist(["this","that", "another"])
        >>> d = c.copy()
        >>> d
        CharArray(['this', 'that', 'another'])
        >>> int(c._data is d._data)
        0
        """
        
        arr = self.view()
        arr._data = memory.new_memory(arr._itemsize * arr.nelements())
        arr._byteoffset = 0
        arr._bytestride = arr._itemsize
        arr._strides = arr._stridesFromShape()
        arr._itemsize = self._itemsize
        if _na.product(self._shape):
            copyfunction = _bytes.functionDict["copyNbytes"]
            copyfunction(arr._shape, self._data, self._byteoffset,
                         self._strides, arr._data, 0, arr._strides,
                         arr._itemsize)
        return arr

    def substringView(self, i, j):
        """substringView returns modified view of the input array which
        represents only the [i:j] substring of each array element.

        >>> c = fromlist([["this","that"],["another", "one"]])
        >>> d = c.substringView(1, 2); d
        CharArray([['h', 'h'],
                   ['n', 'n']])
        >>> d[:] = [["1","2"],["3","4"]]; d
        CharArray([['1', '2'],
                   ['3', '4']])
        >>> c
        CharArray([['t1is', 't2at'],
                   ['a3other', 'o4e']])
        """

        n = _na.arange(self._itemsize)[i:j]
        if len(n) != 0:
            i = n[0]
            j = n[-1]+1
        else:
            i = j = 0
        r = self.view()
        substr_size = j-i
        if substr_size < 0:
            substr_size = 0
        r._itemsize = substr_size
        r._byteoffset += i
        return r
        
    def _broadcast(self, other):
        return _na.NDArray._broadcast(self, other)

    def _dualbroadcast(self, other):
        s, o = _na.NDArray._dualbroadcast(self, other)
        if not _na.product(s._strides):
            s = s.copy()
        if not _na.product(o._strides):
            o = o.copy()
        return s, o

    def concatenate(self, other):
        """concatenate(self, other) concatenates two numarray element by element
        >>> array(["this", "that", "another"]).stripAll() + "."
        CharArray(['this.', 'that.', 'another.'])
        >>> array([""])+array([""])
        CharArray([''])
        """
        a = asarray(other, padc=self._padc, kind=self.__class__)
        self, a = self._dualbroadcast(a)
        result = array(buffer=None, shape=self._shape,
                       itemsize=self._itemsize+a._itemsize,
                       padc=self._padc, kind=self.__class__)
        if a is other:  # since stripAll() mutates the array
            a = a.copy()
        _chararray.Concat(self.__class__ is RawCharArray,
                          self, a.stripAll(), result)
        return result.padAll()

    def __add__(self, other):
        """
        >>> map(str, range(3)) + array(["this","that","another one"])
        CharArray(['0this', '1that', '2another one'])
        >>> "" + array(["this", "that"])
        CharArray(['this', 'that'])
        >>> "prefix with trailing whitespace   " + array(["."])
        CharArray(['prefix with trailing whitespace   .'])
        >>> "" + array("")
        CharArray([''])
        >>> array(["this", "that", "another one"], kind=RawCharArray) + map(str, range(3))
        RawCharArray(['this       0', 'that       1', 'another one2'])
        """
        return self.concatenate(other)

    def __radd__(self, other):
        return asarray(other, padc=self._padc,
                       kind=self.__class__).concatenate(self)

    def __iadd__(self, other):
        self[:] = self.concatenate(other)
        return self

    def strip(self, value):
        return value

    def pad(self, value):
        return value

    def stripAll(self):
        return self

    def padAll(self):
        return self

    def _format(self, x):
        """_format() formats a single array element for str() or repr()"""
        return repr(self.strip(x))

    def __cmp__(self, other):
        s, t = str(self), str(other)
        return cmp(s,t)

    def fill(self, char):
        """fill(self, char)   fills the array entirely with 'char'.

        >>> x=array([""])
        >>> x.fill(' ')
        >>> x
        CharArray([''])
        >>> x=array(["this"])
        >>> x.fill("x")
        >>> x
        CharArray(['xxxx'])
        """
        if self._itemsize and self.nelements():
            if self.rank > 0:
                self[:] = char*self._itemsize
            else:
                self[()] =  char*self._itemsize

    def raw(self):
        """raw(self) returns a raw view of self.
        >>> c=fromlist(["this","that","another"])
        >>> c.raw()
        RawCharArray(['this   ', 'that   ', 'another'])
        """
        arr = self.view()
        arr.__class__ = RawCharArray    # "Anchor" on RawCharArray.
        return arr

    def contiguous(self):
        """contiguous(self) returns a version of self which is guaranteed to
        be contiguous.  If self is contiguous already, it returns self.
        Otherwise, it returns a copy of self.
        """
        if self.iscontiguous():
            return self
        else:
            return self.copy()

    def resized(self, n, fill='\0'):
        """resized(self, n) returns a copy of self, resized so that each
        item is of length n characters.  Extra characters are filled with
        the value of 'fill'. If self._itemsize == n, self is returned.

        >>> c = fromlist(["this","that","another"])
        >>> c._itemsize
        7
        >>> d=c.resized(20)
        >>> d
        CharArray(['this', 'that', 'another'])
        >>> d._itemsize
        20
        """
        if self._itemsize != n:
            ext = self.__class__(shape=self._shape, itemsize=n)
            ext.fill(fill)
            ext[:] = self
            return ext
        else:
            return self

    # First half of comparison operators,  slow version.
    def _StrCmp(self, mode, raw, other0):
        """StrCmp(self, other0)  calls strncmp on corresponding items of the
        two numarray, self and other.
        """
        if not isinstance(other0, self.__class__):
            other = asarray(other0, padc=self._padc, kind=self.__class__)
        else:
            other = other0
        if self._shape != other._shape:
            self, other = self._dualbroadcast(other)
        if self._itemsize < other._itemsize:
            self = self.resized(other._itemsize)
        elif other._itemsize < self._itemsize:
            other = other.resized(self._itemsize)
        if self is None:
            raise ValueError("Incompatible array dimensions")
        return _chararray.StrCmp(self, mode, raw, other)

    # rich comparisons (only works in Python 2.1 and later)
    def __eq__(self, other):
        """
        >>> array(["this ", "thar", "other"]).__eq__(array(["this", "that", "another"]))
        array([1, 0, 0], type=Bool)
        >>> array([""]).__eq__(array([""]))
        array([1], type=Bool)
        >>> array([""]).__eq__(array(["x"]))
        array([0], type=Bool)
        >>> array(["x"]).__eq__(array([""]))
        array([0], type=Bool)
        """
        return _chararray.StrCmp(self, 0, self._is_raw, other)

    def __ne__(self, other):
        """
        >>> s=array(["this ", "thar", "other"])
        >>> t=array(["this", "that", "another"])
        >>> s.__ne__(t)
        array([0, 1, 1], type=Bool)
        """
        return _chararray.StrCmp(self, 1, self._is_raw, other)

    def __lt__(self, other):
        """
        >>> s=array(["this ", "thar", "other"])
        >>> t=array(["this", "that", "another"])
        >>> s.__lt__(t)
        array([0, 1, 0], type=Bool)
        """
        return _chararray.StrCmp(self, 2, self._is_raw, other)

    def __gt__(self, other):
        """
        >>> s=array(["this ", "thar", "other"])
        >>> t=array(["this", "that", "another"])
        >>> s.__gt__(t)
        array([0, 0, 1], type=Bool)
        """
        return _chararray.StrCmp(self, 3, self._is_raw, other)

    def __le__(self, other):
        """
        >>> s=array(["this ", "thar", "other"])
        >>> t=array(["this", "that", "another"])
        >>> s.__le__(t)
        array([1, 1, 0], type=Bool)
        """
        return _chararray.StrCmp(self, 4, self._is_raw, other)

    def __ge__(self, other):
        """
        >>> s=array(["this ", "thar", "other"])
        >>> t=array(["this", "that", "another"])
        >>> s.__ge__(t)
        array([1, 0, 1], type=Bool)
        """
        return _chararray.StrCmp(self, 5, self._is_raw, other)

    if sys.version_info >= (2,1,0):
        def _test_rich_comparisons():
            """
            >>> s=array(["this ", "thar", "other"])
            >>> t=array(["this", "that", "another"])
            >>> s == t
            array([1, 0, 0], type=Bool)
            >>> s < t
            array([0, 1, 0], type=Bool)
            >>> s >= t
            array([1, 0, 1], type=Bool)
            >>> s <= t
            array([1, 1, 0], type=Bool)
            >>> s > t
            array([0, 0, 1], type=Bool)
            """
            pass

    def __contains__(self, str):
        """
        Returns 1 if-and-only-if 'self' has an element == to 'str'

        >>> s=array(["this ", "thar", "other"])
        >>> int("this" in s)
        1
        >>> int("tjt" in s)
        0
        >>> x=array([""])
        >>> int("this" in x)
        0
        >>> int("" in x)
        1
        """
        return _na.logical_or.reduce(_na.ravel(self.__eq__(str)))

    def sort(self):
        """
        >>> a=fromlist(["other","this","that","another"])
        >>> a.sort()
        >>> a
        CharArray(['another', 'other', 'that', 'this'])
        """
        l = self.tolist()
        l.sort()
        self[:] = fromlist(l)

    def argsort(self, axis=-1):
        """
        >>> a=fromlist(["other","that","this","another"])
        >>> a.argsort()
        array([3, 0, 1, 2])
        """
        if axis != -1:
            raise TypeError("CharArray.argsort() does not support the axis parameter.")
        ax = range(len(self))
        ax.sort(lambda x,y,z=self: cmp(z[x],z[y]))
        return _na.array(ax)

    def amap(self, f):
        """amap() returns the nested list which results from applying
        function 'f' to each element of 'self'.
        """
        if len(self._shape) == 1:
            return [f(i) for i in self]
        else:
            ans = []
            for i in self:
                ans.append(i.amap(f))
            return ans

    def match(self, pattern, flags=0):
        """
        >>> a=fromlist([["wo","what"],["wen","erewh"]])
        >>> a.match("wh[aebd]")
        (array([0]), array([1]))
        >>> a.match("none")
        (array([], type=Long), array([], type=Long))
        >>> b=array([""])
        >>> b.match("this")
        (array([], type=Long),)
        >>> b.match("")
        (array([0]),)
        """
        matcher = re.compile(pattern, flags).match
        l = lambda x, f=matcher: int(f(x) is not None)
        matches = _na.array(self.amap(l), type=_na.Bool)
        if len(matches):
            return _na.nonzero(matches)
        else:
            return ()

    def search(self, pattern, flags=0):
        """
        >>> a=fromlist([["wo","what"],["wen","erewh"]])
        >>> a.search("wh")
        (array([0, 1]), array([1, 1]))
        >>> a.search("1")
        (array([], type=Long), array([], type=Long))
        >>> b=array(["",""])
        >>> b.search("1")
        (array([], type=Long),)
        >>> b.search("")
        (array([0, 1]),)
        """
        searcher = re.compile(pattern, flags).search
        l = lambda x, f=searcher: int(f(x) is not None)
        matches = _na.array(self.amap(l), type=_na.Bool)
        if len(matches):
            return _na.nonzero(matches)
        else:
            return ()

    def grep(self, pattern, flags=0):
        """
        >>> a=fromlist([["who","what"],["when","where"]])
        >>> a.grep("whe")
        CharArray(['when', 'where'])
        """
        return _gen.take(self, self.match(pattern, flags), axis=(0,))

    def sub(self, pattern, replacement, flags=0, count=0):
        """
        >>> a=fromlist([["who","what"],["when","where"]])
        >>> a.sub("wh", "ph")
        >>> a
        CharArray([['pho', 'phat'],
                   ['phen', 'phere']])
        """
        cpat = re.compile(pattern, flags)
        l = lambda x, p=cpat, r=replacement, c=count: re.sub(p, r, x, c)
        self[:] = fromlist( self.amap(l) )

    def eval(self):
        """eval(self) converts CharArray 'self' into a NumArray.
        This is the original slow implementation based on a Python loop
        and the eval() function.

        >>> array([["1","2"],["3","4"]]).eval()
        array([[1, 2],
               [3, 4]])
        >>> try:
        ...    array([["1","2"],["3","other"]]).eval()
        ... except NameError:
        ...    pass
        """
        n = _na.array([ eval(x,{},{}) for x in _na.ravel(self)])
        n.setshape(self._shape)
        return n

    def fasteval(self, type=_na.Float64):

        """fasteval(self, type=Float64) converts CharArray 'self' into
        a NumArray of the specified type.  fasteval() can't convert
        complex arrays at all, and loses precision when converting
        UInt64 or Int64.

        >>> array([["1","2"],["3","4"]]).fasteval().astype('Long')
        array([[1, 2],
               [3, 4]])
        >>> try:
        ...    array([["1","2"],["3","other"]]).fasteval()
        ... except _chararray.error:
        ...    pass
        """
        n = _na.array(shape=self._shape, type=_na.Float64)
        type = _nt.getType(type)
        _chararray.Eval((), self, n);
        if type != _na.Float64:
	    if ((type is _na.Int64) or 
		(_numinclude.hasUInt64 and type is _na.UInt64)):
                warnings.warn("Loss of precision converting to 64-bit type.  Consider using eval().", PrecisionWarning)
            return n.astype(type)
        else:
            return n
        
class CharArray(RawCharArray):
    """
    >>> array("thisthatthe othe",shape=(4,),itemsize=4)
    CharArray(['this', 'that', 'the', 'othe'])
    >>> array("thisthatthe othe",shape=(4,),itemsize=4)._shape
    (4,)
    >>> array("thisthatthe othe",shape=(4,),itemsize=4)._itemsize
    4
    >>> array([["this","that"],["x","y"]])
    CharArray([['this', 'that'],
               ['x', 'y']])
    >>> array([["this","that"],["x","y"]])._shape
    (2, 2)
    >>> array([["this","that"],["x","y"]])._itemsize
    4
    >>> s=array([["this","that"],["x","y"]], itemsize=10)
    >>> s
    CharArray([['this', 'that'],
               ['x', 'y']])
    >>> s._itemsize
    10
    >>> s[0][0]
    'this'
    >>> # s[1,1][0] = 'z' # Char assigment doesn't work!
    >>> s[1,1] = 'z'      # But padding may do what you want.
    >>> s                 # Otherwise, specify all of s[1,1] or subclass.
    CharArray([['this', 'that'],
               ['x', 'z']])
    """

    _is_raw = 0
    
    def resized(self, n, fill=' '):
        """Same as RawCharArray.resized() but fills with blanks rather than
        NUL."""        
        return RawCharArray.resized(self, n, fill)
    
    def pad(self, value):
        """
        pad(self, value)   implements CharArray's string-filling policy
        which is used when strings are assigned to elements of a CharArray.
        Pad extends 'value' to length self._itemsize using spaces.
        """
        return _chararray.Pad(value, self._itemsize, ord(self._padc))

    def strip(self, value):
        """
        strip(self, value) implements CharArray's string fetching
        "cleanup" policy. strip truncates 'value' at the first NULL
        and removes all trailing whitespace from the remainder.  For
        compatability with FITS, leading whitespace is never
        completely stripped: a string beginning with a space always
        returns at least one space to distinguish it from the empty
        string.
        """
        return _chararray.Strip(value)

    def stripAll(self):
        """
        stripAll(self) applies the chararray strip function to each element
        of self and returns the result.  The result may be a new array.
        """
        _chararray.StripAll(None, self)
        return self

    def padAll(self):
        """
        padAll(self) applies the chararray pad function to each element
        of self and returns the result.  The result may be a new array.
        """
        _chararray.PadAll(self._padc, self)
        return self

    def toUpper(self):
        """toUpper(self) converts all elements of self to upper case

        >>> a = fromlist(["That","this","another"])
        >>> a.toUpper()
        >>> a
        CharArray(['THAT', 'THIS', 'ANOTHER'])
        """
        _chararray.ToUpper(None, self)

    def toLower(self):
        """toLower(self) converts all elements of self to upper case

        >>> a = fromlist(["THAT","this","anOther"])
        >>> a.toLower()
        >>> a
        CharArray(['that', 'this', 'another'])
        """
        _chararray.ToLower(None, self)

    def maxLen(self):
        """
        maxLen(self) computes the length of the longest string in
        self.  Maxlen will applies the strip function to each element
        prior to computing its length.

        >>> array(["this  ","that"]).maxLen()
        4
        """
        n = _na.NumArray(buffer=None, shape=self.shape, type=_na.Int32)
        _chararray.StrLen(None, self, n)
        return n.max()

    def truncated(self):
        """
        truncate(self) returns a new array with the smallest possible itemsize
        which will hold the stripped contents of self.

        >>> array(["this  ","that"])._itemsize
        6
        >>> array(["this  ","that"]).truncated()._itemsize
        4
        """
        return self.resized(self.maxLen())

    def count(self, s):
        """count(self, s) counts the number of occurences of string 's'.
        >>> int(array(["this","that","another","this"]).count("this"))
        2
        """
        return self.__eq__(s).sum('Int64')

    def index(self, s):
        """index(self, s) returns the index of the first occurenced of
        's' in 'self'.

        >>> array([["this","that","another"],
        ...        ["another","this","that"]]).index("another")
        (0, 2)
        >>> array([["this","that","another"],
        ...        ["another","this","that"]]).index("not here")
        Traceback (most recent call last):
        ValueError: string 'not here' not in array

        """
        indices = _na.nonzero(self.__eq__(s))
        if len(indices[0]):
            first = map(lambda x: x[0], indices)
            return tuple(first)
        else:
            raise ValueError("string " + `s` +" not in array")

def isString(s):
    return isinstance(s, types.StringType)

def isPySequence(s):
    return hasattr(s, '__getitem__') and hasattr(s,'__len__')

def _slistShape0(slist):
    """_slistShape0(slist) computes the (shape+(itemsize,)) tuple
    of string list 'slist'.

    itemsize is set to the maximum of all string lengths in slist.

    >>> s=["this","that","the other"]
    >>> _slistShape(s)
    ((3,), 9)
    >>> _slistShape((s,s,s,s))
    ((4, 3), 9)
    >>> _slistShape(["this", ["that","other"]])
    Traceback (most recent call last):
    ...
    ValueError: Nested sequences with different lengths.
    """
    if isinstance(slist, types.StringType):
        return ((), len(slist),)
    elif len(slist) == 0:
        return ((0,), 0)
    else:
        maxs = _slistShape0(slist[0])
        sizes = {}
        for s in slist:
            if isinstance(s, types.StringType):
                maxs = max(maxs, ((), len(s)))
                sizes[1] = 1  # ignore 
            else:
                maxs = max(maxs, _slistShape0(s))
                sizes[len(s)] = 1
        if len(sizes.keys()) != 1:
            raise ValueError("Nested sequences with different lengths.")
        return (((len(slist),)+ maxs[0]), maxs[1])

def _slistShape(slist, itemsize=None, shape=None):
    """_slistShape(slist, itemsize=None, shape=None)  computes the "natural"
    shape and itemsize of slist, and combines this with the specified
    itemsize and shape,  checking for consistency.

    Specifying an itemsize overrides the slist's natural itemsize.

    >>> _slistShape(["this","that"], itemsize=10)
    ((2,), 10)
    >>> _slistShape(["this","that"], itemsize=3)
    ((2,), 3)

    Specifying a shape checks for consistency against the slist's shape.

    >>> _slistShape(["this","that"], shape=(2,1,1))
    ((2, 1, 1), 4)
    >>> _slistShape(["this","that"], shape=(3,2))
    Traceback (most recent call last):
    ...    
    ValueError: Inconsistent list and shape
    
    """
    shape_items = _slistShape0(slist)
    if shape is None:
        shape = shape_items[0]
    else:
        if _gen.product(shape) != _gen.product(shape_items[0]):
            raise ValueError("Inconsistent list and shape")
    if itemsize is None:
        itemsize = shape_items[-1]
    else:
        pass # specified itemsize => padded extension or silent truncation.
    return (shape, itemsize)

def _pad(slist, n, c=" "):
    """_pad(slist, n, c=' ') pads each member of string list 'slist' with
    fill character 'c' to a total length of 'n' characters and returns
    the concatenated results.

    strings longer than n are *truncated*.

    >>>
    >>> _pad(["this","that","the other"],9," ")
    'this     that     the other'
    """
    if isinstance(slist, types.StringType):
        if n > len(slist):
            return slist + c*(n-len(slist))
        else:
            return slist[:n]
    else:
        result = []
        for s in slist:
            if isinstance(s, types.StringType):
                if n > len(s):
                    t = s + c*(n-len(s))
                else:
                    t = s[:n]
            else:
                t = _pad(s, n, c)
            result.append(t)
        return "".join(result)

def fromlist(slist, itemsize=None, shape=None, padc=" ", kind=CharArray):
    """fromlist(slist, padc=" ") creates a CharArray from a multi-dimensional
    list of strings, 'slist', padding each string to the length of the
    longest string with character 'padc'.

    >>> s=fromlist([["this","that"],["x","y"]])
    >>> s
    CharArray([['this', 'that'],
               ['x', 'y']])
    >>> s[0][0]
    'this'
    >>> s[1][1]
    'y'
    >>> s[1][1] = "whom"
    >>> s[1][1]
    'whom'
    >>> fromlist(['this', 'that'], itemsize=2)
    CharArray(['th', 'th'])
    >>> fromlist(['t','u'], itemsize=3)
    CharArray(['t', 'u'])
    >>> fromlist(['t','u'], itemsize=3)._itemsize
    3
    """
    slist = list(slist)  # convert tuples
    shape, itemsize = _slistShape(slist, itemsize=itemsize, shape=shape)
    s = _pad(slist, itemsize, padc)  # compute padded concatenation of slist
    return fromstring(s, shape=shape, itemsize=itemsize, padc=padc, kind=kind)

def _stringToBuffer(datastring):
    """_stringToBuffer(datastring)  allocates a buffer, copies datastring into
    it, and returns the buffer.
    """
    abuff = memory.new_memory( len(datastring) )
    memory.writeable_buffer(abuff)[:] = datastring
    return abuff

def fromstring(s, itemsize=None, shape=None, padc=" ", kind=CharArray):
    """Create an array from binary data contained in a string (by copying)
    >>> fromstring('thisthat', itemsize=4)
    CharArray(['this', 'that'])
    >>> fromstring('thisthat', shape=2)
    CharArray(['this', 'that'])
    >>> fromstring('this is a test', shape=(1,))
    CharArray(['this is a test'])
    """
    if isinstance(shape, types.IntType):
        shape = (shape,)
    if ((shape in [None, (1,)])
        and (itemsize is not None
             and itemsize > len(s))):
        s = _pad(s, itemsize, padc)
    if shape is None and not itemsize:
        shape = (1,)
        itemsize = len(s)
    return kind(_stringToBuffer(s), shape=shape, itemsize=itemsize, padc=padc)


def fromfile(file, itemsize=None, shape=None, padc=" ", kind=CharArray):
    """Create an array from binary file data

    If file is a string then that file is opened, else it is assumed
    to be a file object. No options at the moment, all file positioning
    must be done prior to this function call with a file object

    >>> import testdata
    >>> s=fromfile(testdata.filename, shape=-1, itemsize=80)
    >>> s[0]
    'SIMPLE  =                    T / file does conform to FITS standard'
    >>> s._shape
    (108,)
    >>> s._itemsize
    80
    """
    if isinstance(shape, types.IntType):
        shape = (shape,)

    name =  0
    if isString(file):
        name = 1
        file = open(file, 'rb')
    size = int(os.path.getsize(file.name) - file.tell())

    if not shape and not itemsize:
        shape = (1,)
        itemsize = size
    elif shape is not None:
        if itemsize is not None:
            shapesize = _na.product(shape)*itemsize
            if shapesize < 0:
                shape = list(shape)
                shape[ shape.index(-1) ] = size / -shapesize
                shape = tuple(shape)
        else:
            shapesize=_na.product(shape)
            if shapesize < 0:
                raise ValueError("Shape dimension of -1 requires itemsize.")
            itemsize = size / shapesize
    elif itemsize:
        shape = (size/itemsize,)
    else:
        raise ValueError("Must define shape or itemsize.")

    nbytes = _na.product(shape)*itemsize

    if nbytes > size:
        raise ValueError(
                "Not enough bytes left in file for specified shape and type")

    # create the array
    arr = kind(None, shape=shape, itemsize=itemsize, padc=padc)
    nbytesread = file.readinto(arr._data)
    if nbytesread != nbytes:
        raise IOError("Didn't read as many bytes as expected")
    if name:
        file.close()
    return arr

def array(buffer=None, itemsize=None, shape=None, byteoffset=0,
          bytestride=None, padc=" ", kind=CharArray):
    """array(buffer=None, itemsize=None, shape=None) creates a new instance
    of a CharArray.

    buffer      specifies the source of the array's initialization data.
                type(buffer) is in [ None, CharArray, StringType,
                                     ListType, FileType, BufferType].

    itemsize    specifies the fixed maximum length of the array's strings.

    shape       specifies the array dimensions.


    >>> array(None,itemsize=3,shape=(1,))
    CharArray([' '])
    >>> array(buffer("abcdsedxxxxxncxn"), itemsize=2, byteoffset=4, bytestride=8)
    CharArray(['se', 'nc'])
    >>> array("abcd", itemsize=2)
    CharArray(['ab', 'cd'])
    >>> array(['this', 'that'], itemsize=10)
    CharArray(['this', 'that'])
    >>> array(['this', 'that'], itemsize=10)._itemsize
    10
    >>> array(array(['this', 'that'], itemsize=10))
    CharArray(['this', 'that'])
    >>> import testdata
    >>> array(open(testdata.filename,"r"),itemsize=80, shape=2)
    CharArray(['SIMPLE  =                    T / file does conform to FITS standard',
               'BITPIX  =                   16 / number of bits per data pixel'])
    """
    if isinstance(shape, types.IntType):
        shape = (shape,)

    if buffer is None or _na.SuitableBuffer(buffer):
        return kind(buffer, itemsize=itemsize, shape=shape,
                    byteoffset=byteoffset, bytestride=bytestride,
                    padc=padc)

    if byteoffset or bytestride is not None:
        raise ValueError('Offset and stride can only be specified if "buffer" is a buffer or None')

    if isString(buffer):
        return fromstring(buffer, itemsize=itemsize, shape=shape,
                          padc=padc, kind=kind)
    elif isPySequence(buffer):
        return fromlist(buffer, itemsize=itemsize, shape=shape,
                        padc=padc, kind=kind)
    elif isinstance(buffer, kind) and buffer.__class__ is kind:
        return buffer.copy()
    elif isinstance(buffer, RawCharArray):
        return kind(buffer=buffer._data,
                    itemsize=itemsize or buffer._itemsize,
                    shape=shape or buffer._shape,
                    padc=padc)
    elif isinstance(buffer, types.FileType):
        return fromfile(buffer, itemsize=itemsize, shape=shape,
                        padc=padc, kind=kind)
    else:
        raise TypeError("Don't know how to handle that kind of buffer")

def asarray(buffer=None, itemsize=None, shape=None, byteoffset=0,
            bytestride=None, padc=" ", kind=CharArray):
    """massages a sequence into a chararray.

    If buffer is *already* a chararray of the appropriate kind, it is
    returned unaltered.
    """
    if isinstance(buffer, kind) and buffer.__class__ is kind:
        return buffer
    else:
        return array(buffer, itemsize, shape, byteoffset, bytestride,
                     padc, kind)

inputarray = asarray  # obosolete synonym

def take(array, indices, outarr=None, axis=0, clipmode=_na.RAISE):
    a = asarray(array)
    return _gen.take(a, indices, outarr, axis, clipmode)
take.__doc__ = _gen.take.__doc__

def put(array, indices, values, axis=0, clipmode=_na.RAISE):
    a = asarray(array)
    v = asarray(values)
    return _gen.put(a, indices, v, axis, clipmode)
put.__doc__ = _gen.put.__doc__

def _bufferItems(buffer, offset=0, bytestride=None, itemsize=None):
    """
    >>> _bufferItems(buffer("0123456789"), offset=2, bytestride=4, itemsize=1)
    2
    >>> _bufferItems(buffer("0123456789"), offset=1, bytestride=2, itemsize=1)
    5
    >>> _bufferItems(buffer("0123456789"), bytestride=5, itemsize=2)
    2
    >>> _bufferItems(buffer("0123456789"), bytestride=None, itemsize=2)
    5
    >>> _bufferItems(buffer("0123456789"), offset=3)
    7
    >>> _bufferItems(buffer("abcdsedxxxxxncxn"),itemsize=2,offset=4,bytestride=8)
    2
    >>> cc=CharArray(buffer('abcdef'*5),shape=(2,),itemsize=2,byteoffset=8,bytestride=20,aligned=0)
    >>> cc
    CharArray(['cd', 'ef'])
    """
    if bytestride is None:
        if itemsize is None:
            return len(buffer) - offset
        else:
            if itemsize:
                return (len(buffer)-offset)/itemsize
            else:
                return 1 # Hack to permit 0 length strings
    else:
        if itemsize is None:
            raise ValueError("Must specify itemsize if bytestride is specified.")
        strides = (len(buffer)-offset)*1.0/bytestride
        istrides = int(strides)
        fstrides = int((strides-istrides)*bytestride+0.5)
        return istrides +  (fstrides >= itemsize)

def num2char(n, format, itemsize=32):
    """num2char formats NumArray 'num' into a CharArray using 'format'

    >>> num2char(_na.arange(0.0,5), '%2.2f')
    CharArray(['0.00', '1.00', '2.00', '3.00', '4.00'])
    >>> num2char(_na.arange(0.0,5), '%d')
    CharArray(['0', '1', '2', '3', '4'])
    >>> num2char(_na.arange(5), "%02d")
    CharArray(['00', '01', '02', '03', '04'])

    Limitations:
    
    1. When formatted values are too large to fit into strings of
    length itemsize, the values are truncated, possibly losing
    significant information.

    2. Complex numbers are not supported.
    
    """
    n = _na.asarray(n)

    if isinstance(n.type(), _nt.ComplexType):
        raise NotImplementedError("num2char doesn't support complex types yet.")
    if n.type() == _na.Float64:
        wnum = n
    else:
        wnum = n.astype(_na.Float64)
    char = CharArray(shape=n.getshape(), itemsize=itemsize)
    _chararray.Format(format, wnum, char)
    return char

def _nothing(*args):  return 0  # Test everything... nothing private.

def test():
    if sys.version_info < (2,4):
        import doctest, strings
        return doctest.testmod(strings, isprivate=_nothing)
    else:
        import numarray.numtest as nt, strings
        t  = nt.Tester(globs=globals())
        t.rundoc(strings)
        return t.summarize()

if __name__ == "__main__":
    test()
