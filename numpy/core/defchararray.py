"""
Module for character arrays.

.. note::
   The chararray module exists for backwards compatibility with Numarray,
   it is not recommended for new development. If one needs arrays of
   strings, use arrays of `dtype` object.

The preferred alias for `defchararray` is `numpy.char`.

"""
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

    An array of fixed size (perhaps unicode) strings.

    .. note::
       The chararray module exists for backwards compatibility with Numarray,
       it is not recommended for new development. If one needs arrays of
       strings, use arrays of `dtype` object.

    Create the array, using `buffer` (with `offset` and `strides`) if it is
    not ``None``. If `buffer` is ``None``, then construct a new array with
    `strides` in "C order," unless both ``len(shape) >= 2`` and
    ``order='Fortran'``, in which case `strides` is in "Fortran order."

    Parameters
    ----------
    shape : tuple
        Shape of the array.

    itemsize : int_like, > 0, optional
        Length of each array element, in number of characters. Default is 1.

    unicode : {True, False}, optional
        Are the array elements of unicode-type (``True``) or string-type
        (``False``, the default).

    buffer : integer, > 0, optional
        Memory address of the start of the array data.  If ``None`` (the
        default), a new array is created.

    offset : integer, >= 0, optional
        Fixed stride displacement from the beginning of an axis? Default is
        0.

    strides : array_like(?), optional
        Strides for the array (see `numpy.ndarray.strides` for full
        description), default is ``None``.

    order : {'C', 'F'}, optional
        The order in which the array data is stored in memory: 'C' -> "row
        major" order (the default), 'F' -> "column major" (Fortran) order

    Examples
    --------
    >>> charar = np.chararray((3, 3))
    >>> charar[:,:] = 'abc'
    >>> charar
    chararray([['a', 'a', 'a'],
           ['a', 'a', 'a'],
           ['a', 'a', 'a']],
          dtype='|S1')
    >>> charar = np.chararray(charar.shape, itemsize=5)
    >>> charar[:,:] = 'abc'
    >>> charar
    chararray([['abc', 'abc', 'abc'],
           ['abc', 'abc', 'abc'],
           ['abc', 'abc', 'abc']],
          dtype='|S5')

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
        """
        Return the indices that sort the array lexicographically.

        For full documentation see `numpy.argsort`, for which this method is
        in fact merely a "thin wrapper."

        Examples
        --------
        >>> c = np.array(['a1b c', '1b ca', 'b ca1', 'Ca1b'], 'S5')
        >>> c = c.view(np.chararray); c
        chararray(['a1b c', '1b ca', 'b ca1', 'Ca1b'],
              dtype='|S5')
        >>> c[c.argsort()]
        chararray(['1b ca', 'Ca1b', 'a1b c', 'b ca1'],
              dtype='|S5')

        """
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
        """
        Capitalize the first character of each array element.

        For each element of `self`, if the first character is a letter
        possessing both "upper-case" and "lower-case" forms, and it is
        presently in lower-case, change it to upper-case; otherwise, leave
        it untouched.

        Parameters
        ----------
        None

        Returns
        -------
        ret : chararray
            `self` with each element "title-cased."

        Examples
        --------
        >>> c = np.array(['a1b2','1b2a','b2a1','2a1b'],'S4').view(np.chararray); c
        chararray(['a1b2', '1b2a', 'b2a1', '2a1b'],
              dtype='|S4')
        >>> c.capitalize()
        chararray(['A1b2', '1b2a', 'B2a1', '2a1b'],
              dtype='|S4')

        """
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
        """
        Return the number of occurrences of a sub-string in each array element.

        Parameters
        ----------
        sub : string
            The sub-string to count.
        start : int, optional
            The string index at which to start counting in each element.
        end : int, optional
            The string index at which to end counting in each element.

        Returns
        -------
        ret : ndarray of ints
            Array whose elements are the number of occurrences of `sub` in each
            element of `self`.

        Examples
        --------
        >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba']).view(np.chararray)
        >>> c
        chararray(['aAaAaA', '  aA', 'abBABba'],
              dtype='|S7')
        >>> c.count('A')
        array([3, 1, 1])
        >>> c.count('aA')
        array([3, 1, 0])
        >>> c.count('A', start=1, end=4)
        array([2, 1, 1])
        >>> c.count('A', start=1, end=3)
        array([1, 0, 0])

        """
        return self._typedmethod('count', broadcast(self, sub, start, end), int)

    def decode(self,encoding=None,errors=None):
        """
        Return elements decoded according to the value of `encoding`.

        Parameters
        ----------
        encoding : string, optional
            The encoding to use; for a list of acceptable values, see the
            Python docstring for the package 'encodings'
        error : Python exception object?, optional
            The exception to raise if decoding fails?

        Returns
        -------
        ret : chararray
            A view of `self`, suitably decoded.

        See Also
        --------
        encode
        encodings
            (package)

        Examples
        --------
        >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba']).view(np.chararray)
        >>> c
        chararray(['aAaAaA', '  aA', 'abBABba'],
              dtype='|S7')
        >>> c = c.encode(encoding='cp037'); c
        chararray(['\\x81\\xc1\\x81\\xc1\\x81\\xc1', '@@\\x81\\xc1@@',
               '\\x81\\x82\\xc2\\xc1\\xc2\\x82\\x81'],
              dtype='|S7')
        >>> c.decode(encoding='cp037')
        chararray(['aAaAaA', '  aA', 'abBABba'],
              dtype='|S7')

        """
        return self._generalmethod('decode', broadcast(self, encoding, errors))

    def encode(self,encoding=None,errors=None):
        """
        Return elements encoded according to the value of `encoding`.

        Parameters
        ----------
        encoding : string, optional
            The encoding to use; for a list of acceptable values, see the
            Python docstring for `encodings`.
        error : Python exception object, optional
            The exception to raise if encoding fails.

        Returns
        -------
        ret : chararray
            A view of `self`, suitably encoded.

        See Also
        --------
        decode

        Examples
        --------
        >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba']).view(np.chararray)
        >>> c
        chararray(['aAaAaA', '  aA', 'abBABba'],
              dtype='|S7')
        >>> c.encode(encoding='cp037')
        chararray(['\\x81\\xc1\\x81\\xc1\\x81\\xc1', '@@\\x81\\xc1@@',
               '\\x81\\x82\\xc2\\xc1\\xc2\\x82\\x81'],
              dtype='|S7')

        """
        return self._generalmethod('encode', broadcast(self, encoding, errors))

    def endswith(self, suffix, start=None, end=None):
        """
        Check whether elements end with specified suffix

        Given an array of strings, return a new bool array of same shape with
        the result of comparing suffix against each element; each element
        of bool array is ``True`` if element ends with specified suffix and
        ``False`` otherwise.

        Parameters
        ----------
        suffix : string
            Compare each element in array to this.
        start : int, optional
            For each element, start searching from this position.
        end : int, optional
            For each element, stop comparing at this position.

        Returns
        -------
        endswith : ndarray
            Output array of bools

        See Also
        --------
        count
        find
        index
        startswith

        Examples
        --------
        >>> s = chararray(3, itemsize=3)
        >>> s[0] = 'foo'
        >>> s[1] = 'bar'
        >>> s
        chararray(['foo', 'bar'],
              dtype='|S3')
        >>> s.endswith('ar')
        array([False,  True], dtype=bool)
        >>> s.endswith('a', start=1, end=2)
        array([False,  True], dtype=bool)

        """
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
        """
        Assure that every character of each array element is lower-case.

        For each character possessing both "upper-case" and "lower-case" forms,
        if it is in upper-case, change it to lower; otherwise, leave it unchanged.

        Parameters
        ----------
        None

        Returns
        -------
        ret : chararray
            `self` with all capital letters changed to lower-case.

        Examples
        --------
        >>> c = np.array(['A1B C', '1BCA', 'BCA1']).view(np.chararray); c
        chararray(['A1B C', '1BCA', 'BCA1'],
              dtype='|S5')
        >>> c.lower()
        chararray(['a1b c', '1bca', 'bca1'],
              dtype='|S5')

        """
        return self._samemethod('lower')

    def lstrip(self, chars):
        """
        Remove leading characters from each element.

        Returns a view of ``self`` with `chars` stripped from the start of
        each element.  Note: **No Default** - `chars` must be specified (but if
        it is explicitly ``None`` or the empty string '', leading whitespace is
        removed).

        Parameters
        ----------
        chars : string_like or None
            Character(s) to strip; whitespace stripped if `chars` == ``None``
            or `chars` == ''.

        Returns
        -------
        ret : chararray
            View of ``self``, each element suitably stripped.

        Raises
        ------
        TypeError: lstrip() takes exactly 2 arguments (1 given)
            If `chars` is not supplied.

        Examples
        --------
        >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba']).view(np.chararray)
        >>> c
        chararray(['aAaAaA', '  aA', 'abBABba'],
              dtype='|S7')
        >>> c.lstrip('a') # 'a' unstripped from c[1] because whitespace leading
        chararray(['AaAaA', '  aA', 'bBABba'],
              dtype='|S6')
        >>> c.lstrip('A') # leaves c unchanged
        chararray(['aAaAaA', '  aA', 'abBABba'],
              dtype='|S7')
        >>> (c.lstrip(' ') == c.lstrip('')).all()
        True
        >>> (c.lstrip(' ') == c.lstrip(None)).all()
        True

        """
        return self._generalmethod('lstrip', broadcast(self, chars))

    def replace(self, old, new, count=None):
        return self._generalmethod('replace', broadcast(self, old, new, count))

    def rfind(self, sub, start=None, end=None):
        return self._typedmethod('rfind', broadcast(self, sub, start, end), int)

    def rindex(self, sub, start=None, end=None):
        return self._typedmethod('rindex', broadcast(self, sub, start, end), int)

    def rstrip(self, chars=None):
        """
        Remove trailing characters.

        Returns a view of ``self`` with `chars` stripped from the end of each
        element.

        Parameters
        ----------
        chars : string_like, optional, default=None
            Character(s) to remove.

        Returns
        -------
        ret : chararray
            View of ``self``, each element suitably stripped.

        Examples
        --------
        >>> c = np.array(['aAaAaA', 'abBABba'], dtype='S7').view(np.chararray); c
        chararray(['aAaAaA', 'abBABba'],
              dtype='|S7')
        >>> c.rstrip('a')
        chararray(['aAaAaA', 'abBABb'],
              dtype='|S6')
        >>> c.rstrip('A')
        chararray(['aAaAa', 'abBABba'],
              dtype='|S7')

        """
        return self._generalmethod('rstrip', broadcast(self, chars))

    def split(self, sep=None, maxsplit=None):
        return self._typedmethod('split', broadcast(self, sep, maxsplit), object)

    def splitlines(self, keepends=None):
        return self._typedmethod('splitlines', broadcast(self, keepends), object)

    def startswith(self, prefix, start=None, end=None):
        return self._typedmethod('startswith', broadcast(self, prefix, start, end), bool)

    def strip(self, chars=None):
        """
        Remove leading and trailing characters, whitespace by default.

        Returns a view of ``self`` with `chars` stripped from the start and end of
        each element; by default leading and trailing whitespace is removed.

        Parameters
        ----------
        chars : string_like, optional, default=None
            Character(s) to strip; whitespace by default.

        Returns
        -------
        ret : chararray
            View of ``self``, each element suitably stripped.

        Examples
        --------
        >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba']).view(np.chararray)
        >>> c
        chararray(['aAaAaA', '  aA', 'abBABba'],
              dtype='|S7')
        >>> c.strip()
        chararray(['aAaAaA', 'aA', 'abBABba'],
              dtype='|S7')
        >>> c.strip('a') # 'a' unstripped from c[1] because whitespace leads
        chararray(['AaAaA', '  aA', 'bBABb'],
              dtype='|S6')
        >>> c.strip('A') # 'A' unstripped from c[1] because (unprinted) ws trails
        chararray(['aAaAa', '  aA', 'abBABba'],
              dtype='|S7')

        """
        return self._generalmethod('strip', broadcast(self, chars))

    def swapcase(self):
        """
        Switch upper-case letters to lower-case, and vice-versa.

        Parameters
        ----------
        None

        Returns
        -------
        ret : chararray
            `self` with all lower-case letters capitalized and all upper-case
            changed to lower case.

        Examples
        --------
        >>> c=np.array(['a1B c','1b Ca','b Ca1','cA1b'],'S5').view(np.chararray);c
        chararray(['a1B c', '1b Ca', 'b Ca1', 'cA1b'],
              dtype='|S5')
        >>> c.swapcase()
        chararray(['A1b C', '1B cA', 'B cA1', 'Ca1B'],
              dtype='|S5')

        """
        return self._samemethod('swapcase')

    def title(self):
        """
        Capitalize the first character of each array element.

        For each element of `self`, if the first character is a letter
        possessing both "upper-case" and "lower-case" forms, and it is
        presently in lower-case, change it to upper-case; otherwise, leave
        it untouched.

        Parameters
        ----------
        None

        Returns
        -------
        ret : chararray
            `self` with

        Examples
        --------
        >>> c=np.array(['a1b c','1b ca','b ca1','ca1b'],'S5').view(np.chararray);c
        chararray(['a1b c', '1b ca', 'b ca1', 'ca1b'],
              dtype='|S5')
        >>> c.title()
        chararray(['A1B C', '1B Ca', 'B Ca1', 'Ca1B'],
              dtype='|S5')

        """
        return self._samemethod('title')

    def translate(self, table, deletechars=None):
        if self.dtype is unicode_:
            return self._generalmethod('translate', broadcast(self, table))
        else:
            return self._generalmethod('translate', broadcast(self, table, deletechars))

    def upper(self):
        """
        Capitalize every character of each array element.

        For each character possessing both "upper-case" and "lower-case" forms,
        if it is in lower-case, change it to upper; otherwise, leave it unchanged.

        Parameters
        ----------
        None

        Returns
        -------
        ret : chararray
            `self` with all characters capitalized.

        Examples
        --------
        >>> c = np.array(['a1b c', '1bca', 'bca1']).view(np.chararray); c
        chararray(['a1b c', '1bca', 'bca1'],
              dtype='|S5')
        >>> c.upper()
        chararray(['A1B C', '1BCA', 'BCA1'],
              dtype='|S5')

        """
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
