#A place for code to be called from C-code
#  that implements more complicated stuff.

import re
import sys

from numpy.compat import asbytes, bytes

if (sys.byteorder == 'little'):
    _nbo = asbytes('<')
else:
    _nbo = asbytes('>')

def _makenames_list(adict):
    from multiarray import dtype
    allfields = []
    fnames = adict.keys()
    for fname in fnames:
        obj = adict[fname]
        n = len(obj)
        if not isinstance(obj, tuple) or n not in [2,3]:
            raise ValueError("entry not a 2- or 3- tuple")
        if (n > 2) and (obj[2] == fname):
            continue
        num = int(obj[1])
        if (num < 0):
            raise ValueError("invalid offset.")
        format = dtype(obj[0])
        if (format.itemsize == 0):
            raise ValueError("all itemsizes must be fixed.")
        if (n > 2):
            title = obj[2]
        else:
            title = None
        allfields.append((fname, format, num, title))
    # sort by offsets
    allfields.sort(key=lambda x: x[2])
    names = [x[0] for x in allfields]
    formats = [x[1] for x in allfields]
    offsets = [x[2] for x in allfields]
    titles = [x[3] for x in allfields]

    return names, formats, offsets, titles

# Called in PyArray_DescrConverter function when
#  a dictionary without "names" and "formats"
#  fields is used as a data-type descriptor.
def _usefields(adict, align):
    from multiarray import dtype
    try:
        names = adict[-1]
    except KeyError:
        names = None
    if names is None:
        names, formats, offsets, titles = _makenames_list(adict)
    else:
        formats = []
        offsets = []
        titles = []
        for name in names:
            res = adict[name]
            formats.append(res[0])
            offsets.append(res[1])
            if (len(res) > 2):
                titles.append(res[2])
            else:
                titles.append(None)

    return dtype({"names" : names,
                  "formats" : formats,
                  "offsets" : offsets,
                  "titles" : titles}, align)


# construct an array_protocol descriptor list
#  from the fields attribute of a descriptor
# This calls itself recursively but should eventually hit
#  a descriptor that has no fields and then return
#  a simple typestring

def _array_descr(descriptor):
    from multiarray import METADATA_DTSTR
    fields = descriptor.fields
    if fields is None:
        subdtype = descriptor.subdtype
        if subdtype is None:
            if descriptor.metadata is None:
                return descriptor.str
            else:
                new = descriptor.metadata.copy()
                # Eliminate any key related to internal implementation
                _ = new.pop(METADATA_DTSTR, None)
                return (descriptor.str, new)
        else:
            return (_array_descr(subdtype[0]), subdtype[1])


    names = descriptor.names
    ordered_fields = [fields[x] + (x,) for x in names]
    result = []
    offset = 0
    for field in ordered_fields:
        if field[1] > offset:
            num = field[1] - offset
            result.append(('','|V%d' % num))
            offset += num
        if len(field) > 3:
            name = (field[2],field[3])
        else:
            name = field[2]
        if field[0].subdtype:
            tup = (name, _array_descr(field[0].subdtype[0]),
                   field[0].subdtype[1])
        else:
            tup = (name, _array_descr(field[0]))
        offset += field[0].itemsize
        result.append(tup)

    return result

# Build a new array from the information in a pickle.
# Note that the name numpy.core._internal._reconstruct is embedded in
# pickles of ndarrays made with NumPy before release 1.0
# so don't remove the name here, or you'll
# break backward compatibilty.
def _reconstruct(subtype, shape, dtype):
    from multiarray import ndarray
    return ndarray.__new__(subtype, shape, dtype)


# format_re and _split were taken from numarray by J. Todd Miller

def _split(input):
    """Split the input formats string into field formats without splitting
       the tuple used to specify multi-dimensional arrays."""

    newlist = []
    hold = asbytes('')

    listinput = input.split(asbytes(','))
    for element in listinput:
        if hold != asbytes(''):
            item = hold + asbytes(',') + element
        else:
            item = element
        left = item.count(asbytes('('))
        right = item.count(asbytes(')'))

        # if the parenthesis is not balanced, hold the string
        if left > right :
            hold = item

        # when balanced, append to the output list and reset the hold
        elif left == right:
            newlist.append(item.strip())
            hold = asbytes('')

        # too many close parenthesis is unacceptable
        else:
            raise SyntaxError(item)

    # if there is string left over in hold
    if hold != asbytes(''):
        raise SyntaxError(hold)

    return newlist

format_datetime = re.compile(asbytes(r"""
     (?P<typecode>M8|m8|datetime64|timedelta64)
     ([[]
       ((?P<num>\d+)?
       (?P<baseunit>Y|M|W|B|D|h|m|s|ms|us|ns|ps|fs|as)
       (/(?P<den>\d+))?
      []])
     (//(?P<events>\d+))?)?"""), re.X)

# Return (baseunit, num, den, events), datetime
#  from date-time string
def _datetimestring(astr):
    res = format_datetime.match(astr)
    if res is None:
        raise ValueError("Incorrect date-time string.")
    typecode = res.group('typecode')
    datetime = (typecode == asbytes('M8') or typecode == asbytes('datetime64'))
    defaults = [asbytes('us'), 1, 1, 1]
    names = ['baseunit', 'num', 'den', 'events']
    func = [bytes, int, int, int]
    dt_tuple = []
    for i, name in enumerate(names):
        value = res.group(name)
        if value:
            dt_tuple.append(func[i](value))
        else:
            dt_tuple.append(defaults[i])

    return tuple(dt_tuple), datetime

format_re = re.compile(asbytes(r'(?P<order1>[<>|=]?)(?P<repeats> *[(]?[ ,0-9]*[)]? *)(?P<order2>[<>|=]?)(?P<dtype>[A-Za-z0-9.]*)'))

# astr is a string (perhaps comma separated)

_convorder = {asbytes('='): _nbo}

def _commastring(astr):
    res = _split(astr)
    if (len(res)) < 1:
        raise ValueError("unrecognized formant")
    result = []
    for k,item in enumerate(res):
        # convert item
        try:
            (order1, repeats, order2, dtype) = format_re.match(item).groups()
        except (TypeError, AttributeError):
            raise ValueError('format %s is not recognized' % item)

        if order2 == asbytes(''):
            order = order1
        elif order1 == asbytes(''):
            order = order2
        else:
            order1 = _convorder.get(order1, order1)
            order2 = _convorder.get(order2, order2)
            if (order1 != order2):
                raise ValueError('in-consistent byte-order specification %s and %s' % (order1, order2))
            order = order1

        if order in [asbytes('|'), asbytes('='), _nbo]:
            order = asbytes('')
        dtype = order + dtype
        if (repeats == asbytes('')):
            newitem = dtype
        else:
            newitem = (dtype, eval(repeats))
        result.append(newitem)

    return result

def _getintp_ctype():
    from multiarray import dtype
    val = _getintp_ctype.cache
    if val is not None:
        return val
    char = dtype('p').char
    import ctypes
    if (char == 'i'):
        val = ctypes.c_int
    elif char == 'l':
        val = ctypes.c_long
    elif char == 'q':
        val = ctypes.c_longlong
    else:
        val = ctypes.c_long
    _getintp_ctype.cache = val
    return val
_getintp_ctype.cache = None

# Used for .ctypes attribute of ndarray

class _missing_ctypes(object):
    def cast(self, num, obj):
        return num

    def c_void_p(self, num):
        return num

class _ctypes(object):
    def __init__(self, array, ptr=None):
        try:
            import ctypes
            self._ctypes = ctypes
        except ImportError:
            self._ctypes = _missing_ctypes()
        self._arr = array
        self._data = ptr
        if self._arr.ndim == 0:
            self._zerod = True
        else:
            self._zerod = False

    def data_as(self, obj):
        return self._ctypes.cast(self._data, obj)

    def shape_as(self, obj):
        if self._zerod:
            return None
        return (obj*self._arr.ndim)(*self._arr.shape)

    def strides_as(self, obj):
        if self._zerod:
            return None
        return (obj*self._arr.ndim)(*self._arr.strides)

    def get_data(self):
        return self._data

    def get_shape(self):
        if self._zerod:
            return None
        return (_getintp_ctype()*self._arr.ndim)(*self._arr.shape)

    def get_strides(self):
        if self._zerod:
            return None
        return (_getintp_ctype()*self._arr.ndim)(*self._arr.strides)

    def get_as_parameter(self):
        return self._ctypes.c_void_p(self._data)

    data = property(get_data, None, doc="c-types data")
    shape = property(get_shape, None, doc="c-types shape")
    strides = property(get_strides, None, doc="c-types strides")
    _as_parameter_ = property(get_as_parameter, None, doc="_as parameter_")


# Given a datatype and an order object
#  return a new names tuple
#  with the order indicated
def _newnames(datatype, order):
    oldnames = datatype.names
    nameslist = list(oldnames)
    if isinstance(order, str):
        order = [order]
    if isinstance(order, (list, tuple)):
        for name in order:
            try:
                nameslist.remove(name)
            except ValueError:
                raise ValueError("unknown field name: %s" % (name,))
        return tuple(list(order) + nameslist)
    raise ValueError("unsupported order value: %s" % (order,))

# Given an array with fields and a sequence of field names
# construct a new array with just those fields copied over
def _index_fields(ary, fields):
    from multiarray import empty, dtype
    dt = ary.dtype
    new_dtype = [(name, dt[name]) for name in dt.names if name in fields]
    if ary.flags.f_contiguous:
        order = 'F'
    else:
        order = 'C'

    newarray = empty(ary.shape, dtype=new_dtype, order=order) 
   
    for name in fields:
        newarray[name] = ary[name]

    return newarray
    
# Given a string containing a PEP 3118 format specifier,
# construct a Numpy dtype

_pep3118_map = {
    'b': 'b',
    'B': 'B',
    'h': 'h',
    'H': 'H',
    'i': 'i',
    'I': 'I',
    'l': 'l',
    'L': 'L',
    'q': 'q',
    'Q': 'Q',
    'f': 'f',
    'd': 'd',
    'g': 'g',
    'Q': 'Q',
    'Zf': 'F',
    'Zd': 'D',
    'Zg': 'G',
    's': 'S',
    'w': 'U',
    'O': 'O',
    'x': 'V', # padding
}

def _dtype_from_pep3118(spec, byteorder='=', is_subdtype=False):
    from numpy.core.multiarray import dtype

    fields = {}
    offset = 0
    findex = 0
    explicit_name = False

    while spec:
        value = None

        # End of structure, bail out to upper level
        if spec[0] == '}':
            spec = spec[1:]
            break

        # Sub-arrays (1)
        shape = None
        if spec[0] == '(':
            j = spec.index(')')
            shape = tuple(map(int, spec[1:j].split(',')))
            spec = spec[j+1:]

        # Byte order
        if spec[0] in ('=', '<', '>'):
            byteorder = spec[0]
            spec = spec[1:]

        # Item sizes
        itemsize = 1
        if spec[0].isdigit():
            j = 1
            for j in xrange(1, len(spec)):
                if not spec[j].isdigit():
                    break
            itemsize = int(spec[:j])
            spec = spec[j:]

        # Data types
        is_padding = False

        if spec[:2] == 'T{':
            value, spec = _dtype_from_pep3118(spec[2:], byteorder=byteorder,
                                              is_subdtype=True)
            if itemsize != 1:
                # Not supported
                raise ValueError("Non item-size 1 structures not supported")
        elif spec[0].isalpha():
            j = 1
            for j in xrange(1, len(spec)):
                if not spec[j].isalpha():
                    break
            typechar = spec[:j]
            spec = spec[j:]
            is_padding = (typechar == 'x')
            dtypechar = _pep3118_map[typechar]
            if dtypechar in 'USV':
                dtypechar += '%d' % itemsize
                itemsize = 1
            value = dtype(byteorder + dtypechar)
        else:
            raise ValueError("Unknown PEP 3118 data type specifier %r" % spec)

        # Convert itemsize to sub-array
        if itemsize != 1:
            value = dtype((value, (itemsize,)))

        # Sub-arrays (2)
        if shape is not None:
            value = dtype((value, shape))

        # Field name
        if spec and spec.startswith(':'):
            i = spec[1:].index(':') + 1
            name = spec[1:i]
            spec = spec[i+1:]
            explicit_name = True
        else:
            name = 'f%d' % findex
            findex += 1

        if not is_padding:
            fields[name] = (value, offset)
        offset += value.itemsize


    if len(fields.keys()) == 1 and not explicit_name and fields['f0'][1] == 0:
        ret = fields['f0'][0]
    else:
        ret = dtype(fields)

    if is_subdtype:
        return ret, spec
    else:
        return ret
