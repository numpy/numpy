#A place for code to be called from C-code
#  that implements more complicated stuff.

import re
import sys

if (sys.byteorder == 'little'):
    _nbo = '<'
else:
    _nbo = '>'

def _makenames_list(adict):
    from multiarray import dtype
    allfields = []
    fnames = adict.keys()
    for fname in fnames:
        obj = adict[fname]
        n = len(obj)
        if not isinstance(obj, tuple) or n not in [2,3]:
            raise ValueError, "entry not a 2- or 3- tuple"
        if (n > 2) and (obj[2] == fname):
            continue
        num = int(obj[1])
        if (num < 0):
            raise ValueError, "invalid offset."
        format = dtype(obj[0])
        if (format.itemsize == 0):
            raise ValueError, "all itemsizes must be fixed."
        if (n > 2):
            title = obj[2]
        else:
            title = None
        allfields.append((fname, format, num, title))
    # sort by offsets
    allfields.sort(lambda x,y: cmp(x[2],y[2]))
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
    fields = descriptor.fields
    if fields is None:
        return descriptor.str

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
    hold = ''

    listinput = input.split(',')
    for element in listinput:
        if hold != '':
            item = hold + ',' + element
        else:
            item = element
        left = item.count('(')
        right = item.count(')')

        # if the parenthesis is not balanced, hold the string
        if left > right :
            hold = item

        # when balanced, append to the output list and reset the hold
        elif left == right:
            newlist.append(item.strip())
            hold = ''

        # too many close parenthesis is unacceptable
        else:
            raise SyntaxError, item

    # if there is string left over in hold
    if hold != '':
        raise SyntaxError, hold

    return newlist

format_re = re.compile(r'(?P<order1>[<>|=]?)(?P<repeats> *[(]?[ ,0-9]*[)]? *)(?P<order2>[<>|=]?)(?P<dtype>[A-Za-z0-9.]*)')

# astr is a string (perhaps comma separated)

_convorder = {'=': _nbo,
              '|': '|',
              '>': '>',
              '<': '<'}

def _commastring(astr):
    res = _split(astr)
    if (len(res)) < 1:
        raise ValueError, "unrecognized formant"
    result = []
    for k,item in enumerate(res):
        # convert item
        try:
            (order1, repeats, order2, dtype) = format_re.match(item).groups()
        except (TypeError, AttributeError):
            raise ValueError('format %s is not recognized' % item)

        if order2 == '':
            order = order1
        elif order1 == '':
            order = order2
        else:
            order1 = _convorder[order1]
            order2 = _convorder[order2]
            if (order1 != order2):
                raise ValueError('in-consistent byte-order specification %s and %s' % (order1, order2))
            order = order1

        if order in ['|', '=', _nbo]:
            order = ''
        dtype = '%s%s' % (order, dtype)
        if (repeats == ''):
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
                raise ValueError, "unknown field name: %s" % (name,)
        return tuple(list(order) + nameslist)
    raise ValueError, "unsupported order value: %s" % (order,)
