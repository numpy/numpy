"""
A place for code to be called from core C-code.

Some things are more easily handled Python.

"""
from __future__ import division, absolute_import, print_function

import re
import sys

from numpy.compat import asbytes, basestring
from .multiarray import dtype, array, ndarray
import ctypes
from .numerictypes import object_

if (sys.byteorder == 'little'):
    _nbo = asbytes('<')
else:
    _nbo = asbytes('>')

def _makenames_list(adict, align):
    allfields = []
    fnames = list(adict.keys())
    for fname in fnames:
        obj = adict[fname]
        n = len(obj)
        if not isinstance(obj, tuple) or n not in [2, 3]:
            raise ValueError("entry not a 2- or 3- tuple")
        if (n > 2) and (obj[2] == fname):
            continue
        num = int(obj[1])
        if (num < 0):
            raise ValueError("invalid offset.")
        format = dtype(obj[0], align=align)
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
    try:
        names = adict[-1]
    except KeyError:
        names = None
    if names is None:
        names, formats, offsets, titles = _makenames_list(adict, align)
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

    return dtype({"names": names,
                  "formats": formats,
                  "offsets": offsets,
                  "titles": titles}, align)


# construct an array_protocol descriptor list
#  from the fields attribute of a descriptor
# This calls itself recursively but should eventually hit
#  a descriptor that has no fields and then return
#  a simple typestring

def _array_descr(descriptor):
    fields = descriptor.fields
    if fields is None:
        subdtype = descriptor.subdtype
        if subdtype is None:
            if descriptor.metadata is None:
                return descriptor.str
            else:
                new = descriptor.metadata.copy()
                if new:
                    return (descriptor.str, new)
                else:
                    return descriptor.str
        else:
            return (_array_descr(subdtype[0]), subdtype[1])

    names = descriptor.names
    ordered_fields = [fields[x] + (x,) for x in names]
    result = []
    offset = 0
    for field in ordered_fields:
        if field[1] > offset:
            num = field[1] - offset
            result.append(('', '|V%d' % num))
            offset += num
        if len(field) > 3:
            name = (field[2], field[3])
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
    return ndarray.__new__(subtype, shape, dtype)


# format_re was originally from numarray by J. Todd Miller

format_re = re.compile(asbytes(
                           r'(?P<order1>[<>|=]?)'
                           r'(?P<repeats> *[(]?[ ,0-9L]*[)]? *)'
                           r'(?P<order2>[<>|=]?)'
                           r'(?P<dtype>[A-Za-z0-9.?]*(?:\[[a-zA-Z0-9,.]+\])?)'))
sep_re = re.compile(asbytes(r'\s*,\s*'))
space_re = re.compile(asbytes(r'\s+$'))

# astr is a string (perhaps comma separated)

_convorder = {asbytes('='): _nbo}

def _commastring(astr):
    startindex = 0
    result = []
    while startindex < len(astr):
        mo = format_re.match(astr, pos=startindex)
        try:
            (order1, repeats, order2, dtype) = mo.groups()
        except (TypeError, AttributeError):
            raise ValueError('format number %d of "%s" is not recognized' %
                                            (len(result)+1, astr))
        startindex = mo.end()
        # Separator or ending padding
        if startindex < len(astr):
            if space_re.match(astr, pos=startindex):
                startindex = len(astr)
            else:
                mo = sep_re.match(astr, pos=startindex)
                if not mo:
                    raise ValueError(
                        'format number %d of "%s" is not recognized' %
                        (len(result)+1, astr))
                startindex = mo.end()

        if order2 == asbytes(''):
            order = order1
        elif order1 == asbytes(''):
            order = order2
        else:
            order1 = _convorder.get(order1, order1)
            order2 = _convorder.get(order2, order2)
            if (order1 != order2):
                raise ValueError(
                    'inconsistent byte-order specification %s and %s' %
                    (order1, order2))
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
    val = _getintp_ctype.cache
    if val is not None:
        return val
    char = dtype('p').char
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

def _index_fields(ary, names):
    """ Given a structured array and a sequence of field names
    construct new array with just those fields.

    Parameters
    ----------
    ary : ndarray
        Structured array being subscripted
    names : string or list of strings
        Either a single field name, or a list of field names

    Returns
    -------
    sub_ary : ndarray
        If `names` is a single field name, the return value is identical to
        ary.getfield, a writeable view into `ary`. If `names` is a list of
        field names the return value is a copy of `ary` containing only those
        fields. This is planned to return a view in the future.
    
    Raises
    ------
    ValueError
        If `ary` does not contain a field given in `names`.

    """
    dt = ary.dtype

    #use getfield to index a single field
    if isinstance(names, basestring):
        try:
            return ary.getfield(dt.fields[names][0], dt.fields[names][1])
        except KeyError:
            raise ValueError("no field of name %s" % names)

    for name in names:
        if name not in dt.fields:
            raise ValueError("no field of name %s" % name)

    formats = [dt.fields[name][0] for name in names]
    offsets = [dt.fields[name][1] for name in names]

    view_dtype = {'names': names, 'formats': formats,
                  'offsets': offsets, 'itemsize': dt.itemsize}

    # return copy for now (future plan to return ary.view(dtype=view_dtype))
    copy_dtype = {'names': view_dtype['names'],
                  'formats': view_dtype['formats']}
    return array(ary.view(dtype=view_dtype), dtype=copy_dtype, copy=True)


def _get_all_field_offsets(dtype, base_offset=0):
    """ Returns the types and offsets of all fields in a (possibly structured)
    data type, including nested fields and subarrays.

    Parameters
    ----------
    dtype : data-type
        Data type to extract fields from.
    base_offset : int, optional
        Additional offset to add to all field offsets.

    Returns
    -------
    fields : list of (data-type, int) pairs
        A flat list of (dtype, byte offset) pairs.

    """
    fields = []
    if dtype.fields is not None:
        for name in dtype.names:
            sub_dtype = dtype.fields[name][0]
            sub_offset = dtype.fields[name][1] + base_offset
            fields.extend(_get_all_field_offsets(sub_dtype, sub_offset))
    else:
        if dtype.shape:
            sub_offsets = _get_all_field_offsets(dtype.base, base_offset)
            count = 1
            for dim in dtype.shape:
                count *= dim
            fields.extend((typ, off + dtype.base.itemsize*j)
                           for j in range(count) for (typ, off) in sub_offsets)
        else:
            fields.append((dtype, base_offset))
    return fields

def _check_field_overlap(new_fields, old_fields):
    """ Perform object memory overlap tests for two data-types (see
    _view_is_safe).

    This function checks that new fields only access memory contained in old
    fields, and that non-object fields are not interpreted as objects and vice
    versa.

    Parameters
    ----------
    new_fields : list of (data-type, int) pairs
        Flat list of (dtype, byte offset) pairs for the new data type, as
        returned by _get_all_field_offsets.
    old_fields: list of (data-type, int) pairs
        Flat list of (dtype, byte offset) pairs for the old data type, as
        returned by _get_all_field_offsets.

    Raises
    ------
    TypeError
        If the new fields are incompatible with the old fields

    """

    #first go byte by byte and check we do not access bytes not in old_fields
    new_bytes = set()
    for tp, off in new_fields:
        new_bytes.update(set(range(off, off+tp.itemsize)))
    old_bytes = set()
    for tp, off in old_fields:
        old_bytes.update(set(range(off, off+tp.itemsize)))
    if new_bytes.difference(old_bytes):
        raise TypeError("view would access data parent array doesn't own")

    #next check that we do not interpret non-Objects as Objects, and vv
    obj_offsets = [off for (tp, off) in old_fields if tp.type is object_]
    obj_size = dtype(object_).itemsize

    for fld_dtype, fld_offset in new_fields:
        if fld_dtype.type is object_:
            # check we do not create object views where
            # there are no objects.
            if fld_offset not in obj_offsets:
                raise TypeError("cannot view non-Object data as Object type")
        else:
            # next check we do not create non-object views
            # where there are already objects.
            # see validate_object_field_overlap for a similar computation.
            for obj_offset in obj_offsets:
                if (fld_offset < obj_offset + obj_size and
                        obj_offset < fld_offset + fld_dtype.itemsize):
                    raise TypeError("cannot view Object as non-Object type")

def _getfield_is_safe(oldtype, newtype, offset):
    """ Checks safety of getfield for object arrays.

    As in _view_is_safe, we need to check that memory containing objects is not
    reinterpreted as a non-object datatype and vice versa.

    Parameters
    ----------
    oldtype : data-type
        Data type of the original ndarray.
    newtype : data-type
        Data type of the field being accessed by ndarray.getfield
    offset : int
        Offset of the field being accessed by ndarray.getfield

    Raises
    ------
    TypeError
        If the field access is invalid

    """
    new_fields = _get_all_field_offsets(newtype, offset)
    old_fields = _get_all_field_offsets(oldtype)
    # raises if there is a problem
    _check_field_overlap(new_fields, old_fields)

def _view_is_safe(oldtype, newtype):
    """ Checks safety of a view involving object arrays, for example when
    doing::

        np.zeros(10, dtype=oldtype).view(newtype)

    We need to check that
    1) No memory that is not an object will be interpreted as a object,
    2) No memory containing an object will be interpreted as an arbitrary type.
    Both cases can cause segfaults, eg in the case the view is written to.
    Strategy here is to also disallow views where newtype has any field in a
    place oldtype doesn't.

    Parameters
    ----------
    oldtype : data-type
        Data type of original ndarray
    newtype : data-type
        Data type of the view

    Raises
    ------
    TypeError
        If the new type is incompatible with the old type.

    """
    new_fields = _get_all_field_offsets(newtype)
    new_size = newtype.itemsize

    old_fields = _get_all_field_offsets(oldtype)
    old_size = oldtype.itemsize

    # if the itemsizes are not equal, we need to check that all the
    # 'tiled positions' of the object match up. Here, we allow
    # for arbirary itemsizes (even those possibly disallowed
    # due to stride/data length issues).
    if old_size == new_size:
        new_num = old_num = 1
    else:
        gcd_new_old = _gcd(new_size, old_size)
        new_num = old_size // gcd_new_old
        old_num = new_size // gcd_new_old

    # get position of fields within the tiling
    new_fieldtile = [(tp, off + new_size*j)
                     for j in range(new_num) for (tp, off) in new_fields]
    old_fieldtile = [(tp, off + old_size*j)
                     for j in range(old_num) for (tp, off) in old_fields]

    # raises if there is a problem
    _check_field_overlap(new_fieldtile, old_fieldtile)

# Given a string containing a PEP 3118 format specifier,
# construct a Numpy dtype

_pep3118_native_map = {
    '?': '?',
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
    'e': 'e',
    'f': 'f',
    'd': 'd',
    'g': 'g',
    'Zf': 'F',
    'Zd': 'D',
    'Zg': 'G',
    's': 'S',
    'w': 'U',
    'O': 'O',
    'x': 'V',  # padding
}
_pep3118_native_typechars = ''.join(_pep3118_native_map.keys())

_pep3118_standard_map = {
    '?': '?',
    'b': 'b',
    'B': 'B',
    'h': 'i2',
    'H': 'u2',
    'i': 'i4',
    'I': 'u4',
    'l': 'i4',
    'L': 'u4',
    'q': 'i8',
    'Q': 'u8',
    'e': 'f2',
    'f': 'f',
    'd': 'd',
    'Zf': 'F',
    'Zd': 'D',
    's': 'S',
    'w': 'U',
    'O': 'O',
    'x': 'V',  # padding
}
_pep3118_standard_typechars = ''.join(_pep3118_standard_map.keys())

def _dtype_from_pep3118(spec, byteorder='@', is_subdtype=False):
    fields = {}
    offset = 0
    explicit_name = False
    this_explicit_name = False
    common_alignment = 1
    is_padding = False

    dummy_name_index = [0]

    def next_dummy_name():
        dummy_name_index[0] += 1

    def get_dummy_name():
        while True:
            name = 'f%d' % dummy_name_index[0]
            if name not in fields:
                return name
            next_dummy_name()

    # Parse spec
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
        if spec[0] in ('@', '=', '<', '>', '^', '!'):
            byteorder = spec[0]
            if byteorder == '!':
                byteorder = '>'
            spec = spec[1:]

        # Byte order characters also control native vs. standard type sizes
        if byteorder in ('@', '^'):
            type_map = _pep3118_native_map
            type_map_chars = _pep3118_native_typechars
        else:
            type_map = _pep3118_standard_map
            type_map_chars = _pep3118_standard_typechars

        # Item sizes
        itemsize = 1
        if spec[0].isdigit():
            j = 1
            for j in range(1, len(spec)):
                if not spec[j].isdigit():
                    break
            itemsize = int(spec[:j])
            spec = spec[j:]

        # Data types
        is_padding = False

        if spec[:2] == 'T{':
            value, spec, align, next_byteorder = _dtype_from_pep3118(
                spec[2:], byteorder=byteorder, is_subdtype=True)
        elif spec[0] in type_map_chars:
            next_byteorder = byteorder
            if spec[0] == 'Z':
                j = 2
            else:
                j = 1
            typechar = spec[:j]
            spec = spec[j:]
            is_padding = (typechar == 'x')
            dtypechar = type_map[typechar]
            if dtypechar in 'USV':
                dtypechar += '%d' % itemsize
                itemsize = 1
            numpy_byteorder = {'@': '=', '^': '='}.get(byteorder, byteorder)
            value = dtype(numpy_byteorder + dtypechar)
            align = value.alignment
        else:
            raise ValueError("Unknown PEP 3118 data type specifier %r" % spec)

        #
        # Native alignment may require padding
        #
        # Here we assume that the presence of a '@' character implicitly implies
        # that the start of the array is *already* aligned.
        #
        extra_offset = 0
        if byteorder == '@':
            start_padding = (-offset) % align
            intra_padding = (-value.itemsize) % align

            offset += start_padding

            if intra_padding != 0:
                if itemsize > 1 or (shape is not None and _prod(shape) > 1):
                    # Inject internal padding to the end of the sub-item
                    value = _add_trailing_padding(value, intra_padding)
                else:
                    # We can postpone the injection of internal padding,
                    # as the item appears at most once
                    extra_offset += intra_padding

            # Update common alignment
            common_alignment = (align*common_alignment
                                / _gcd(align, common_alignment))

        # Convert itemsize to sub-array
        if itemsize != 1:
            value = dtype((value, (itemsize,)))

        # Sub-arrays (2)
        if shape is not None:
            value = dtype((value, shape))

        # Field name
        this_explicit_name = False
        if spec and spec.startswith(':'):
            i = spec[1:].index(':') + 1
            name = spec[1:i]
            spec = spec[i+1:]
            explicit_name = True
            this_explicit_name = True
        else:
            name = get_dummy_name()

        if not is_padding or this_explicit_name:
            if name in fields:
                raise RuntimeError("Duplicate field name '%s' in PEP3118 format"
                                   % name)
            fields[name] = (value, offset)
            if not this_explicit_name:
                next_dummy_name()

        byteorder = next_byteorder

        offset += value.itemsize
        offset += extra_offset

    # Check if this was a simple 1-item type
    if (len(fields) == 1 and not explicit_name and
            fields['f0'][1] == 0 and not is_subdtype):
        ret = fields['f0'][0]
    else:
        ret = dtype(fields)

    # Trailing padding must be explicitly added
    padding = offset - ret.itemsize
    if byteorder == '@':
        padding += (-offset) % common_alignment
    if is_padding and not this_explicit_name:
        ret = _add_trailing_padding(ret, padding)

    # Finished
    if is_subdtype:
        return ret, spec, common_alignment, byteorder
    else:
        return ret

def _add_trailing_padding(value, padding):
    """Inject the specified number of padding bytes at the end of a dtype"""
    if value.fields is None:
        vfields = {'f0': (value, 0)}
    else:
        vfields = dict(value.fields)

    if (value.names and value.names[-1] == '' and
           value[''].char == 'V'):
        # A trailing padding field is already present
        vfields[''] = ('V%d' % (vfields[''][0].itemsize + padding),
                       vfields[''][1])
        value = dtype(vfields)
    else:
        # Get a free name for the padding field
        j = 0
        while True:
            name = 'pad%d' % j
            if name not in vfields:
                vfields[name] = ('V%d' % padding, value.itemsize)
                break
            j += 1

        value = dtype(vfields)
        if '' not in vfields:
            # Strip out the name of the padding field
            names = list(value.names)
            names[-1] = ''
            value.names = tuple(names)
    return value

def _prod(a):
    p = 1
    for x in a:
        p *= x
    return p

def _gcd(a, b):
    """Calculate the greatest common divisor of a and b"""
    while b:
        a, b = b, a % b
    return a
