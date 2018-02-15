"""
A place for code to be called from the implementation of np.dtype

Some things are more easily handled in Python.

"""
from __future__ import division, absolute_import, print_function

import numpy as np


def dtype_construction_repr(dtype, include_align, short):
    """
    Creates a string repr of the dtype, excluding the 'dtype()' part
    surrounding the object. This object may be a string, a list, or
    a dict depending on the nature of the dtype. This
    is the object passed as the first parameter to the dtype
    constructor, and if no additional constructor parameters are
    given, will reproduce the exact memory layout.

    Parameters
    ----------
    short : bool
        If true, this creates a shorter repr using 'kind' and 'itemsize', instead
        of the longer type name.

    include_align : bool
        If true, this includes the 'align=True' parameter
        inside the struct dtype construction dict when needed. Use this flag
        if you want a proper repr string without the 'dtype()' part around it.

        If false, this does not preserve the
        'align=True' parameter or sticky NPY_ALIGNED_STRUCT flag for
        struct arrays like the regular repr does, because the 'align'
        flag is not part of first dtype constructor parameter. This
        mode is intended for a full 'repr', where the 'align=True' is
        provided as the second parameter.
    """
    if dtype.fields:
        return dtype_struct_str(dtype, include_align)
    elif dtype.subdtype:
        return dtype_subarray_str(dtype)


    byteorder = dtype_byte_order_str(dtype)

    if dtype.type == np.bool_:
        if short:
            return "'?'"
        else:
            return "'bool'"

    elif dtype.type == np.object_:
        # The object reference may be different sizes on different
        # platforms, so it should never include the itemsize here.
        return "'O'"

    elif dtype.type == np.string_:
        if dtype.itemsize == 0:  # TODO: PyDataType_ISUNSIZED
            return "'S'"
        else:
            return "'S%d'" % dtype.itemsize

    elif dtype.type == np.unicode_:
        if dtype.itemsize == 0:  # TODO: PyDataType_ISUNSIZED
            return "'%sU'" % byteorder
        else:
            return "'%sU%d'" % (byteorder, dtype.itemsize / 4)

    elif dtype.type == np.void:
        if dtype.itemsize == 0:  # TODO: PyDataType_ISUNSIZED
            return "'V'"
        else:
            return "'V%d'" % dtype.itemsize

    elif dtype.type == np.datetime64:
        return "'%sM8%s'" % (byteorder, get_datetimemeta(dtype))

    elif dtype.type == np.timedelta64:
        return "'%sm8%s'" % (byteorder, get_datetimemeta(dtype))

    elif np.issubdtype(dtype, np.number):
        # Short repr with endianness, like '<f8'
        if short or dtype.byteorder not in ('=', '|'):
            return "'%s%c%d'" % (byteorder, dtype.kind, dtype.itemsize)

        # Longer repr, like 'float64'
        else:
            kindstrs = {
                'u': "uint",
                'i': "int",
                'f': "float",
                'c': "complex"
            }
            try:
                kindstr = kindstrs[dtype.kind]
            except KeyError:
                raise RuntimeError(
                    "internal dtype repr error, unknown kind {!r}"
                    .format(dtype.kind)
                )
            return "'%s%d'" % (kindstr, 8*dtype.itemsize)

    elif dtype.isbuiltin == 2:
        return dtype.type.__name__

    else:
        raise RuntimeError(
            "Internal error: NumPy dtype unrecognized type number")


def dtype_byte_order_str(dtype):
    """ Normalize byteorder to '<' or '>' """
    # hack to obtain the native and swapped byte order characters
    swapped = np.dtype(int).newbyteorder('s')
    native = swapped.newbyteorder('s')

    byteorder = dtype.byteorder
    if byteorder == '=':
        return native.byteorder
    if byteorder == 's':
        return swapped.byteorder
    elif byteorder == '|':
        return ''
    else:
        return byteorder


def dtype_datetime_metadata_str(dtype):
    # This is a hack since the data is not exposed to python in any other way
    return dtype.name[dtype.name.rfind('['):]


def unpack_field(dtype, offset, title=None):
    return dtype, offset, title


def dtype_struct_dict_str(dtype, includealignedflag):
    # unpack the fields dictionary into ls
    names = dtype.names
    fld_dtypes = []
    offsets = []
    titles = []
    for name in names:
        fld_dtype, offset, title = unpack_field(*dtype.fields[name])
        fld_dtypes.append(fld_dtype)
        offsets.append(offset)
        titles.append(title)

    # Build up a string to make the dictionary

    # First, the names
    ret = "{'names':["
    ret += ",".join(repr(name) for name in names)

    # Second, the formats
    ret += "], 'formats':["
    ret += ",".join(
        dtype_construction_repr(fld_dtype, 0, 1) for fld_dtype in fld_dtypes)

    # Third, the offsets
    ret += "], 'offsets':["
    ret += ",".join("%d" % offset for offset in offsets)

    # Fourth, the titles
    if any(title is not None for title in titles):
        ret += "], 'titles':["
        ret += ",".join(repr(title) for title in titles)

    # Fifth, the itemsize
    ret += "], 'itemsize':%d}" % dtype.itemsize

    if (includealignedflag and dtype.isalignedstruct):
        # Finally, the aligned flag
        ret += ", 'aligned':True}" % dtype.itemsize
    else:
        ret += "}"

    return ret


def is_dtype_struct_simple_unaligned_layout(dtype):
    """
    Checks whether the structured data type in 'dtype'
    has a simple layout, where all the fields are in order,
    and follow each other with no alignment padding.

    When this returns true, the dtype can be reconstructed
    from a list of the field names and dtypes with no additional
    dtype parameters.
    """
    total_offset = 0
    for name in dtype.names:
        fld_dtype, fld_offset, title = unpack_field(*dtype.fields[name])
        if fld_offset != total_offset:
            return False
        total_offset += fld_dtype.itemsize
    if total_offset != dtype.itemsize:
        return False
    return True

#
# Returns a string representation of a structured array,
# in a list format.
#

def dtype_struct_list_str(dtype):
    # Build up a string to make the list

    items = []
    for name in dtype.names:
        fld_dtype, fld_offset, title = unpack_field(*dtype.fields[name])

        item = "("
        if title is not None:
            item += "({!r}, {!r}), ".format(title, name)
        else:
            item += "{!r}, ".format(name)
        # Special case subarray handling here
        if fld_dtype.subdtype is not None:
            base, shape = fld_dtype.subdtype
            item += "{}, {}".format(
                dtype_construction_repr(fld_dtype.subarray.base, 0, 1),
                shape
            )
        else:
            item += dtype_construction_repr(fld_dtype, 0, 1)

        item += ")"
        items.append(item)

    return "[" + ", ".join(items) + "]"


def dtype_struct_str(dtype, include_align):
    # The list str representation can't include the 'align=' flag,
    # so if it is requested and the struct has the aligned flag set,
    # we must use the dict str instead.
    if not (include_align and dtype.isalignedstruct) and \
                        is_dtype_struct_simple_unaligned_layout(dtype):
        sub = dtype_struct_list_str(dtype)

    else:
        sub = dtype_struct_dict_str(dtype, include_align)


    # If the data type isn't the default, void, show it
    if dtype.type != np.void:
        # Note: We cannot get the type name from dtype.typeobj.tp_name
        # because its value depends on whether the type is dynamically or
        # statically allocated.  Instead use __name__ and __module__.
        # See https://docs.python.org/2/c-api/typeobj.html.
        return "({t.__module__}.{t.__name__}, {f})".format(t=dtype.type, f=sub)
    else:
        return sub


def dtype_subarray_str(dtype):
    base, shape = dtype.subdtype
    return "({}, {})".format(
        dtype_construction_repr(base, 0, 1),
        shape
    )

def dtype_str(dtype):
    if dtype.fields:
        return dtype_struct_str(dtype, 1)
    elif dtype.subdtype:
        return dtype_subarray_str(dtype)
    elif not dtype.isnative:
        return dtype_protocol_typestr_get(dtype)
    else:
        return dtype_typename_get(dtype)


def dtype_struct_repr(dtype):
    s = "dtype("
    s += dtype_struct_str(dtype, 0)
    if dtype.isalignedstruct:
        s += ", align=True"
    s += ")"
    return s


def dtype_repr(dtype):
    if dtype.fields:
        return dtype_struct_repr(dtype)
    else:
        return "dtype({})".format(dtype_construction_repr(dtype, 1, 0))
