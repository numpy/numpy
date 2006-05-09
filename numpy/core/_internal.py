
#A place for code to be called from C-code
#  that implements more complicated stuff.

import re
from multiarray import dtype, ndarray

# make sure the tuple entries are PyArray_Descr
#          or convert them
#
# make sure offsets are all interpretable
#          as positive integers and
#          convert them to positive integers if so
#
#
# return totalsize from last offset and size

# Called in PyArray_DescrConverter function when
#  a dictionary without "names" and "formats"
#  fields is used as a data-type descriptor.
def _usefields(adict, align):
    try:
        names = adict[-1]
    except KeyError:
        names = None
    if names is None:
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

    ordered_fields = [fields[x] + (x,) for x in fields[-1]]
    result = []
    offset = 0
    for field in ordered_fields:
        if field[1] > offset:
            result.append(('','|V%d' % (field[1]-offset)))
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

def _reconstruct(subtype, shape, dtype):
    return ndarray.__new__(subtype, shape, dtype)


# format_re and _split were taken from numarray by J. Todd Miller

def _split(input):
    """Split the input formats string into field formats without splitting
       the tuple used to specify multi-dimensional arrays."""

    newlist = []
    hold = ''

    for element in input.split(','):
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

format_re = re.compile(r'(?P<repeat> *[(]?[ ,0-9]*[)]? *)(?P<dtype>[><|A-Za-z0-9.]*)')

# astr is a string (perhaps comma separated)

def _commastring(astr):
    res = _split(astr)
    if (len(res)) < 1:
        raise ValueError, "unrecognized formant"
    result = []
    for k,item in enumerate(res):
        # convert item
        try:
            (repeats, dtype) = format_re.match(item).groups()
        except (TypeError, AttributeError):
            raise ValueError('format %s is not recognized' % item)

        if (repeats == ''):
            newitem = dtype
        else:
            newitem = (dtype, eval(repeats))
        result.append(newitem)

    return result


