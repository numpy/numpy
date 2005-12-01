import numeric as sb
import numerictypes as nt
import sys
import types
import re

# formats regular expression
# allows multidimension spec with a tuple syntax in front 
# of the letter code '(2,3)f4' and ' (  2 ,  3  )  f4  ' 
# are equally allowed
format_re = re.compile(r'(?P<repeat> *[(]?[ ,0-9]*[)]? *)(?P<dtype>[A-Za-z0-9.]*)')

numfmt = nt.typeDict



def find_duplicate(list):
    """Find duplication in a list, return a list of dupicated elements"""
    dup = []
    for i in range(len(list)):
        if (list[i] in list[i+1:]):
            if (list[i] not in dup):
                dup.append(list[i])
    return dup

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
            newlist.append(item)
            hold = ''

        # too many close parenthesis is unacceptable
        else:
            raise SyntaxError, item

    # if there is string left over in hold
    if hold != '':
        raise SyntaxError, hold

    return newlist


class format_parser:
    def __init__(self, formats, aligned=False):
        self._parseFormats(formats, aligned)

    def _parseFormats(self, formats, aligned=0):
        """ Parse the field formats """

        _alignment = nt._alignment
        _bytes = nt.nbytes
        _typestr = nt._typestr
        if (type(formats) in [types.ListType, types.TupleType]):
            _fmt = formats[:]
        elif (type(formats) == types.StringType):
            _fmt = _split(formats)
        else:
            raise NameError, "illegal input formats %s" % `formats`

        self._nfields = len(_fmt)
        self._repeats = [1] * self._nfields
        self._itemsizes = [0] * self._nfields
        self._sizes = [0] * self._nfields
        self._stops = [0] * self._nfields
        self._rec_aligned = aligned

        # preserve the input for future reference
        self._formats = [''] * self._nfields

        sum = 0
        maxalign = 1
        unisize = nt._unicodesize
        for i in range(self._nfields):

            # parse the formats into repeats and formats
            try:
                (_repeat, _dtype) = format_re.match(_fmt[i].strip()).groups()
            except TypeError, AttributeError: 
                raise ValueError('format %s is not recognized' % _fmt[i])

            # Flexible types need special treatment
            _dtype = _dtype.strip()
            if _dtype[0] in ['V','S','U']:
                self._itemsizes[i] = int(_dtype[1:])
                if _dtype[0] == 'U':
                    self._itemsizes[i] *= unisize
                _dtype = _dtype[0]

            if _repeat == '': 
                _repeat = 1
            else: 
                _repeat = eval(_repeat)
            _fmt[i] = numfmt[_dtype]
            if not issubclass(_fmt[i], nt.flexible):
                self._itemsizes[i] = _bytes[_fmt[i]]
            self._repeats[i] = _repeat

            if (type(_repeat) in [types.ListType, types.TupleType]):
                self._sizes[i] = self._itemsizes[i] * reduce(lambda x,y: x*y, _repeat)
            else:
                self._sizes[i] = self._itemsizes[i] * _repeat

            sum += self._sizes[i]
            if self._rec_aligned:
                # round sum up to multiple of alignment factor
                align = _alignment[_fmt[i]]
                sum = ((sum + align - 1)/align) * align
                maxalign = max(maxalign, align)
            self._stops[i] = sum - 1

            # Unify the appearance of _format, independent of input formats
            revfmt = _typestr[_fmt[i]]
            self._formats[i] = `_repeat`+revfmt
            if issubclass(_fmt[i], nt.flexible):
                if issubclass(_fmt[i], nt.unicode_):
                    self._formats[i] += `self._itemsizes[i] / unisize`
                else:
                    self._formats[i] += `self._itemsizes[i]`
                    
        self._fmt = _fmt
        # This pads record so next record is aligned if self._rec_align is true.
        # Otherwise next the record starts right after the end of the last one.
        self._total_itemsize = (self._stops[-1]/maxalign + 1) * maxalign


class record(nt.void):
    pass

class ndrecarray(sb.ndarray):
    def __new__(subtype, *args, **kwds):
        shape = args[0]
        formats = args[1]
        buf = kwds.get('buf',None)
        aligned = kwds.get('aligned',0)
        parsed = format_parser(formats, aligned)
        itemsize = parsed._total_itemsize

        if buf is None:
            self = sb.ndarray.__new__(subtype, shape, record, itemsize)
        else:
            byteorder = kwds.get('byteorder', sys.byteorder)
            swapped = 0
            if (byteorder != sys.byteorder):
                swapped = 1
            self = sb.ndarray.__new__(subtype, shape, record, itemsize, buffer=buf,
                                      swapped=swapped)
        self.parsed = parsed
        return self    

    def __init__(self, shape, formats, names=None, buf=None, offset=0,
                 strides=None, byteorder=sys.byteorder, aligned=0):
        print "init: ", buf, formats, shape, names, offset, strides,\
              byteorder, aligned
        self._updateattr()        
        self._fieldNames(names)
        self._fields = {}

        # This should grab the names out of self.parsed that are important
        #  to have later and should set self._attributes
        #  to the list of meta information that needs to be carried around
    def _updateattr(self):
        self._nfields = self.parsed._nfields
        self._attributes = ['parsed']

    def __array_finalize__(self, obj):
        self._attributes = obj._attributes
        for key in self._attributes:
            setattr(self, key, getattr(obj, key))

    def _fieldNames(self, names=None):
        """convert input field names into a list and assign to the _names
        attribute """

        if (names):
            if (type(names) in [types.ListType, types.TupleType]):
                pass
            elif (type(names) == types.StringType):
                names = names.split(',')
            else:
                raise NameError, "illegal input names %s" % `names`

            self._names = map(lambda n:n.strip(), names)[:self._nfields]
        else: 
            self._names = []

        # if the names are not specified, they will be assigned as "c1, c2,..."
        # if not enough names are specified, they will be assigned as "c[n+1],
        # c[n+2],..." etc. where n is the number of specified names..."
        self._names += map(lambda i: 
            'c'+`i`, range(len(self._names)+1,self._nfields+1))

        # check for redundant names
        _dup = find_duplicate(self._names)
        if _dup:
            raise ValueError, "Duplicate field names: %s" % _dup

    def _get_fields(self):
        self._fields = {}
        parsed = self.parsed
        basearr = self.__array__()
        for indx in range(self._nfields):
            # We need the offset and the data type of the field
            _offset = parsed._stops[indx] - parsed._sizes[indx] + 1
            _type = parsed._fmt[indx]
            if issubclass(_type, nt.flexible):
                _type = nt.dtype2char(_type)+`parsed._itemsizes[indx]`
            arr = basearr.getfield(_type, _offset)
            # Put this array as a value in dictionary
            # Do both name and index
            self._fields[indx] = arr
            self._fields[self._names[indx]] = arr
            
    def field(self, field_name):
        if self._fields == {}:
            self._get_fields()
        return self._fields[field_name]

    
