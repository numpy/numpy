__all__ = ['record', 'ndrecarray','format_parser']

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
    def __init__(self, formats, names, titles, aligned=False):
        self._parseFormats(formats, aligned)
        self._setfieldnames(names, titles)
        self._createdescr()

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
        stops = [0] * self._nfields
        self._offsets = [0] * self._nfields
        self._rec_aligned = aligned


        # preserve the input for future reference
        self._formats = [''] * self._nfields

        # fields-compatible formats
        self._f_formats = [''] * self._nfields
                          
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
                self._sizes[i] = self._itemsizes[i] * \
                                 reduce(lambda x,y: x*y, _repeat)
            else:
                self._sizes[i] = self._itemsizes[i] * _repeat

            sum += self._sizes[i]
            if self._rec_aligned:
                # round sum up to multiple of alignment factor
                align = _alignment[_fmt[i]]
                sum = ((sum + align - 1)/align) * align
                maxalign = max(maxalign, align)
            stops[i] = sum - 1

            self._offsets[i] = stops[i] - self._sizes[i] + 1

            # Unify the appearance of _format, independent of input formats
            revfmt = _typestr[_fmt[i]]
            self._f_formats[i] = revfmt
            if issubclass(_fmt[i], nt.flexible):
                if issubclass(_fmt[i], nt.unicode_):
                    self._f_formats[i] += `self._itemsizes[i] / unisize`
                else:
                    self._f_formats[i] += `self._itemsizes[i]`

            self._formats[i] = `_repeat`+self._f_formats[i]
            if (_repeat != 1):
                self._f_formats[i] = (self._f_formats[i], _repeat)
                                            
        self._fmt = _fmt
        # This pads record so next record is aligned if self._rec_align is
        #   true. Otherwise next the record starts right after the end
        #   of the last one.
        self._total_itemsize = (stops[-1]/maxalign + 1) * maxalign

    def _setfieldnames(self, names, titles):
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

        # if the names are not specified, they will be assigned as "f1, f2,..."
        # if not enough names are specified, they will be assigned as "f[n+1],
        # f[n+2],..." etc. where n is the number of specified names..."
        self._names += map(lambda i: 
            'f'+`i`, range(len(self._names)+1,self._nfields+1))

        # check for redundant names
        _dup = find_duplicate(self._names)
        if _dup:
            raise ValueError, "Duplicate field names: %s" % _dup

        if (titles):
            self._titles = [n.strip() for n in titles][:self._nfields]
        else:
            self._titles = []
            titles = []

        if (self._nfields > len(titles)):
            self._titles += [None]*(self._nfields-len(titles))
            
    def _createdescr(self):
        self._descr = sb.dtypedescr({'names':self._names,
                                     'formats':self._f_formats,
                                     'offsets':self._offsets,
                                     'titles':self._titles})
        
class record(nt.void):
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        fdict = self.fields
        names = fdict.keys()
        all = []
        for name in names:
            item = fdict[name]
            if (len(item) > 3) and item[2] == name:
                continue
            all.append(item)

        all.sort(lambda x,y: cmp(x[1],y[1]))

        outlist = [self.getfield(item[0], item[1]) for item in all]
        return str(outlist)

    def __getattribute__(self, attr):
        if attr in ['setfield', 'getfield', 'fields']:
            return nt.void.__getattribute__(self, attr)
        fielddict = nt.void.__getattribute__(self, 'fields')
        res = fielddict.get(attr,None)
        if res:
            return self.getfield(*res[:2])
        return nt.void.__getattribute__(self, attr)

    def __setattr__(self, attr, val):
        if attr in ['setfield', 'getfield', 'fields']:
            raise AttributeError, "Cannot set '%s' attribute" % attr;
        fielddict = nt.void.__getattribute__(self,'fields')
        res = fielddict.get(attr,None)
        if res:
            return self.setfield(val,*res[:2])

        return nt.void.__setattr__(self,attr,val)
    
    def __getitem__(self, obj):
        return self.getfield(*(self.fields[obj][:2]))
       
    def __setitem__(self, obj, val):
        return self.setfield(val, *(self.fields[obj][:2]))
        

# The ndrecarray is almost identical to a standard array (which supports
#   named fields already)  The biggest difference is that it is always of
#   record data-type, has fields, and can use attribute-lookup to access
#   those fields.


class ndrecarray(sb.ndarray):
    def __new__(subtype, shape, formats, names=None, titles=None,
                buf=None, offset=0, strides=None, swap=0, aligned=0):

        if isinstance(formats,str):
            parsed = format_parser(formats, aligned, names, titles)
            descr = parsed._descr
        else:
            if aligned:
                if not isinstance(formats, dict) and \
                   not isinstance(formats, list):
                    raise ValueError, "Can only deal with alignment"\
                          "for list and dictionary type-descriptors."
            descr = sb.dtypedescr(formats, aligned)
        
        if buf is None:
            self = sb.ndarray.__new__(subtype, shape, (record, descr))
        else:
            self = sb.ndarray.__new__(subtype, shape, (record, descr),
                                      buffer=buf, swap=swap)
        return self

    def __getattribute__(self, attr):
        fielddict = sb.ndarray.__getattribute__(self,'dtypedescr').fields
        try:
            res = fielddict[attr][:2]
        except:
            return sb.ndarray.__getattribute__(self,attr)
        
        return self.getfield(*res)
    
    def __setattr__(self, attr, val):
        fielddict = sb.ndarray.__getattribute__(self,'dtypedescr').fields
        try:
            res = fielddict[attr][:2]
        except:
            return sb.ndarray.__setattr__(self,attr,val)
        
        return self.setfield(val,*res)
    
