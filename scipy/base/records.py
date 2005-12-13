__all__ = ['record', 'recarray','format_parser']

import numeric as sb
import numerictypes as nt
import sys
import types
import re, stat, os

_sysbyte = sys.byteorder
_byteorders = ['big','little']

# formats regular expression
# allows multidimension spec with a tuple syntax in front 
# of the letter code '(2,3)f4' and ' (  2 ,  3  )  f4  ' 
# are equally allowed
format_re = re.compile(r'(?P<repeat> *[(]?[ ,0-9]*[)]? *)(?P<dtype>[A-Za-z0-9.]*)')

numfmt = nt.typeDict
_typestr = nt._typestr

def find_duplicate(list):
    """Find duplication in a list, return a list of duplicated elements"""
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
            if _dtype[0] in ['V','S','U','a']:
                self._itemsizes[i] = int(_dtype[1:])
                if _dtype[0] == 'U':
                    self._itemsizes[i] *= unisize
                if _dtype[0] == 'a':
                    _dtype = 'S'
                else:
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
        return str(tuple(outlist))

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
        

# The recarray is almost identical to a standard array (which supports
#   named fields already)  The biggest difference is that it can use
#   attribute-lookup to the fields.


class recarray(sb.ndarray):
    def __new__(subtype, shape, formats, names=None, titles=None,
                buf=None, offset=0, strides=None, byteorder=_sysbyte,
                aligned=0):

        if isinstance(formats, sb.dtypedescr):
            descr = formats
        elif isinstance(formats,str):
            parsed = format_parser(formats, names, titles, aligned)
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
            swap = ((byteorder != _sysbyte) and (byteorder in _byteorders))
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


def fromarrays(arrayList, formats=None, names=None, titles=None, shape=None,
               aligned=0):
    """ create a record array from a (flat) list of arrays

    >>> x1=array([1,2,3,4])
    >>> x2=array(['a','dd','xyz','12'])
    >>> x3=array([1.1,2,3,4])
    >>> r=fromarrays([x1,x2,x3],names='a,b,c')
    >>> print r[1]
    (2, 'dd\x00', 2.0)
    >>> x1[1]=34
    >>> r.a
    array([1, 2, 3, 4])
    """

    if shape is None or shape == 0:
        shape = arrayList[0].shape

    if isinstance(shape, int):
        shape = (shape,)
            
    if formats is None:
        # go through each object in the list to see if it is an ndarray
        # and determine the formats.
        formats = ''
        for obj in arrayList:
            if not isinstance(obj, sb.ndarray):
                raise ValueError, "item in the array list must be an ndarray."
            if obj.ndim == 1: 
                _repeat = ''
            elif len(obj._shape) >= 2:
                _repeat = `obj._shape[1:]`
            formats += _repeat + _typestr[obj.dtype]
            if issubclass(obj.dtype, nt.flexible):
                formats += `obj.itemsize`
            formats += ','
        formats=formats[:-1]

    for obj in arrayList:
        if obj.shape != shape:
            raise ValueError, "array has different shape"

    parsed = format_parser(formats, names, titles, aligned)
    _names = parsed._names
    _array = recarray(shape, parsed._descr)
    
    # populate the record array (makes a copy)
    for i in range(len(arrayList)):
        _array[_names[i]] = arrayList[i]

    return _array

# shape must be 1-d 
def fromrecords(recList, formats=None, names=None, titles=None, shape=None,
                aligned=0):
    """ create a Record Array from a list of records in text form

        The data in the same field can be heterogeneous, they will be promoted
        to the highest data type.  This method is intended for creating
        smaller record arrays.  If used to create large array e.g.

        r=fromrecords([[2,3.,'abc']]*100000)

        it is slow.

    >>> r=fromrecords([[456,'dbe',1.2],[2,'de',1.3]],names='col1,col2,col3')
    >>> print r[0]
    (456, 'dbe', 1.2)
    >>> r.col1
    array([456,   2])
    >>> r.col2
    array(['dbe', 'de'])
    >>> import cPickle
    >>> print cPickle.loads(cPickle.dumps(r))
    recarray[ 
    (456, 'dbe', 1.2),
    (2, 'de', 1.3)
    ]
    """

    if (shape is None or shape == 0):
        shape = len(recList)

    if isinstance(shape, (int, long)):
        shape = (shape,)

    if len(shape > 1):
        raise ValueError, "Can only deal with 1-d list of records"

    nfields = len(recList[0])
    if formats is None:  # slower
        obj = sb.array(recList,dtype=object)
        arrlist = [sb.array(obj[:,i].tolist()) for i in xrange(nfields)]
        return fromarrays(arrlist, formats=formats, shape=shape, names=names,
                          titles=titles, aligned=aligned)
    
    parsed = format_parser(formats, names, titles, aligned)
    _names = parsed._names
    _array = recarray(shape, parsed._descr)
    farr = _array.flat
    
    for k in xrange(_array.size):
        for j in xrange(nfields):
            farr[k][_names[j]] = recList[k][j]

    return _array

def fromstring(datastring, formats, shape=None, names=None, titles=None,
               byteorder=_sysbyte, aligned=0, offset=0):
    """ create a (read-only) record array from binary data contained in
    a string"""

    formats = format_parser(formats, names, titles, aligned)
    itemsize = parsed._descr.itemsize
    if (shape is None or shape == 0 or shape == -1):
        shape = (len(datastring)-offset) / itemsize
        
    _array = recarray(shape, parsed._descr, shape=shape, names=names,
                      titles=titles, buf=datastring, offset=offset,
                      byteorder=byteorder)
    return _array

def fromfile(fd, formats, shape=None, names=None, titles=None,
             byteorder=_sysbyte, aligned=0, offset=0):
    """Create an array from binary file data

    If file is a string then that file is opened, else it is assumed
    to be a file object. No options at the moment, all file positioning
    must be done prior to this function call with a file object

    >>> import testdata, sys
    >>> fd=open(testdata.filename)
    >>> fd.seek(2880*2)
    >>> r=fromfile(fd, formats='f8,i4,a5', shape=3, byteorder='big')
    >>> print r[0]
    (5.1000000000000005, 61, 'abcde')
    >>> r._shape
    (3,)
    """

    if (shape is None or shape == 0):
        shape = (-1,)
    elif isinstance(shape, (int, long)):
        shape = (shape,)

    name = 0
    if isinstance(fd, str):
        name = 1
        fd = open(fd, 'rb')
    if (offset > 0):
        fd.seek(offset, 1)
    try:
        size = os.fstat(fd.fileno())[stat.ST_SIZE] - fd.tell()
    except:
        size = os.path.getsize(fd.name) - fd.tell()

    parsed = format_parser(formats, names, titles, aligned)
    itemsize = parsed._descr.itemsize

    shapeprod = sb.array(shape).prod()
    shapesize = shapeprod*itemsize
    if shapesize < 0:
        shape = list(shape)
        shape[ shape.index(-1) ] = size / -shapesize
        shape = tuple(shape)
        shapeprod = sb.array(shape).prod()
        
    nbytes = shapeprod*itemsize

    if nbytes > size:
        raise ValueError(
                "Not enough bytes left in file for specified shape and type")

    # create the array
    _array = recarray(shape, parsed._descr)
    nbytesread = fd.readinto(_array.data)
    if nbytesread != nbytes:
        raise IOError("Didn't read as many bytes as expected")
    if name:
        fd.close()

    # update swap flag if byteorder does not match
    if ((byteorder != _sysbyte) and (byteorder in _byteorders)):
        _array.flags.swapped=True
        
    return _array


def array(obj, formats=None, names=None, titles=None, shape=None,
          byteorder=_sysbyte, aligned=0, offset=0, strides=None):
    
    if isinstance(obj, (type(None), str, file)) and (formats is None):
        raise ValueError("Must define formats if object is "\
                         "None, string, or a file pointer")

    elif obj is None:
        if shape is None:
            raise ValueError("Must define a shape if obj is None")
        return recarray(shape, formats, names=names, titles=titles,
                        buf=obj, offset=offset, strides=strides,
                        byteorder=byteorder, aligned=aligned)
    elif isinstance(obj, str):
        return fromstring(obj, formats, names=names, titles=titles,
                          shape=shape, byteorder=byteorder, aligned=aligned,
                          offset=offset)
    elif isinstance(obj, (list, tuple)):
        if isinstance(obj[0], sb.ndarray):
            return fromarrays(obj, formats=formats, names=names, titles=titles,
                              shape=shape, aligned=aligned)
        else:
            return fromrecords(obj, formats=formats, names=names, titles=titles,
                               shape=shape, aligned=aligned)
    elif isinstance(obj, recarray):
        new = obj.copy()
        parsed = format_parser(formats, names, titles, aligned)
        new.dtypedescr = parsed._descr
        return new
    elif isinstance(obj, file):
        return fromfile(obj, formats=formats, names=names, titles=titles,
                        shape=shape, byteorder=byteorder, aligned=aligned,
                        offset=offset)
    else:
        raise ValueError("Unknown input type")
    
    
                
        
