__all__ = ['record', 'recarray', 'format_parser']

import numeric as sb
from defchararray import chararray
import numerictypes as nt
import types
import stat, os

_byteorderconv = {'b':'>',
                  'l':'<',
                  'n':'=',
                  'B':'>',
                  'L':'<',
                  'N':'=',
                  'S':'s',
                  's':'s',
                  '>':'>',
                  '<':'<',
                  '=':'=',
                  '|':'|',
                  'I':'|',
                  'i':'|'}

# formats regular expression
# allows multidimension spec with a tuple syntax in front
# of the letter code '(2,3)f4' and ' (  2 ,  3  )  f4  '
# are equally allowed

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


class format_parser:
    def __init__(self, formats, names, titles, aligned=False):
        self._parseFormats(formats, aligned)
        self._setfieldnames(names, titles)
        self._createdescr()

    def _parseFormats(self, formats, aligned=0):
        """ Parse the field formats """

        dtype = sb.dtype(formats, aligned)
        fields = dtype.fields
        keys = fields[-1]
        self._f_formats = [fields[key][0] for key in keys]
        self._offsets = [fields[key][1] for key in keys]
        self._nfields = len(keys)

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

            self._names = [n.strip() for n in names[:self._nfields]]
        else:
            self._names = []

        # if the names are not specified, they will be assigned as "f1, f2,..."
        # if not enough names are specified, they will be assigned as "f[n+1],
        # f[n+2],..." etc. where n is the number of specified names..."
        self._names += ['f%d' % i for i in range(len(self._names)+1,
                                                 self._nfields+1)]
        # check for redundant names
        _dup = find_duplicate(self._names)
        if _dup:
            raise ValueError, "Duplicate field names: %s" % _dup

        if (titles):
            self._titles = [n.strip() for n in titles[:self._nfields]]
        else:
            self._titles = []
            titles = []

        if (self._nfields > len(titles)):
            self._titles += [None]*(self._nfields-len(titles))

    def _createdescr(self):
        self._descr = sb.dtype({'names':self._names,
                                'formats':self._f_formats,
                                'offsets':self._offsets,
                                'titles':self._titles})

class record(nt.void):
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.item())

    def __getattribute__(self, attr):
        if attr in ['setfield', 'getfield', 'dtype']:
            return nt.void.__getattribute__(self, attr)
        try:
            return nt.void.__getattribute__(self, attr)
        except AttributeError:
            pass
        fielddict = nt.void.__getattribute__(self, 'dtype').fields
        res = fielddict.get(attr,None)
        if res:
            return self.getfield(*res[:2])
        else:
            raise AttributeError, "'record' object has no "\
                  "attribute '%s'" % attr
        

    def __setattr__(self, attr, val):
        if attr in ['setfield', 'getfield', 'dtype']:
            raise AttributeError, "Cannot set '%s' attribute" % attr;
        try:
            return nt.void.__setattr__(self,attr,val)
        except AttributeError:
            pass
        fielddict = nt.void.__getattribute__(self,'dtype').fields
        res = fielddict.get(attr,None)
        if res:
            return self.setfield(val,*res[:2])
        else:
            raise AttributeError, "'record' object has no "\
                  "attribute '%s'" % attr


# The recarray is almost identical to a standard array (which supports
#   named fields already)  The biggest difference is that it can use
#   attribute-lookup to find the fields and it is constructed using
#   a record.

# If byteorder is given it forces a particular byteorder on all
#  the fields (and any subfields)

class recarray(sb.ndarray):
    def __new__(subtype, shape, formats, names=None, titles=None,
                buf=None, offset=0, strides=None, byteorder=None,
                aligned=0):

        if isinstance(formats, sb.dtype):
            descr = formats
        else:
            parsed = format_parser(formats, names, titles, aligned)
            descr = parsed._descr

        if (byteorder is not None):
            byteorder = _byteorderconv[byteorder[0]]
            descr = descr.newbyteorder(byteorder)

        if buf is None:
            self = sb.ndarray.__new__(subtype, shape, (record, descr))
        else:
            self = sb.ndarray.__new__(subtype, shape, (record, descr),
                                      buffer=buf, offset=offset,
                                      strides=strides)
        return self

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self,attr)
        except AttributeError: # attr must be a fieldname
            pass
        fielddict = sb.ndarray.__getattribute__(self,'dtype').fields
        try:
            res = fielddict[attr][:2]
        except KeyError:
            raise AttributeError, "record array has no attribute %s" % attr
        obj = self.getfield(*res)
        # if it has fields return a recarray, otherwise return
        # normal array
        if obj.dtype.fields:
            return obj
        if obj.dtype.char in 'SU':
            return obj.view(chararray)
        return obj.view(sb.ndarray)

    def __setattr__(self, attr, val):
        try:
            return object.__setattr__(self, attr, val)
        except AttributeError: # Must be a fieldname
            pass
        fielddict = sb.ndarray.__getattribute__(self,'dtype').fields
        try:
            res = fielddict[attr][:2]
        except KeyError:
            raise AttributeError, "record array has no attribute %s" % attr 
        return self.setfield(val,*res)

    def field(self,attr, val=None):
        fielddict = sb.ndarray.__getattribute__(self,'dtype').fields

        if isinstance(attr,int):
            attr=fielddict[-1][attr]

        res = fielddict[attr][:2]

        if val is None:
            obj = self.getfield(*res)
            if obj.dtype.fields:
                return obj
            if obj.dtype.char in 'SU':
                return obj.view(chararray)
            return obj.view(sb.ndarray)
        else:
            return self.setfield(val, *res)

def fromarrays(arrayList, formats=None, names=None, titles=None, shape=None,
               aligned=0):
    """ create a record array from a (flat) list of arrays

    >>> x1=array([1,2,3,4])
    >>> x2=array(['a','dd','xyz','12'])
    >>> x3=array([1.1,2,3,4])
    >>> r=fromarrays([x1,x2,x3],names='a,b,c')
    >>> print r[1]
    (2, 'dd', 2.0)
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
            formats += _typestr[obj.dtype.type]
            if issubclass(obj.dtype.type, nt.flexible):
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

# shape must be 1-d if you use list of lists...
def fromrecords(recList, formats=None, names=None, titles=None, shape=None,
                aligned=0):
    """ create a recarray from a list of records in text form

        The data in the same field can be heterogeneous, they will be promoted
        to the highest data type.  This method is intended for creating
        smaller record arrays.  If used to create large array without formats
        defined

        r=fromrecords([(2,3.,'abc')]*100000)

        it can be slow.

        If formats is None, then this will auto-detect formats. Use list of
        tuples rather than list of lists for faster processing.

    >>> r=fromrecords([(456,'dbe',1.2),(2,'de',1.3)],names='col1,col2,col3')
    >>> print r[0]
    (456, 'dbe', 1.2)
    >>> r.col1
    array([456,   2])
    >>> r.col2
    chararray(['dbe', 'de'])
    >>> import cPickle
    >>> print cPickle.loads(cPickle.dumps(r))
    recarray[
    (456, 'dbe', 1.2),
    (2, 'de', 1.3)
    ]
    """

    nfields = len(recList[0])
    if formats is None:  # slower
        obj = sb.array(recList,dtype=object)
        arrlist = [sb.array(obj[...,i].tolist()) for i in xrange(nfields)]
        return fromarrays(arrlist, formats=formats, shape=shape, names=names,
                          titles=titles, aligned=aligned)

    parsed = format_parser(formats, names, titles, aligned)
    try:
        retval = sb.array(recList, dtype = parsed._descr)
    except TypeError:  # list of lists instead of list of tuples
        if (shape is None or shape == 0):
            shape = len(recList)
        if isinstance(shape, (int, long)):
            shape = (shape,)
        if len(shape) > 1:
            raise ValueError, "Can only deal with 1-d array."
        _array = recarray(shape, parsed._descr)
        for k in xrange(_array.size):
            _array[k] = tuple(recList[k])
        return _array
    else:
        if shape is not None and retval.shape != shape:
            retval.shape = shape

    res = retval.view(recarray)
    res.dtype = sb.dtype((record, res.dtype))
    return res


def fromstring(datastring, formats, shape=None, names=None, titles=None,
               byteorder=None, aligned=0, offset=0):
    """ create a (read-only) record array from binary data contained in
    a string"""

    parsed = format_parser(formats, names, titles, aligned)
    itemsize = parsed._descr.itemsize
    if (shape is None or shape == 0 or shape == -1):
        shape = (len(datastring)-offset) / itemsize

    _array = recarray(shape, parsed._descr, names=names,
                      titles=titles, buf=datastring, offset=offset,
                      byteorder=byteorder)
    return _array

def fromfile(fd, formats, shape=None, names=None, titles=None,
             byteorder=None, aligned=0, offset=0):
    """Create an array from binary file data

    If file is a string then that file is opened, else it is assumed
    to be a file object.

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
    _array = recarray(shape, parsed._descr, byteorder=byteorder)
    nbytesread = fd.readinto(_array.data)
    if nbytesread != nbytes:
        raise IOError("Didn't read as many bytes as expected")
    if name:
        fd.close()

    return _array


def array(obj, formats=None, names=None, titles=None, shape=None,
          byteorder=None, aligned=0, offset=0, strides=None):

    if isinstance(obj, (type(None), str, file)) and (formats is None):
        raise ValueError("Must define formats if object is "\
                         "None, string, or an open file")

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
        new.dtype = parsed._descr
        return new
    elif isinstance(obj, file):
        return fromfile(obj, formats=formats, names=names, titles=titles,
                        shape=shape, byteorder=byteorder, aligned=aligned,
                        offset=offset)
    elif isinstance(obj, sb.ndarray):
        res = obj.view(recarray)
        if issubclass(res.dtype.type, nt.void):
            res.dtype = sb.dtype((record, res.dtype))
        return res
    else:
        raise ValueError("Unknown input type")
