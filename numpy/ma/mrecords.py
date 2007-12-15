"""mrecords
Defines a class of record arrays supporting masked arrays.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: mrecords.py 3473 2007-10-29 15:18:13Z jarrod.millman $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: jarrod.millman $)"
__version__ = '1.0'
__revision__ = "$Revision: 3473 $"
__date__     = '$Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $'

import sys
import types

import numpy
from numpy import bool_, complex_, float_, int_, str_, object_
from numpy import array as narray
import numpy.core.numeric as numeric
import numpy.core.numerictypes as ntypes
from numpy.core.defchararray import chararray
from numpy.core.records import find_duplicate

from numpy.core.records import format_parser, record, recarray
from numpy.core.records import fromarrays as recfromarrays

ndarray = numeric.ndarray
_byteorderconv = numpy.core.records._byteorderconv
_typestr = ntypes._typestr

import numpy.ma
from numpy.ma import MaskedArray, masked, nomask, masked_array,\
    make_mask, mask_or, getmask, getmaskarray, filled
from numpy.ma.core import default_fill_value, masked_print_option

import warnings

reserved_fields = ['_data','_mask','_fieldmask', 'dtype']

def _getformats(data):
    "Returns the formats of each array of arraylist as a comma-separated string."
    if hasattr(data,'dtype'):
        return ",".join([desc[1] for desc in data.dtype.descr])

    formats = ''
    for obj in data:
        obj = numeric.asarray(obj)
#        if not isinstance(obj, ndarray):
##        if not isinstance(obj, ndarray):
#            raise ValueError, "item in the array list must be an ndarray."
        formats += _typestr[obj.dtype.type]
        if issubclass(obj.dtype.type, ntypes.flexible):
            formats += `obj.itemsize`
        formats += ','
    return formats[:-1]

def _checknames(descr, names=None):
    """Checks that the field names of the descriptor ``descr`` are not some
reserved keywords. If this is the case, a default 'f%i' is substituted.
If the argument `names` is not None, updates the field names to valid names.
    """
    ndescr = len(descr)
    default_names = ['f%i' % i for i in range(ndescr)]
    if names is None:
        new_names = default_names
    else:
        if isinstance(names, (tuple, list)):
            new_names = names
        elif isinstance(names, str):
            new_names = names.split(',')
        else:
            raise NameError, "illegal input names %s" % `names`
        nnames = len(new_names)
        if nnames < ndescr:
            new_names += default_names[nnames:]
    ndescr = []
    for (n, d, t) in zip(new_names, default_names, descr.descr):
        if n in reserved_fields:
            if t[0] in reserved_fields:
                ndescr.append((d,t[1]))
            else:
                ndescr.append(t)
        else:
            ndescr.append((n,t[1]))
    return numeric.dtype(ndescr)



class MaskedRecords(MaskedArray, object):
    """

*IVariables*:
    _data : {recarray}
        Underlying data, as a record array.
    _mask : {boolean array}
        Mask of the records. A record is masked when all its fields are masked.
    _fieldmask : {boolean recarray}
        Record array of booleans, setting the mask of each individual field of each record.
    _fill_value : {record}
        Filling values for each field.
    """
    _defaultfieldmask = nomask
    _defaulthardmask = False
    def __new__(cls, data, mask=nomask, dtype=None,
                hard_mask=False, fill_value=None,
#                offset=0, strides=None,
                formats=None, names=None, titles=None,
                byteorder=None, aligned=False):
        # Get the new descriptor ................
        if dtype is not None:
            descr = numeric.dtype(dtype)
        else:
            if formats is None:
                formats = _getformats(data)
            parsed = format_parser(formats, names, titles, aligned, byteorder)
            descr = parsed._descr
        if names is not None:
            descr = _checknames(descr,names)
        _names = descr.names
        mdescr = [(n,'|b1') for n in _names]
        # get the shape .........................
        try:
            shape = numeric.asarray(data[0]).shape
        except IndexError:
            shape = len(data.dtype)
        if isinstance(shape, int):
            shape = (shape,)
        # Construct the _data recarray ..........
        if isinstance(data, record):
            _data = numeric.asarray(data).view(recarray)
            _fieldmask = mask
        elif isinstance(data, MaskedRecords):
            _data = data._data
            _fieldmask = data._fieldmask
        elif isinstance(data, recarray):
            _data = data
            if mask is nomask:
                _fieldmask = data.astype(mdescr)
                _fieldmask.flat = tuple([False]*len(mdescr))
            else:
                _fieldmask = mask
        elif (isinstance(data, (tuple, numpy.void)) or\
              hasattr(data,'__len__') and isinstance(data[0], (tuple, numpy.void))):
            data = numeric.array(data, dtype=descr).view(recarray)
            _data = data
            if mask is nomask:
                _fieldmask = data.astype(mdescr)
                _fieldmask.flat = tuple([False]*len(mdescr))
            else:
                _fieldmask = mask
        else:
            _data = recarray(shape, dtype=descr)
            _fieldmask = recarray(shape, dtype=mdescr)
            for (n,v) in zip(_names, data):
                _data[n] = numeric.asarray(v).view(ndarray)
                _fieldmask[n] = getmaskarray(v)
        #........................................
        _data = _data.view(cls)
        _data._fieldmask = _fieldmask
        _data._hardmask = hard_mask
        if fill_value is None:
            _data._fill_value = [default_fill_value(numeric.dtype(d[1]))
                                 for d in descr.descr]
        else:
            _data._fill_value = fill_value
        return _data

    def __array_finalize__(self,obj):
        if isinstance(obj, MaskedRecords):
            self.__dict__.update(_fieldmask=obj._fieldmask,
                                 _hardmask=obj._hardmask,
                                 _fill_value=obj._fill_value
                                 )
        else:
            self.__dict__.update(_fieldmask = nomask,
                                 _hardmask = False,
                                 fill_value = None
                                )
        return

    def _getdata(self):
        "Returns the data as a recarray."
        return self.view(recarray)
    _data = property(fget=_getdata)

    #......................................................
    def __getattribute__(self, attr):
        "Returns the given attribute."
        try:
            # Returns a generic attribute
            return object.__getattribute__(self,attr)
        except AttributeError:
            # OK, so attr must be a field name
            pass
        # Get the list of fields ......
        _names = self.dtype.names
        if attr in _names:
            _data = self._data
            _mask = self._fieldmask
#            obj = masked_array(_data.__getattribute__(attr), copy=False,
#                               mask=_mask.__getattribute__(attr))
            # Use a view in order to avoid the copy of the mask in MaskedArray.__new__
            obj = narray(_data.__getattribute__(attr), copy=False).view(MaskedArray)
            obj._mask = _mask.__getattribute__(attr)
            if not obj.ndim and obj._mask:
                return masked
            return obj
        raise AttributeError,"No attribute '%s' !" % attr

    def __setattr__(self, attr, val):
        "Sets the attribute attr to the value val."
        newattr = attr not in self.__dict__
        try:
            # Is attr a generic attribute ?
            ret = object.__setattr__(self, attr, val)
        except:
            # Not a generic attribute: exit if it's not a valid field
            fielddict = self.dtype.names or {}
            if attr not in fielddict:
                exctype, value = sys.exc_info()[:2]
                raise exctype, value
        else:
            if attr not in list(self.dtype.names) + ['_mask','mask']:
                return ret
            if newattr:         # We just added this one
                try:            #  or this setattr worked on an internal
                                #  attribute.
                    object.__delattr__(self, attr)
                except:
                    return ret
        # Case #1.: Basic field ............
        base_fmask = self._fieldmask
        _names = self.dtype.names
        if attr in _names:
            fval = filled(val)
            mval = getmaskarray(val)
            if self._hardmask:
                mval = mask_or(mval, base_fmask.__getattr__(attr))
            self._data.__setattr__(attr, fval)
            base_fmask.__setattr__(attr, mval)
            return
        elif attr == '_mask':
            self.__setmask__(val)
            return
    #............................................
    def __getitem__(self, indx):
        """Returns all the fields sharing the same fieldname base.
The fieldname base is either `_data` or `_mask`."""
        _localdict = self.__dict__
        _data = self._data
        # We want a field ........
        if isinstance(indx, str):
            obj = _data[indx].view(MaskedArray)
            obj._set_mask(_localdict['_fieldmask'][indx])
            # Force to nomask if the mask is empty
            if not obj._mask.any():
                obj._mask = nomask
            return obj
        # We want some elements ..
        # First, the data ........
        obj = ndarray.__getitem__(self, indx)
        if isinstance(obj, numpy.void):
            obj = self.__class__(obj, dtype=self.dtype)
        else:
            obj = obj.view(type(self))
        obj._fieldmask = numpy.asarray(_localdict['_fieldmask'][indx]).view(recarray)
        return obj
    #............................................
    def __setitem__(self, indx, value):
        "Sets the given record to value."
        MaskedArray.__setitem__(self, indx, value)


    def __setslice__(self, i, j, value):
        "Sets the slice described by [i,j] to `value`."
        _localdict = self.__dict__
        d = self._data
        m = _localdict['_fieldmask']
        names = self.dtype.names
        if value is masked:
            for n in names:
                m[i:j][n] = True
        elif not self._hardmask:
            fval = filled(value)
            mval = getmaskarray(value)
            for n in names:
                d[n][i:j] = fval
                m[n][i:j] = mval
        else:
            mindx = getmaskarray(self)[i:j]
            dval = numeric.asarray(value)
            valmask = getmask(value)
            if valmask is nomask:
                for n in names:
                    mval = mask_or(m[n][i:j], valmask)
                    d[n][i:j][~mval] = value
            elif valmask.size > 1:
                for n in names:
                    mval = mask_or(m[n][i:j], valmask)
                    d[n][i:j][~mval] = dval[~mval]
                    m[n][i:j] = mask_or(m[n][i:j], mval)
        self._fieldmask = m

    #.....................................................
    def __setmask__(self, mask):
        "Sets the mask."
        names = self.dtype.names
        fmask = self.__dict__['_fieldmask']
        newmask = make_mask(mask, copy=False)
#        self.unshare_mask()
        if self._hardmask:
            for n in names:
                fmask[n].__ior__(newmask)
        else:
            for n in names:
                fmask[n].flat = newmask
        return

    def _getmask(self):
        """Returns the mask of the mrecord: a record is masked when all the fields
are masked."""
        if self.size > 1:
            return self._fieldmask.view((bool_, len(self.dtype))).all(1)

    _setmask = __setmask__
    _mask = property(fget=_getmask, fset=_setmask)

    #......................................................
    def __str__(self):
        "Calculates the string representation."
        if self.size > 1:
            mstr = ["(%s)" % ",".join([str(i) for i in s])
                    for s in zip(*[getattr(self,f) for f in self.dtype.names])]
            return "[%s]" % ", ".join(mstr)
        else:
            mstr = ["%s" % ",".join([str(i) for i in s])
                    for s in zip([getattr(self,f) for f in self.dtype.names])]
            return "(%s)" % ", ".join(mstr)

    def __repr__(self):
        "Calculates the repr representation."
        _names = self.dtype.names
        fmt = "%%%is : %%s" % (max([len(n) for n in _names])+4,)
        reprstr = [fmt % (f,getattr(self,f)) for f in self.dtype.names]
        reprstr.insert(0,'masked_records(')
        reprstr.extend([fmt % ('    fill_value', self._fill_value),
                         '              )'])
        return str("\n".join(reprstr))
    #......................................................
    def view(self, obj):
        """Returns a view of the mrecarray."""
        try:
            if issubclass(obj, ndarray):
                return ndarray.view(self, obj)
        except TypeError:
            pass
        dtype = numeric.dtype(obj)
        if dtype.fields is None:
            return self.__array__().view(dtype)
        return ndarray.view(self, obj)
    #......................................................
    def filled(self, fill_value=None):
        """Returns an array of the same class as ``_data``, with masked values
filled with ``fill_value``. If ``fill_value`` is None, ``self.fill_value`` is
used instead.

Subclassing is preserved.

        """
        _localdict = self.__dict__
        d = self._data
        fm = _localdict['_fieldmask']
        if not numeric.asarray(fm, dtype=bool_).any():
            return d
        #
        if fill_value is None:
            value = _localdict['_fill_value']
        else:
            value = fill_value
            if numeric.size(value) == 1:
                value = [value,] * len(self.dtype)
        #
        if self is masked:
            result = numeric.asanyarray(value)
        else:
            result = d.copy()
            for (n, v) in zip(d.dtype.names, value):
                numpy.putmask(numeric.asarray(result[n]),
                              numeric.asarray(fm[n]), v)
        return result
    #............................................
    def harden_mask(self):
        "Forces the mask to hard"
        self._hardmask = True
    def soften_mask(self):
        "Forces the mask to soft"
        self._hardmask = False
    #.............................................
    def copy(self):
        """Returns a copy of the masked record."""
        _localdict = self.__dict__
        return MaskedRecords(self._data.copy(),
                        mask=_localdict['_fieldmask'].copy(),
                       dtype=self.dtype)
    #.............................................


#####---------------------------------------------------------------------------
#---- --- Constructors ---
#####---------------------------------------------------------------------------

def fromarrays(arraylist, dtype=None, shape=None, formats=None,
               names=None, titles=None, aligned=False, byteorder=None):
    """Creates a mrecarray from a (flat) list of masked arrays.

*Parameters*:
    arraylist : {sequence}
        A list of (masked) arrays. Each element of the sequence is first converted
        to a masked array if needed. If a 2D array is passed as argument, it is
        processed line by line
    dtype : {numeric.dtype}
        Data type descriptor.
    {shape} : {integer}
        Number of records. If None, ``shape`` is defined from the shape of the
        first array in the list.
    formats : {sequence}
        Sequence of formats for each individual field. If None, the formats will
        be autodetected by inspecting the fields and selecting the highest dtype
        possible.
    names : {sequence}
        Sequence of the names of each field.
    -titles : {sequence}
      (Description to write)
    aligned : {boolean}
      (Description to write, not used anyway)
    byteorder: {boolean}
      (Description to write, not used anyway)

*Notes*:
    Lists of tuples should be preferred over lists of lists for faster processing.
    """
    arraylist = [masked_array(x) for x in arraylist]
    # Define/check the shape.....................
    if shape is None or shape == 0:
        shape = arraylist[0].shape
    if isinstance(shape, int):
        shape = (shape,)
    # Define formats from scratch ...............
    if formats is None and dtype is None:
        formats = _getformats(arraylist)
    # Define the dtype ..........................
    if dtype is not None:
        descr = numeric.dtype(dtype)
        _names = descr.names
    else:
        parsed = format_parser(formats, names, titles, aligned, byteorder)
        _names = parsed._names
        descr = parsed._descr
    # Determine shape from data-type.............
    if len(descr) != len(arraylist):
        msg = "Mismatch between the number of fields (%i) and the number of "\
              "arrays (%i)"
        raise ValueError, msg % (len(descr), len(arraylist))
    d0 = descr[0].shape
    nn = len(d0)
    if nn > 0:
        shape = shape[:-nn]
    # Make sure the shape is the correct one ....
    for k, obj in enumerate(arraylist):
        nn = len(descr[k].shape)
        testshape = obj.shape[:len(obj.shape)-nn]
        if testshape != shape:
            raise ValueError, "Array-shape mismatch in array %d" % k
    # Reconstruct the descriptor, by creating a _data and _mask version
    return MaskedRecords(arraylist, dtype=descr)
#..............................................................................
def fromrecords(reclist, dtype=None, shape=None, formats=None, names=None,
                titles=None, aligned=False, byteorder=None):
    """Creates a MaskedRecords from a list of records.

*Parameters*:
    arraylist : {sequence}
        A list of (masked) arrays. Each element of the sequence is first converted
        to a masked array if needed. If a 2D array is passed as argument, it is
        processed line by line
    dtype : {numeric.dtype}
        Data type descriptor.
    {shape} : {integer}
        Number of records. If None, ``shape`` is defined from the shape of the
        first array in the list.
    formats : {sequence}
        Sequence of formats for each individual field. If None, the formats will
        be autodetected by inspecting the fields and selecting the highest dtype
        possible.
    names : {sequence}
        Sequence of the names of each field.
    -titles : {sequence}
      (Description to write)
    aligned : {boolean}
      (Description to write, not used anyway)
    byteorder: {boolean}
      (Description to write, not used anyway)

*Notes*:
    Lists of tuples should be preferred over lists of lists for faster processing.
    """
    # reclist is in fact a mrecarray .................
    if isinstance(reclist, MaskedRecords):
        mdescr = reclist.dtype
        shape = reclist.shape
        return MaskedRecords(reclist, dtype=mdescr)
    # No format, no dtype: create from to arrays .....
    nfields = len(reclist[0])
    if formats is None and dtype is None:  # slower
        if isinstance(reclist, recarray):
            arrlist = [reclist.field(i) for i in range(len(reclist.dtype))]
            if names is None:
                names = reclist.dtype.names
        else:
            obj = numeric.array(reclist,dtype=object)
            arrlist = [numeric.array(obj[...,i].tolist())
                               for i in xrange(nfields)]
        return MaskedRecords(arrlist, formats=formats, names=names,
                             titles=titles, aligned=aligned, byteorder=byteorder)
    # Construct the descriptor .......................
    if dtype is not None:
        descr = numeric.dtype(dtype)
        _names = descr.names
    else:
        parsed = format_parser(formats, names, titles, aligned, byteorder)
        _names = parsed._names
        descr = parsed._descr

    try:
        retval = numeric.array(reclist, dtype = descr).view(recarray)
    except TypeError:  # list of lists instead of list of tuples
        if (shape is None or shape == 0):
            shape = len(reclist)*2
        if isinstance(shape, (int, long)):
            shape = (shape*2,)
        if len(shape) > 1:
            raise ValueError, "Can only deal with 1-d array."
        retval = recarray(shape, mdescr)
        for k in xrange(retval.size):
            retval[k] = tuple(reclist[k])
        return MaskedRecords(retval, dtype=descr)
    else:
        if shape is not None and retval.shape != shape:
            retval.shape = shape
    #
    return MaskedRecords(retval, dtype=descr)

def _guessvartypes(arr):
    """Tries to guess the dtypes of the str_ ndarray `arr`, by testing element-wise
conversion. Returns a list of dtypes.
The array is first converted to ndarray. If the array is 2D, the test is performed
on the first line. An exception is raised if the file is 3D or more.
    """
    vartypes = []
    arr = numeric.asarray(arr)
    if len(arr.shape) == 2 :
        arr = arr[0]
    elif len(arr.shape) > 2:
        raise ValueError, "The array should be 2D at most!"
    # Start the conversion loop .......
    for f in arr:
        try:
            val = int(f)
        except ValueError:
            try:
                val = float(f)
            except ValueError:
                try:
                    val = complex(f)
                except ValueError:
                    vartypes.append(arr.dtype)
                else:
                    vartypes.append(complex_)
            else:
                vartypes.append(float_)
        else:
            vartypes.append(int_)
    return vartypes

def openfile(fname):
    "Opens the file handle of file `fname`"
    # A file handle ...................
    if hasattr(fname, 'readline'):
        return fname
    # Try to open the file and guess its type
    try:
        f = open(fname)
    except IOError:
        raise IOError, "No such file: '%s'" % fname
    if f.readline()[:2] != "\\x":
        f.seek(0,0)
        return f
    raise NotImplementedError, "Wow, binary file"


def fromtextfile(fname, delimitor=None, commentchar='#', missingchar='',
                 varnames=None, vartypes=None):
    """Creates a mrecarray from data stored in the file `filename`.

*Parameters* :
    filename : {file name/handle}
        Handle of an opened file.
    delimitor : {string}
        Alphanumeric character used to separate columns in the file.
        If None, any (group of) white spacestring(s) will be used.
    commentchar : {string}
        Alphanumeric character used to mark the start of a comment.
    missingchar` : {string}
        String indicating missing data, and used to create the masks.
    varnames : {sequence}
        Sequence of the variable names. If None, a list will be created from
        the first non empty line of the file.
    vartypes : {sequence}
        Sequence of the variables dtypes. If None, it will be estimated from
        the first non-commented line.


    Ultra simple: the varnames are in the header, one line"""
    # Try to open the file ......................
    f = openfile(fname)
    # Get the first non-empty line as the varnames
    while True:
        line = f.readline()
        firstline = line[:line.find(commentchar)].strip()
        _varnames = firstline.split(delimitor)
        if len(_varnames) > 1:
            break
    if varnames is None:
        varnames = _varnames
    # Get the data ..............................
    _variables = masked_array([line.strip().split(delimitor) for line in f
                                  if line[0] != commentchar and len(line) > 1])
    (_, nfields) = _variables.shape
    # Try to guess the dtype ....................
    if vartypes is None:
        vartypes = _guessvartypes(_variables[0])
    else:
        vartypes = [numeric.dtype(v) for v in vartypes]
        if len(vartypes) != nfields:
            msg = "Attempting to %i dtypes for %i fields!"
            msg += " Reverting to default."
            warnings.warn(msg % (len(vartypes), nfields))
            vartypes = _guessvartypes(_variables[0])
    # Construct the descriptor ..................
    mdescr = [(n,f) for (n,f) in zip(varnames, vartypes)]
    # Get the data and the mask .................
    # We just need a list of masked_arrays. It's easier to create it like that:
    _mask = (_variables.T == missingchar)
    _datalist = [masked_array(a,mask=m,dtype=t)
                     for (a,m,t) in zip(_variables.T, _mask, vartypes)]
    return MaskedRecords(_datalist, dtype=mdescr)

#....................................................................
def addfield(mrecord, newfield, newfieldname=None):
    """Adds a new field to the masked record array, using `newfield` as data
and `newfieldname` as name. If `newfieldname` is None, the new field name is
set to 'fi', where `i` is the number of existing fields.
    """
    _data = mrecord._data
    _mask = mrecord._fieldmask
    if newfieldname is None or newfieldname in reserved_fields:
        newfieldname = 'f%i' % len(_data.dtype)
    newfield = masked_array(newfield)
    # Get the new data ............
    # Create a new empty recarray
    newdtype = numeric.dtype(_data.dtype.descr + \
                             [(newfieldname, newfield.dtype)])
    newdata = recarray(_data.shape, newdtype)
    # Add the exisintg field
    [newdata.setfield(_data.getfield(*f),*f)
         for f in _data.dtype.fields.values()]
    # Add the new field
    newdata.setfield(newfield._data, *newdata.dtype.fields[newfieldname])
    newdata = newdata.view(MaskedRecords)
    # Get the new mask .............
    # Create a new empty recarray
    newmdtype = numeric.dtype([(n,bool_) for n in newdtype.names])
    newmask = recarray(_data.shape, newmdtype)
    # Add the old masks
    [newmask.setfield(_mask.getfield(*f),*f)
         for f in _mask.dtype.fields.values()]
    # Add the mask of the new field
    newmask.setfield(getmaskarray(newfield),
                     *newmask.dtype.fields[newfieldname])
    newdata._fieldmask = newmask
    return newdata

################################################################################
if __name__ == '__main__':
    import numpy as N
    from numpy.ma.testutils import assert_equal
    if 1:
        d = N.arange(5)
        m = numpy.ma.make_mask([1,0,0,1,1])
        base_d = N.r_[d,d[::-1]].reshape(2,-1).T
        base_m = N.r_[[m, m[::-1]]].T
        base = masked_array(base_d, mask=base_m).T
        mrecord = fromarrays(base,dtype=[('a',N.float_),('b',N.float_)])
        mrec = MaskedRecords(mrecord.copy())
        #
    if 1:
        mrec = mrec.copy()
        mrec.harden_mask()
        assert(mrec._hardmask)
        mrec._mask = nomask
        assert_equal(mrec._mask, N.r_[[m,m[::-1]]].all(0))
        mrec.soften_mask()
        assert(not mrec._hardmask)
        mrec.mask = nomask
        tmp = mrec['b']._mask
        assert(mrec['b']._mask is nomask)
        assert_equal(mrec['a']._mask,mrec['b']._mask)
