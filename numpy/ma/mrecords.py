"""mrecords

Defines the equivalent of recarrays for maskedarray.
Masked arrays already support named fields, but masking works only by records.
By comparison, mrecarrays support masking individual fields.

:author: Pierre Gerard-Marchant
"""
#TODO: We should make sure that no field is called '_mask','mask','_fieldmask',
#TODO: ...or whatever restricted keywords.
#TODO: An idea would be to no bother in the first place, and then rename the
#TODO: invalid fields with a trailing underscore...
#TODO: Maybe we could just overload the parser function ?


__author__ = "Pierre GF Gerard-Marchant"

import sys
import types

import numpy as np
from numpy import bool_, complex_, float_, int_, str_, object_, dtype, \
    chararray, ndarray, recarray, record, array as narray
import numpy.core.numerictypes as ntypes
from numpy.core.records import find_duplicate, format_parser
from numpy.core.records import fromarrays as recfromarrays, \
    fromrecords as recfromrecords

_byteorderconv = np.core.records._byteorderconv
_typestr = ntypes._typestr

import numpy.ma as ma
from numpy.ma import MAError, MaskedArray, masked, nomask, masked_array,\
    make_mask, mask_or, getdata, getmask, getmaskarray, filled, \
    default_fill_value, masked_print_option
_check_fill_value = ma.core._check_fill_value

import warnings

__all__ = ['MaskedRecords','mrecarray',
           'fromarrays','fromrecords','fromtextfile','addfield',
           ]

reserved_fields = ['_data','_mask','_fieldmask', 'dtype']

def _getformats(data):
    "Returns the formats of each array of arraylist as a comma-separated string."
    if hasattr(data,'dtype'):
        return ",".join([desc[1] for desc in data.dtype.descr])

    formats = ''
    for obj in data:
        obj = np.asarray(obj)
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


def _get_fieldmask(self):
    mdescr = [(n,'|b1') for n in self.dtype.names]
    fdmask = np.empty(self.shape, dtype=mdescr)
    fdmask.flat = tuple([False]*len(mdescr))
    return fdmask


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
    #............................................
    def __new__(cls, shape, dtype=None, buf=None, offset=0, strides=None,
                formats=None, names=None, titles=None,
                byteorder=None, aligned=False,
                mask=nomask, hard_mask=False, fill_value=None, keep_mask=True,
                copy=False,
                **options):
        #
        self = recarray.__new__(cls, shape, dtype=dtype, buf=buf, offset=offset,
                                strides=strides, formats=formats,
                                byteorder=byteorder, aligned=aligned,)
#        self = self.view(cls)
        #
        mdtype = [(k,'|b1') for (k,_) in self.dtype.descr]
        if mask is nomask or not np.size(mask):
            if not keep_mask:
                self._fieldmask = tuple([False]*len(mdtype))
        else:
            mask = np.array(mask, copy=copy)
            if mask.shape != self.shape:
                (nd, nm) = (self.size, mask.size)
                if nm == 1:
                    mask = np.resize(mask, self.shape)
                elif nm == nd:
                    mask = np.reshape(mask, self.shape)
                else:
                    msg = "Mask and data not compatible: data size is %i, "+\
                          "mask size is %i."
                    raise MAError(msg % (nd, nm))
                copy = True
            if not keep_mask:
                self.__setmask__(mask)
                self._sharedmask = True
            else:
                if mask.dtype == mdtype:
                    _fieldmask = mask
                else:
                    _fieldmask = np.array([tuple([m]*len(mdtype)) for m in mask],
                                          dtype=mdtype)
                self._fieldmask = _fieldmask
        return self
    #......................................................
    def __array_finalize__(self,obj):
        # Make sure we have a _fieldmask by default ..
        _fieldmask = getattr(obj, '_fieldmask', None)
        if _fieldmask is None:
            mdescr = [(n,'|b1') for (n,_) in self.dtype.descr]
            _mask = getattr(obj, '_mask', nomask)
            if _mask is nomask:
                _fieldmask = np.empty(self.shape, dtype=mdescr).view(recarray)
                _fieldmask.flat = tuple([False]*len(mdescr))
            else:
                _fieldmask = narray([tuple([m]*len(mdescr)) for m in _mask],
                                    dtype=mdescr).view(recarray)
        # Update some of the attributes
        if obj is not None:
            _baseclass = getattr(obj,'_baseclass',type(obj))
        else:
            _baseclass = recarray
        attrdict = dict(_fieldmask=_fieldmask,
                        _hardmask=getattr(obj,'_hardmask',False),
                        _fill_value=getattr(obj,'_fill_value',None),
                        _sharedmask=getattr(obj,'_sharedmask',False),
                        _baseclass=_baseclass)
        self.__dict__.update(attrdict)
        # Finalize as a regular maskedarray .....
        # Update special attributes ...
        self._basedict = getattr(obj, '_basedict', getattr(obj,'__dict__',None))
        if self._basedict is not None:
            self.__dict__.update(self._basedict)
        return
    #......................................................
    def _getdata(self):
        "Returns the data as a recarray."
        return ndarray.view(self,recarray)
    _data = property(fget=_getdata)
    #......................................................
    def __setmask__(self, mask):
        "Sets the mask and update the fieldmask."
        names = self.dtype.names
        fmask = self.__dict__['_fieldmask']
        #
        if isinstance(mask,ndarray) and mask.dtype.names == names:
            for n in names:
                fmask[n] = mask[n].astype(bool)
#            self.__dict__['_fieldmask'] = fmask.view(recarray)
            return
        newmask = make_mask(mask, copy=False)
        if names is not None:
            if self._hardmask:
                for n in names:
                    fmask[n].__ior__(newmask)
            else:
                for n in names:
                    fmask[n].flat = newmask
        return
    _setmask = __setmask__
    #
    def _getmask(self):
        """Return the mask of the mrecord.
    A record is masked when all the fields are masked.

        """
        if self.size > 1:
            return self._fieldmask.view((bool_, len(self.dtype))).all(1)
        else:
            return self._fieldmask.view((bool_, len(self.dtype))).all()
    mask = _mask = property(fget=_getmask, fset=_setmask)
    #......................................................
    def get_fill_value(self):
        """Return the filling value.

        """
        if self._fill_value is None:
            ddtype = self.dtype
            fillval = _check_fill_value(None, ddtype)
            self._fill_value = np.array(tuple(fillval), dtype=ddtype)
        return self._fill_value

    def set_fill_value(self, value=None):
        """Set the filling value to value.

        If value is None, use a default based on the data type.

        """
        ddtype = self.dtype
        fillval = _check_fill_value(value, ddtype)
        self._fill_value = np.array(tuple(fillval), dtype=ddtype)

    fill_value = property(fget=get_fill_value, fset=set_fill_value,
                          doc="Filling value.")
    #......................................................
    def __len__(self):
        "Returns the length"
        # We have more than one record
        if self.ndim:
            return len(self._data)
        # We have only one record: return the nb of fields
        return len(self.dtype)
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
#        newattr = attr not in self.__dict__
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
            if attr in ['_mask','fieldmask']:
                self.__setmask__(val)
                return
            # Get the list of names ......
            _names = self.dtype.names
            if _names is None:
                _names = []
            else:
                _names = list(_names)
            # Check the attribute
            self_dict = self.__dict__
            if attr not in _names+list(self_dict):
                return ret
            if attr not in self_dict:         # We just added this one
                try:            #  or this setattr worked on an internal
                                #  attribute.
                    object.__delattr__(self, attr)
                except:
                    return ret
        # Case #1.: Basic field ............
        base_fmask = self._fieldmask
        _names = self.dtype.names or []
        if attr in _names:
            if val is masked:
                fval = self.fill_value[attr]
                mval = True
            else:
                fval = filled(val)
                mval = getmaskarray(val)
            if self._hardmask:
                mval = mask_or(mval, base_fmask.__getattr__(attr))
            self._data.__setattr__(attr, fval)
            base_fmask.__setattr__(attr, mval)
            return
    #............................................
    def __getitem__(self, indx):
        """Returns all the fields sharing the same fieldname base.
The fieldname base is either `_data` or `_mask`."""
        _localdict = self.__dict__
        _fieldmask = _localdict['_fieldmask']
        _data = self._data
        # We want a field ........
        if isinstance(indx, basestring):
            obj = _data[indx].view(MaskedArray)
            obj._set_mask(_fieldmask[indx])
            # Force to nomask if the mask is empty
            if not obj._mask.any():
                obj._mask = nomask
            # Force to masked if the mask is True
            if not obj.ndim and obj._mask:
                return masked
            return obj
        # We want some elements ..
        # First, the data ........
        obj = narray(_data[indx], copy=False).view(mrecarray)
        obj._fieldmask = narray(_fieldmask[indx], copy=False).view(recarray)
        return obj
    #....
    def __setitem__(self, indx, value):
        "Sets the given record to value."
        MaskedArray.__setitem__(self, indx, value)
    #............................................
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
            dval = np.asarray(value)
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
    #
    def __repr__(self):
        "Calculates the repr representation."
        _names = self.dtype.names
        fmt = "%%%is : %%s" % (max([len(n) for n in _names])+4,)
        reprstr = [fmt % (f,getattr(self,f)) for f in self.dtype.names]
        reprstr.insert(0,'masked_records(')
        reprstr.extend([fmt % ('    fill_value', self.fill_value),
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
        dtype = np.dtype(obj)
        if dtype.fields is None:
            return self.__array__().view(dtype)
        return ndarray.view(self, obj)
    #......................................................
    def filled(self, fill_value=None):
        """Returns an array of the same class as the _data part, where masked
    values are filled with fill_value.
    If fill_value is None, self.fill_value is used instead.

    Subclassing is preserved.

        """
        _localdict = self.__dict__
        d = self._data
        fm = _localdict['_fieldmask']
        if not np.asarray(fm, dtype=bool_).any():
            return d
        #
        if fill_value is None:
            value = _check_fill_value(_localdict['_fill_value'],self.dtype)
        else:
            value = fill_value
            if np.size(value) == 1:
                value = [value,] * len(self.dtype)
        #
        if self is masked:
            result = np.asanyarray(value)
        else:
            result = d.copy()
            for (n, v) in zip(d.dtype.names, value):
                np.putmask(np.asarray(result[n]), np.asarray(fm[n]), v)
        return result
    #......................................................
    def harden_mask(self):
        "Forces the mask to hard"
        self._hardmask = True
    def soften_mask(self):
        "Forces the mask to soft"
        self._hardmask = False
    #......................................................
    def copy(self):
        """Returns a copy of the masked record."""
        _localdict = self.__dict__
        copied = self._data.copy().view(type(self))
        copied._fieldmask = self._fieldmask.copy()
        return copied
    #......................................................
    def tolist(self, fill_value=None):
        """Copy the data portion of the array to a hierarchical python
        list and returns that list.

        Data items are converted to the nearest compatible Python
        type.  Masked values are converted to fill_value. If
        fill_value is None, the corresponding entries in the output
        list will be ``None``.

        """
        if fill_value is not None:
            return self.filled(fill_value).tolist()
        result = narray(self.filled().tolist(), dtype=object)
        mask = narray(self._fieldmask.tolist())
        result[mask] = None
        return result.tolist()
    #--------------------------------------------
    # Pickling
    def __getstate__(self):
        """Return the internal state of the masked array, for pickling purposes.

        """
        state = (1,
                 self.shape,
                 self.dtype,
                 self.flags.fnc,
                 self._data.tostring(),
                 self._fieldmask.tostring(),
                 self._fill_value,
                 )
        return state
    #
    def __setstate__(self, state):
        """Restore the internal state of the masked array, for pickling purposes.
    ``state`` is typically the output of the ``__getstate__`` output, and is a
    5-tuple:

        - class name
        - a tuple giving the shape of the data
        - a typecode for the data
        - a binary string for the data
        - a binary string for the mask.

        """
        (ver, shp, typ, isf, raw, msk, flv) = state
        ndarray.__setstate__(self, (shp, typ, isf, raw))
        mdtype = dtype([(k,bool_) for (k,_) in self.dtype.descr])
        self.__dict__['_fieldmask'].__setstate__((shp, mdtype, isf, msk))
        self.fill_value = flv
    #
    def __reduce__(self):
        """Return a 3-tuple for pickling a MaskedArray.

        """
        return (_mrreconstruct,
                (self.__class__, self._baseclass, (0,), 'b', ),
                self.__getstate__())

def _mrreconstruct(subtype, baseclass, baseshape, basetype,):
    """Internal function that builds a new MaskedArray from the
    information stored in a pickle.

    """
    _data = ndarray.__new__(baseclass, baseshape, basetype).view(subtype)
#    _data._mask = ndarray.__new__(ndarray, baseshape, 'b1')
#    return _data
    _mask = ndarray.__new__(ndarray, baseshape, 'b1')
    return subtype.__new__(subtype, _data, mask=_mask, dtype=basetype,)


mrecarray = MaskedRecords

#####---------------------------------------------------------------------------
#---- --- Constructors ---
#####---------------------------------------------------------------------------

def fromarrays(arraylist, dtype=None, shape=None, formats=None,
               names=None, titles=None, aligned=False, byteorder=None,
               fill_value=None):
    """Creates a mrecarray from a (flat) list of masked arrays.

    Parameters
    ----------
    arraylist : sequence
        A list of (masked) arrays. Each element of the sequence is first converted
        to a masked array if needed. If a 2D array is passed as argument, it is
        processed line by line
    dtype : numeric.dtype
        Data type descriptor.
    shape : integer
        Number of records. If None, shape is defined from the shape of the
        first array in the list.
    formats : sequence
        Sequence of formats for each individual field. If None, the formats will
        be autodetected by inspecting the fields and selecting the highest dtype
        possible.
    names : sequence
        Sequence of the names of each field.
    titles : sequence
      (Description to write)
    aligned : boolean
      (Description to write, not used anyway)
    byteorder: boolean
      (Description to write, not used anyway)
    fill_value : sequence
        Sequence of data to be used as filling values.

    Notes
    -----
    Lists of tuples should be preferred over lists of lists for faster processing.
    """
    datalist = [getdata(x) for x in arraylist]
    masklist = [getmaskarray(x) for x in arraylist]
    _array = recfromarrays(datalist,
                           dtype=dtype, shape=shape, formats=formats,
                           names=names, titles=titles, aligned=aligned,
                           byteorder=byteorder).view(mrecarray)
    _array._fieldmask[:] = zip(*masklist)
    if fill_value is not None:
        _array.fill_value = fill_value
    return _array


#..............................................................................
def fromrecords(reclist, dtype=None, shape=None, formats=None, names=None,
                titles=None, aligned=False, byteorder=None,
                fill_value=None, mask=nomask):
    """Creates a MaskedRecords from a list of records.

    Parameters
    ----------
    arraylist : sequence
        A list of (masked) arrays. Each element of the sequence is first converted
        to a masked array if needed. If a 2D array is passed as argument, it is
        processed line by line
    dtype : numeric.dtype
        Data type descriptor.
    shape : integer
        Number of records. If None, ``shape`` is defined from the shape of the
        first array in the list.
    formats : sequence
        Sequence of formats for each individual field. If None, the formats will
        be autodetected by inspecting the fields and selecting the highest dtype
        possible.
    names : sequence
        Sequence of the names of each field.
    titles : sequence
      (Description to write)
    aligned : boolean
      (Description to write, not used anyway)
    byteorder: boolean
      (Description to write, not used anyway)
    fill_value : sequence
        Sequence of data to be used as filling values.
    mask : sequence or boolean.
        External mask to apply on the data.

*Notes*:
    Lists of tuples should be preferred over lists of lists for faster processing.
    """
    # Grab the initial _fieldmask, if needed:
    _fieldmask = getattr(reclist, '_fieldmask', None)
    # Get the list of records.....
    nfields = len(reclist[0])
    if isinstance(reclist, ndarray):
        # Make sure we don't have some hidden mask
        if isinstance(reclist,MaskedArray):
            reclist = reclist.filled().view(ndarray)
        # Grab the initial dtype, just in case
        if dtype is None:
            dtype = reclist.dtype
        reclist = reclist.tolist()
    mrec = recfromrecords(reclist, dtype=dtype, shape=shape, formats=formats,
                          names=names, titles=titles,
                          aligned=aligned, byteorder=byteorder).view(mrecarray)
    # Set the fill_value if needed
    if fill_value is not None:
        mrec.fill_value = fill_value
    # Now, let's deal w/ the mask
    if mask is not nomask:
        mask = np.array(mask, copy=False)
        maskrecordlength = len(mask.dtype)
        if maskrecordlength:
            mrec._fieldmask.flat = mask
        elif len(mask.shape) == 2:
            mrec._fieldmask.flat = [tuple(m) for m in mask]
        else:
            mrec._mask = mask
    if _fieldmask is not None:
        mrec._fieldmask[:] = _fieldmask
    return mrec

def _guessvartypes(arr):
    """Tries to guess the dtypes of the str_ ndarray `arr`, by testing element-wise
conversion. Returns a list of dtypes.
The array is first converted to ndarray. If the array is 2D, the test is performed
on the first line. An exception is raised if the file is 3D or more.
    """
    vartypes = []
    arr = np.asarray(arr)
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
                    vartypes.append(complex)
            else:
                vartypes.append(float)
        else:
            vartypes.append(int)
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
        vartypes = [np.dtype(v) for v in vartypes]
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
    return fromarrays(_datalist, dtype=mdescr)

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
    newfield = ma.array(newfield)
    # Get the new data ............
    # Create a new empty recarray
    newdtype = np.dtype(_data.dtype.descr + [(newfieldname, newfield.dtype)])
    newdata = recarray(_data.shape, newdtype)
    # Add the exisintg field
    [newdata.setfield(_data.getfield(*f),*f)
         for f in _data.dtype.fields.values()]
    # Add the new field
    newdata.setfield(newfield._data, *newdata.dtype.fields[newfieldname])
    newdata = newdata.view(MaskedRecords)
    # Get the new mask .............
    # Create a new empty recarray
    newmdtype = np.dtype([(n,bool_) for n in newdtype.names])
    newmask = recarray(_data.shape, newmdtype)
    # Add the old masks
    [newmask.setfield(_mask.getfield(*f),*f)
         for f in _mask.dtype.fields.values()]
    # Add the mask of the new field
    newmask.setfield(getmaskarray(newfield),
                     *newmask.dtype.fields[newfieldname])
    newdata._fieldmask = newmask
    return newdata

###############################################################################
#
if 1:
    import numpy.ma as ma
    from numpy.ma.testutils import *
    if 1:
        ilist = [1,2,3,4,5]
        flist = [1.1,2.2,3.3,4.4,5.5]
        slist = ['one','two','three','four','five']
        ddtype = [('a',int),('b',float),('c','|S8')]
        mask = [0,1,0,0,1]
        self_base = ma.array(zip(ilist,flist,slist), mask=mask, dtype=ddtype)
    if 1:
        mbase = self_base.copy().view(mrecarray)
        import cPickle
        _ = cPickle.dumps(mbase)
        mrec_ = cPickle.loads(_)
