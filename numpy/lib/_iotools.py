"""
A collection of functions designed to help I/O with ascii file.

"""
__docformat__ = "restructuredtext en"

import numpy as np
import numpy.core.numeric as nx
from __builtin__ import bool, int, long, float, complex, object, unicode, str


def _is_string_like(obj):
    """
    Check whether obj behaves like a string.
    """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True


def _to_filehandle(fname, flag='r', return_opened=False):
    """
    Returns the filehandle corresponding to a string or a file.
    If the string ends in '.gz', the file is automatically unzipped.
    
    Parameters
    ----------
    fname : string, filehandle
        Name of the file whose filehandle must be returned.
    flag : string, optional
        Flag indicating the status of the file ('r' for read, 'w' for write).
    return_opened : boolean, optional
        Whether to return the opening status of the file.
    """
    if _is_string_like(fname):
        if fname.endswith('.gz'):
            import gzip
            fhd = gzip.open(fname, flag)
        elif fname.endswith('.bz2'):
            import bz2
            fhd = bz2.BZ2File(fname)
        else:
            fhd = file(fname, flag)
        opened = True
    elif hasattr(fname, 'seek'):
        fhd = fname
        opened = False
    else:
        raise ValueError('fname must be a string or file handle')
    if return_opened:
        return fhd, opened
    return fhd


def has_nested_fields(ndtype):
    """
    Returns whether one or several fields of a structured array are nested.
    """
    for name in ndtype.names or ():
        if ndtype[name].names:
            return True
    return False


def flatten_dtype(ndtype, flatten_base=False):
    """
    Unpack a structured data-type by collapsing nested fields and/or fields with
    a shape.

    Note that the field names are lost.

    Parameters
    ----------
    ndtype : dtype
        The datatype to collapse
    flatten_base : {False, True}, optional
        Whether to transform a field with a shape into several fields or not.

    Examples
    --------
    >>> dt = np.dtype([('name', 'S4'), ('x', float), ('y', float),
   ...                 ('block', int, (2, 3))])
    >>> flatten_dtype(dt)
     [dtype('|S4'), dtype('float64'), dtype('float64'), dtype(('int32',(2, 3)))]
    >>> flatten_dtype(dt, flatten_base=True)
    [dtype('|S4'), dtype('float64'), dtype('float64'), dtype('int32'),
     dtype('int32'), dtype('int32'), dtype('int32'), dtype('int32'),
     dtype('int32')]
    """
    names = ndtype.names
    if names is None:
        if flatten_base:
            return [ndtype.base] * int(np.prod(ndtype.shape))
        return [ndtype.base]
    else:
        types = []
        for field in names:
            (typ, _) = ndtype.fields[field]
            flat_dt = flatten_dtype(typ, flatten_base)
            types.extend(flat_dt)
        return types



class LineSplitter:
    """
    Defines a function to split a string at a given delimiter or at given places.
    
    Parameters
    ----------
    comment : {'#', string}
        Character used to mark the beginning of a comment.
    delimiter : var, optional
        If a string, character used to delimit consecutive fields.
        If an integer or a sequence of integers, width(s) of each field.
    autostrip : boolean, optional
        Whether to strip each individual fields
    """

    def autostrip(self, method):
        "Wrapper to strip each member of the output of `method`."
        return lambda input: [_.strip() for _ in method(input)]
    #
    def __init__(self, delimiter=None, comments='#', autostrip=True):
        self.comments = comments
        # Delimiter is a character
        if (delimiter is None) or _is_string_like(delimiter):
            delimiter = delimiter or None
            _handyman = self._delimited_splitter
        # Delimiter is a list of field widths
        elif hasattr(delimiter, '__iter__'):
            _handyman = self._variablewidth_splitter
            idx = np.cumsum([0]+list(delimiter))
            delimiter = [slice(i,j) for (i,j) in zip(idx[:-1], idx[1:])]
        # Delimiter is a single integer
        elif int(delimiter):
            (_handyman, delimiter) = (self._fixedwidth_splitter, int(delimiter))
        else:
            (_handyman, delimiter) = (self._delimited_splitter, None)
        self.delimiter = delimiter
        if autostrip:
            self._handyman = self.autostrip(_handyman)
        else:
            self._handyman = _handyman
    #
    def _delimited_splitter(self, line):
        line = line.split(self.comments)[0].strip()
        if not line:
            return []
        return line.split(self.delimiter)
    #
    def _fixedwidth_splitter(self, line):
        line = line.split(self.comments)[0]
        if not line:
            return []
        fixed = self.delimiter
        slices = [slice(i, i+fixed) for i in range(len(line))[::fixed]]
        return [line[s] for s in slices]
    #
    def _variablewidth_splitter(self, line):
        line = line.split(self.comments)[0]
        if not line:
            return []
        slices = self.delimiter
        return [line[s] for s in slices]
    #
    def __call__(self, line):
        return self._handyman(line)



class NameValidator:
    """
    Validates a list of strings to use as field names.
    The strings are stripped of any non alphanumeric character, and spaces
    are replaced by `_`. If the optional input parameter `case_sensitive`
    is False, the strings are set to upper case.

    During instantiation, the user can define a list of names to exclude, as 
    well as a list of invalid characters. Names in the exclusion list
    are appended a '_' character.

    Once an instance has been created, it can be called with a list of names
    and a list of valid names will be created.
    The `__call__` method accepts an optional keyword, `default`, that sets
    the default name in case of ambiguity. By default, `default = 'f'`, so
    that names will default to `f0`, `f1`

    Parameters
    ----------
    excludelist : sequence, optional
        A list of names to exclude. This list is appended to the default list
        ['return','file','print']. Excluded names are appended an underscore:
        for example, `file` would become `file_`.
    deletechars : string, optional
        A string combining invalid characters that must be deleted from the names.
    casesensitive : {True, False, 'upper', 'lower'}, optional
        If True, field names are case_sensitive.
        If False or 'upper', field names are converted to upper case.
        If 'lower', field names are converted to lower case.
    """
    #
    defaultexcludelist = ['return','file','print']
    defaultdeletechars = set("""~!@#$%^&*()-=+~\|]}[{';: /?.>,<""")
    #
    def __init__(self, excludelist=None, deletechars=None, case_sensitive=None):
        #
        if excludelist is None:
            excludelist = []
        excludelist.extend(self.defaultexcludelist)
        self.excludelist = excludelist
        #
        if deletechars is None:
            delete = self.defaultdeletechars
        else:
            delete = set(deletechars)
        delete.add('"')
        self.deletechars = delete
        
        if (case_sensitive is None) or (case_sensitive is True):
            self.case_converter = lambda x: x
        elif (case_sensitive is False) or ('u' in case_sensitive):
            self.case_converter = lambda x: x.upper()
        elif 'l' in case_sensitive:
            self.case_converter = lambda x: x.lower()
        else:
            self.case_converter = lambda x: x
    #
    def validate(self, names, default='f'):
        #
        if names is None:
            return
        #
        validatednames = []
        seen = dict()
        #
        deletechars = self.deletechars
        excludelist = self.excludelist
        #
        case_converter = self.case_converter
        #
        for i, item in enumerate(names):
            item = case_converter(item)
            item = item.strip().replace(' ', '_')
            item = ''.join([c for c in item if c not in deletechars])
            if not len(item):
                item = '%s%d' % (default, i)
            elif item in excludelist:
                item += '_'
            cnt = seen.get(item, 0)
            if cnt > 0:
                validatednames.append(item + '_%d' % cnt)
            else:
                validatednames.append(item)
            seen[item] = cnt+1
        return validatednames
    #
    def __call__(self, names, default='f'):
        return self.validate(names, default)



def str2bool(value):
    """
    Tries to transform a string supposed to represent a boolean to a boolean.
    
    Raises
    ------
    ValueError
        If the string is not 'True' or 'False' (case independent)
    """
    value = value.upper()
    if value == 'TRUE':
        return True
    elif value == 'FALSE':
        return False
    else:
        raise ValueError("Invalid boolean")



class StringConverter:
    """
    Factory class for function transforming a string into another object (int,
    float).

    After initialization, an instance can be called to transform a string 
    into another object. If the string is recognized as representing a missing
    value, a default value is returned.

    Parameters
    ----------
    dtype_or_func : {None, dtype, function}, optional
        Input data type, used to define a basic function and a default value
        for missing data. For example, when `dtype` is float, the :attr:`func`
        attribute is set to ``float`` and the default value to `np.nan`.
        Alternatively, function used to convert a string to another object.
        In that later case, it is recommended to give an associated default
        value as input.
    default : {None, var}, optional
        Value to return by default, that is, when the string to be converted
        is flagged as missing.
    missing_values : {sequence}, optional
        Sequence of strings indicating a missing value.
    locked : {boolean}, optional
        Whether the StringConverter should be locked to prevent automatic 
        upgrade or not.

    Attributes
    ----------
    func : function
        Function used for the conversion
    default : var
        Default value to return when the input corresponds to a missing value.
    type : type
        Type of the output.
    _status : integer
        Integer representing the order of the conversion.
    _mapper : sequence of tuples
        Sequence of tuples (dtype, function, default value) to evaluate in order.
    _locked : boolean
        Whether the StringConverter is locked, thereby preventing automatic any
        upgrade or not.

    """
    #
    _mapper = [(nx.bool_, str2bool, False),
               (nx.integer, int, -1),
               (nx.floating, float, nx.nan),
               (complex, complex, nx.nan+0j),
               (nx.string_, str, '???')]
    (_defaulttype, _defaultfunc, _defaultfill) = zip(*_mapper)
    #
    @classmethod
    def _getsubdtype(cls, val):
        """Returns the type of the dtype of the input variable."""
        return np.array(val).dtype.type
    #
    @classmethod
    def upgrade_mapper(cls, func, default=None):
        """
    Upgrade the mapper of a StringConverter by adding a new function and its
    corresponding default.
    
    The input function (or sequence of functions) and its associated default 
    value (if any) is inserted in penultimate position of the mapper.
    The corresponding type is estimated from the dtype of the default value.
    
    Parameters
    ----------
    func : var
        Function, or sequence of functions

    Examples
    --------
    >>> import dateutil.parser
    >>> import datetime
    >>> dateparser = datetutil.parser.parse
    >>> defaultdate = datetime.date(2000, 1, 1)
    >>> StringConverter.upgrade_mapper(dateparser, default=defaultdate)
        """
        # Func is a single functions
        if hasattr(func, '__call__'):
            cls._mapper.insert(-1, (cls._getsubdtype(default), func, default))
            return
        elif hasattr(func, '__iter__'):
            if isinstance(func[0], (tuple, list)):
                for _ in func:
                    cls._mapper.insert(-1, _)
                return
            if default is None:
                default = [None] * len(func)
            else:
                default = list(default)
                default.append([None] * (len(func)-len(default)))
            for (fct, dft) in zip(func, default):
                cls._mapper.insert(-1, (cls._getsubdtype(dft), fct, dft))
    #
    def __init__(self, dtype_or_func=None, default=None, missing_values=None,
                 locked=False):
        # Defines a lock for upgrade
        self._locked = bool(locked)
        # No input dtype: minimal initialization
        if dtype_or_func is None:
            self.func = str2bool
            self._status = 0
            self.default = default or False
            ttype = np.bool
        else:
            # Is the input a np.dtype ?
            try:
                self.func = None
                ttype = np.dtype(dtype_or_func).type
            except TypeError:
                # dtype_or_func must be a function, then
                if not hasattr(dtype_or_func, '__call__'):
                    errmsg = "The input argument `dtype` is neither a function"\
                             " or a dtype (got '%s' instead)"
                    raise TypeError(errmsg % type(dtype_or_func))
                # Set the function
                self.func = dtype_or_func
                # If we don't have a default, try to guess it or set it to None
                if default is None:
                    try:
                        default = self.func('0')
                    except ValueError:
                        default = None
                ttype = self._getsubdtype(default)
            # Set the status according to the dtype
            _status = -1
            for (i, (deftype, func, default_def)) in enumerate(self._mapper):
                if np.issubdtype(ttype, deftype):
                    _status = i
                    self.default = default or default_def
                    break
            if _status == -1:
                # We never found a match in the _mapper...
                _status = 0
                self.default = default
            self._status = _status
            # If the input was a dtype, set the function to the last we saw
            if self.func is None:
                self.func = func
            # If the status is 1 (int), change the function to smthg more robust
            if self.func == self._mapper[1][1]:
                self.func = lambda x : int(float(x))
        # Store the list of strings corresponding to missing values.
        if missing_values is None:
            self.missing_values = set([''])
        else:
            self.missing_values = set(list(missing_values) + [''])
        #
        self._callingfunction = self._strict_call
        self.type = ttype
        self._checked = False
    #
    def _loose_call(self, value):
        try:
            return self.func(value)
        except ValueError:
            return self.default
    #
    def _strict_call(self, value):
        try:
            return self.func(value)
        except ValueError:
            if value.strip() in self.missing_values:
                if not self._status:
                    self._checked = False
                return self.default
            raise ValueError("Cannot convert string '%s'" % value)
    #
    def __call__(self, value):
        return self._callingfunction(value)
    #
    def upgrade(self, value):
        """
    Tries to find the best converter for `value`, by testing different
    converters in order.
    The order in which the converters are tested is read from the
    :attr:`_status` attribute of the instance.
        """
        self._checked = True
        try:
            self._strict_call(value)
        except ValueError:
            # Raise an exception if we locked the converter...
            if self._locked:
                raise ValueError("Converter is locked and cannot be upgraded")
            _statusmax = len(self._mapper)
            # Complains if we try to upgrade by the maximum
            if self._status == _statusmax:
                raise ValueError("Could not find a valid conversion function")
            elif self._status < _statusmax - 1:
                self._status += 1
            (self.type, self.func, self.default) = self._mapper[self._status]
            self.upgrade(value)
    #
    def update(self, func, default=None, missing_values='', locked=False):
        """
    Sets the :attr:`func` and :attr:`default` attributes directly.

    Parameters
    ----------
    func : function
        Conversion function.
    default : {var}, optional
        Default value to return when a missing value is encountered.
    missing_values : {var}, optional
        Sequence of strings representing missing values.
    locked : {False, True}, optional
        Whether the status should be locked to prevent automatic upgrade.
        """
        self.func = func
        self._locked = locked
        # Don't reset the default to None if we can avoid it
        if default is not None:
            self.default = default
        # Add the missing values to the existing set
        if missing_values is not None:
            if _is_string_like(missing_values):
                self.missing_values.add(missing_values)
            elif hasattr(missing_values, '__iter__'):
                for val in missing_values:
                    self.missing_values.add(val)
        else:
            self.missing_values = []
        # Update the type
        try:
            tester = func('0')
        except ValueError:
            tester = None
        self.type = self._getsubdtype(tester)

