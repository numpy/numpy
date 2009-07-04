__all__ = ['savetxt', 'loadtxt',
           'genfromtxt', 'ndfromtxt', 'mafromtxt', 'recfromtxt', 'recfromcsv',
           'load', 'loads',
           'save', 'savez',
           'packbits', 'unpackbits',
           'fromregex',
           'DataSource']

import numpy as np
import format
import cStringIO
import os
import itertools

from cPickle import load as _cload, loads
from _datasource import DataSource
from _compiled_base import packbits, unpackbits

from _iotools import LineSplitter, NameValidator, StringConverter, \
                     _is_string_like, has_nested_fields, flatten_dtype

_file = file
_string_like = _is_string_like

def seek_gzip_factory(f):
    """Use this factory to produce the class so that we can do a lazy
    import on gzip.

    """
    import gzip, new

    def seek(self, offset, whence=0):
        # figure out new position (we can only seek forwards)
        if whence == 1:
            offset = self.offset + offset

        if whence not in [0, 1]:
            raise IOError, "Illegal argument"

        if offset < self.offset:
            # for negative seek, rewind and do positive seek
            self.rewind()
            count = offset - self.offset
            for i in range(count // 1024):
                self.read(1024)
            self.read(count % 1024)

    def tell(self):
        return self.offset

    if isinstance(f, str):
        f = gzip.GzipFile(f)

    f.seek = new.instancemethod(seek, f)
    f.tell = new.instancemethod(tell, f)

    return f

class BagObj(object):
    """A simple class that converts attribute lookups to
    getitems on the class passed in.
    """
    def __init__(self, obj):
        self._obj = obj
    def __getattribute__(self, key):
        try:
            return object.__getattribute__(self, '_obj')[key]
        except KeyError:
            raise AttributeError, key

class NpzFile(object):
    """A dictionary-like object with lazy-loading of files in the zipped
    archive provided on construction.

    The arrays and file strings are lazily loaded on either
    getitem access using obj['key'] or attribute lookup using obj.f.key

    A list of all files (without .npy) extensions can be obtained
    with .files and the ZipFile object itself using .zip
    """
    def __init__(self, fid):
        # Import is postponed to here since zipfile depends on gzip, an optional
        # component of the so-called standard library.
        import zipfile
        _zip = zipfile.ZipFile(fid)
        self._files = _zip.namelist()
        self.files = []
        for x in self._files:
            if x.endswith('.npy'):
                self.files.append(x[:-4])
            else:
                self.files.append(x)
        self.zip = _zip
        self.f = BagObj(self)

    def __getitem__(self, key):
        # FIXME: This seems like it will copy strings around
        #   more than is strictly necessary.  The zipfile
        #   will read the string and then
        #   the format.read_array will copy the string
        #   to another place in memory.
        #   It would be better if the zipfile could read
        #   (or at least uncompress) the data
        #   directly into the array memory.
        member = 0
        if key in self._files:
            member = 1
        elif key in self.files:
            member = 1
            key += '.npy'
        if member:
            bytes = self.zip.read(key)
            if bytes.startswith(format.MAGIC_PREFIX):
                value = cStringIO.StringIO(bytes)
                return format.read_array(value)
            else:
                return bytes
        else:
            raise KeyError, "%s is not a file in the archive" % key


    def __iter__(self):
        return iter(self.files)

    def items(self):
        return [(f, self[f]) for f in self.files]

    def iteritems(self):
        for f in self.files:
            yield (f, self[f])

    def keys(self):
        return self.files

    def iterkeys(self):
        return self.__iter__()

    def __contains__(self, key):
        return self.files.__contains__(key)


def load(file, mmap_mode=None):
    """
    Load a pickled, ``.npy``, or ``.npz`` binary file.

    Parameters
    ----------
    file : file-like object or string
        The file to read.  It must support ``seek()`` and ``read()`` methods.
        If the filename extension is ``.gz``, the file is first decompressed.
    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, then memory-map the file, using the given mode
        (see `numpy.memmap`).  The mode has no effect for pickled or
        zipped files.
        A memory-mapped array is stored on disk, and not directly loaded
        into memory.  However, it can be accessed and sliced like any
        ndarray.  Memory mapping is especially useful for accessing
        small fragments of large files without reading the entire file
        into memory.

    Returns
    -------
    result : array, tuple, dict, etc.
        Data stored in the file.

    Raises
    ------
    IOError
        If the input file does not exist or cannot be read.

    See Also
    --------
    save, savez, loadtxt
    memmap : Create a memory-map to an array stored in a file on disk.

    Notes
    -----
    - If the file contains pickle data, then whatever is stored in the
      pickle is returned.
    - If the file is a ``.npy`` file, then an array is returned.
    - If the file is a ``.npz`` file, then a dictionary-like object is
      returned, containing ``{filename: array}`` key-value pairs, one for
      each file in the archive.

    Examples
    --------
    Store data to disk, and load it again:

    >>> np.save('/tmp/123', np.array([[1, 2, 3], [4, 5, 6]]))
    >>> np.load('/tmp/123.npy')
    array([[1, 2, 3],
           [4, 5, 6]])

    Mem-map the stored array, and then access the second row
    directly from disk:

    >>> X = np.load('/tmp/123.npy', mmap_mode='r')
    >>> X[1, :]
    memmap([4, 5, 6])

    """
    import gzip

    if isinstance(file, basestring):
        fid = _file(file,"rb")
    elif isinstance(file, gzip.GzipFile):
        fid = seek_gzip_factory(file)
    else:
        fid = file

    # Code to distinguish from NumPy binary files and pickles.
    _ZIP_PREFIX = 'PK\x03\x04'
    N = len(format.MAGIC_PREFIX)
    magic = fid.read(N)
    fid.seek(-N,1) # back-up
    if magic.startswith(_ZIP_PREFIX):  # zip-file (assume .npz)
        return NpzFile(fid)
    elif magic == format.MAGIC_PREFIX: # .npy file
        if mmap_mode:
            return format.open_memmap(file, mode=mmap_mode)
        else:
            return format.read_array(fid)
    else:  # Try a pickle
        try:
            return _cload(fid)
        except:
            raise IOError, \
                "Failed to interpret file %s as a pickle" % repr(file)

def save(file, arr):
    """
    Save an array to a binary file in NumPy ``.npy`` format.

    Parameters
    ----------
    file : file or string
        File or filename to which the data is saved.  If the filename
        does not already have a ``.npy`` extension, it is added.
    arr : array_like
        Array data to be saved.

    See Also
    --------
    savez : Save several arrays into a .npz compressed archive
    savetxt, load

    Examples
    --------
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()

    >>> x = np.arange(10)
    >>> np.save(outfile, x)

    >>> outfile.seek(0) # only necessary in this example (with tempfile)
    >>> np.load(outfile)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    """
    if isinstance(file, basestring):
        if not file.endswith('.npy'):
            file = file + '.npy'
        fid = open(file, "wb")
    else:
        fid = file

    arr = np.asanyarray(arr)
    format.write_array(fid, arr)

def savez(file, *args, **kwds):
    """
    Save several arrays into a single, compressed file with extension ".npz"

    If keyword arguments are given, the names for variables assigned to the
    keywords are the keyword names (not the variable names in the caller).
    If arguments are passed in with no keywords, the corresponding variable
    names are arr_0, arr_1, etc.

    Parameters
    ----------
    file : Either the filename (string) or an open file (file-like object)
        If file is a string, it names the output file.  ".npz" will be appended
        if it is not already there.
    args : Arguments
        Any function arguments other than the file name are variables to save.
        Since it is not possible for Python to know their names outside the
        savez function, they will be saved with names "arr_0", "arr_1", and so
        on.  These arguments can be any expression.
    kwds : Keyword arguments
        All keyword=value pairs cause the value to be saved with the name of
        the keyword.

    See Also
    --------
    save : Save a single array to a binary file in NumPy format
    savetxt : Save an array to a file as plain text

    Notes
    -----
    The .npz file format is a zipped archive of files named after the variables
    they contain.  Each file contains one variable in .npy format.

    Examples
    --------
    >>> x = np.random.random((3, 3))
    >>> y = np.zeros((3, 2))
    >>> np.savez('data', x=x, y=y)

    """

    # Import is postponed to here since zipfile depends on gzip, an optional
    # component of the so-called standard library.
    import zipfile
    # Import deferred for startup time improvement
    import tempfile

    if isinstance(file, basestring):
        if not file.endswith('.npz'):
            file = file + '.npz'

    namedict = kwds
    for i, val in enumerate(args):
        key = 'arr_%d' % i
        if key in namedict.keys():
            raise ValueError, "Cannot use un-named variables and keyword %s" % key
        namedict[key] = val

    zip = zipfile.ZipFile(file, mode="w")

    # Stage arrays in a temporary file on disk, before writing to zip.
    fd, tmpfile = tempfile.mkstemp(suffix='-numpy.npy')
    os.close(fd)
    try:
        for key, val in namedict.iteritems():
            fname = key + '.npy'
            fid = open(tmpfile, 'wb')
            try:
                format.write_array(fid, np.asanyarray(val))
                fid.close()
                fid = None
                zip.write(tmpfile, arcname=fname)
            finally:
                if fid:
                    fid.close()
    finally:
        os.remove(tmpfile)

    zip.close()

# Adapted from matplotlib

def _getconv(dtype):
    typ = dtype.type
    if issubclass(typ, np.bool_):
        return lambda x: bool(int(x))
    if issubclass(typ, np.integer):
        return lambda x: int(float(x))
    elif issubclass(typ, np.floating):
        return float
    elif issubclass(typ, np.complex):
        return complex
    else:
        return str



def loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None,
            skiprows=0, usecols=None, unpack=False):
    """
    Load data from a text file.

    Each row in the text file must have the same number of values.

    Parameters
    ----------
    fname : file or string
        File or filename to read.  If the filename extension is ``.gz`` or
        ``.bz2``, the file is first decompressed.
    dtype : data-type
        Data type of the resulting array.  If this is a record data-type,
        the resulting array will be 1-dimensional, and each row will be
        interpreted as an element of the array.   In this case, the number
        of columns used must match the number of fields in the data-type.
    comments : string, optional
        The character used to indicate the start of a comment.
    delimiter : string, optional
        The string used to separate values.  By default, this is any
        whitespace.
    converters : {}
        A dictionary mapping column number to a function that will convert
        that column to a float.  E.g., if column 0 is a date string:
        ``converters = {0: datestr2num}``. Converters can also be used to
        provide a default value for missing data:
        ``converters = {3: lambda s: float(s or 0)}``.
    skiprows : int
        Skip the first `skiprows` lines.
    usecols : sequence
        Which columns to read, with 0 being the first.  For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
    unpack : bool
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = loadtxt(...)``

    Returns
    -------
    out : ndarray
        Data read from the text file.

    See Also
    --------
    scipy.io.loadmat : reads Matlab(R) data files

    Examples
    --------
    >>> from StringIO import StringIO   # StringIO behaves like a file object
    >>> c = StringIO("0 1\\n2 3")
    >>> np.loadtxt(c)
    array([[ 0.,  1.],
           [ 2.,  3.]])

    >>> d = StringIO("M 21 72\\nF 35 58")
    >>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
    ...                      'formats': ('S1', 'i4', 'f4')})
    array([('M', 21, 72.0), ('F', 35, 58.0)],
          dtype=[('gender', '|S1'), ('age', '<i4'), ('weight', '<f4')])

    >>> c = StringIO("1,0,2\\n3,0,4")
    >>> x,y = np.loadtxt(c, delimiter=',', usecols=(0,2), unpack=True)
    >>> x
    array([ 1.,  3.])
    >>> y
    array([ 2.,  4.])

    """
    user_converters = converters

    if usecols is not None:
        usecols = list(usecols)

    isstring = False
    if _is_string_like(fname):
        isstring = True
        if fname.endswith('.gz'):
            import gzip
            fh = seek_gzip_factory(fname)
        elif fname.endswith('.bz2'):
            import bz2
            fh = bz2.BZ2File(fname)
        else:
            fh = file(fname)
    elif hasattr(fname, 'readline'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')
    X = []

    def flatten_dtype(dt):
        """Unpack a structured data-type."""
        if dt.names is None:
            # If the dtype is flattened, return.
            # If the dtype has a shape, the dtype occurs
            # in the list more than once.
            return [dt.base] * int(np.prod(dt.shape))
        else:
            types = []
            for field in dt.names:
                tp, bytes = dt.fields[field]
                flat_dt = flatten_dtype(tp)
                types.extend(flat_dt)
            return types

    def split_line(line):
        """Chop off comments, strip, and split at delimiter."""
        line = line.split(comments)[0].strip()
        if line:
            return line.split(delimiter)
        else:
            return []

    try:
        # Make sure we're dealing with a proper dtype
        dtype = np.dtype(dtype)
        defconv = _getconv(dtype)

        # Skip the first `skiprows` lines
        for i in xrange(skiprows):
            fh.readline()

        # Read until we find a line with some values, and use
        # it to estimate the number of columns, N.
        first_vals = None
        while not first_vals:
            first_line = fh.readline()
            if first_line == '': # EOF reached
                raise IOError('End-of-file reached before encountering data.')
            first_vals = split_line(first_line)
        N = len(usecols or first_vals)

        dtype_types = flatten_dtype(dtype)
        if len(dtype_types) > 1:
            # We're dealing with a structured array, each field of
            # the dtype matches a column
            converters = [_getconv(dt) for dt in dtype_types]
        else:
            # All fields have the same dtype
            converters = [defconv for i in xrange(N)]

        # By preference, use the converters specified by the user
        for i, conv in (user_converters or {}).iteritems():
            if usecols:
                try:
                    i = usecols.index(i)
                except ValueError:
                    # Unused converter specified
                    continue
            converters[i] = conv

        # Parse each line, including the first
        for i, line in enumerate(itertools.chain([first_line], fh)):
            vals = split_line(line)
            if len(vals) == 0:
                continue

            if usecols:
                vals = [vals[i] for i in usecols]

            # Convert each value according to its column and store
            X.append(tuple([conv(val) for (conv, val) in zip(converters, vals)]))
    finally:
        if isstring:
            fh.close()

    if len(dtype_types) > 1:
        # We're dealing with a structured array, with a dtype such as
        # [('x', int), ('y', [('s', int), ('t', float)])]
        #
        # First, create the array using a flattened dtype:
        # [('x', int), ('s', int), ('t', float)]
        #
        # Then, view the array using the specified dtype.
        try:
            X = np.array(X, dtype=np.dtype([('', t) for t in dtype_types]))
            X = X.view(dtype)
        except TypeError:
            # In the case we have an object dtype
            X = np.array(X, dtype=dtype)
    else:
        X = np.array(X, dtype)

    X = np.squeeze(X)
    if unpack:
        return X.T
    else:
        return X


def savetxt(fname, X, fmt='%.18e',delimiter=' '):
    """
    Save an array to a text file.

    Parameters
    ----------
    fname : filename or file handle
        If the filename ends in ``.gz``, the file is automatically saved in
        compressed gzip format.  `loadtxt` understands gzipped files
        transparently.
    X : array_like
        Data to be saved to a text file.
    fmt : str or sequence of strs
        A single format (%10.5f), a sequence of formats, or a
        multi-format string, e.g. 'Iteration %d -- %10.5f', in which
        case `delimiter` is ignored.
    delimiter : str
        Character separating columns.

    See Also
    --------
    save : Save an array to a binary file in NumPy format
    savez : Save several arrays into an .npz compressed archive

    Notes
    -----
    Further explanation of the `fmt` parameter
    (``%[flag]width[.precision]specifier``):

    flags:
        ``-`` : left justify

        ``+`` : Forces to preceed result with + or -.

        ``0`` : Left pad the number with zeros instead of space (see width).

    width:
        Minimum number of characters to be printed. The value is not truncated
        if it has more characters.

    precision:
        - For integer specifiers (eg. ``d,i,o,x``), the minimum number of
          digits.
        - For ``e, E`` and ``f`` specifiers, the number of digits to print
          after the decimal point.
        - For ``g`` and ``G``, the maximum number of significant digits.
        - For ``s``, the maximum number of characters.

    specifiers:
        ``c`` : character

        ``d`` or ``i`` : signed decimal integer

        ``e`` or ``E`` : scientific notation with ``e`` or ``E``.

        ``f`` : decimal floating point

        ``g,G`` : use the shorter of ``e,E`` or ``f``

        ``o`` : signed octal

        ``s`` : string of characters

        ``u`` : unsigned decimal integer

        ``x,X`` : unsigned hexadecimal integer

    This explanation of ``fmt`` is not complete, for an exhaustive
    specification see [1]_.

    References
    ----------
    .. [1] `Format Specification Mini-Language
           <http://docs.python.org/library/string.html#
           format-specification-mini-language>`_, Python Documentation.

    Examples
    --------
    >>> savetxt('test.out', x, delimiter=',')   # X is an array
    >>> savetxt('test.out', (x,y,z))   # x,y,z equal sized 1D arrays
    >>> savetxt('test.out', x, fmt='%1.4e')   # use exponential notation

    """

    if _is_string_like(fname):
        if fname.endswith('.gz'):
            import gzip
            fh = gzip.open(fname,'wb')
        else:
            fh = file(fname,'w')
    elif hasattr(fname, 'seek'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')

    X = np.asarray(X)

    # Handle 1-dimensional arrays
    if X.ndim == 1:
        # Common case -- 1d array of numbers
        if X.dtype.names is None:
            X = np.atleast_2d(X).T
            ncol = 1

        # Complex dtype -- each field indicates a separate column
        else:
            ncol = len(X.dtype.descr)
    else:
        ncol = X.shape[1]

    # `fmt` can be a string with multiple insertion points or a list of formats.
    # E.g. '%10.5f\t%10d' or ('%10.5f', '$10d')
    if type(fmt) in (list, tuple):
        if len(fmt) != ncol:
            raise AttributeError('fmt has wrong shape.  %s' % str(fmt))
        format = delimiter.join(fmt)
    elif type(fmt) is str:
        if fmt.count('%') == 1:
            fmt = [fmt,]*ncol
            format = delimiter.join(fmt)
        elif fmt.count('%') != ncol:
            raise AttributeError('fmt has wrong number of %% formats.  %s'
                                 % fmt)
        else:
            format = fmt

    for row in X:
        fh.write(format % tuple(row) + '\n')

import re
def fromregex(file, regexp, dtype):
    """
    Construct an array from a text file, using regular-expressions parsing.

    Array is constructed from all matches of the regular expression
    in the file. Groups in the regular expression are converted to fields.

    Parameters
    ----------
    file : str or file
        File name or file object to read.
    regexp : str or regexp
        Regular expression used to parse the file.
        Groups in the regular expression correspond to fields in the dtype.
    dtype : dtype or dtype list
        Dtype for the structured array

    Examples
    --------
    >>> f = open('test.dat', 'w')
    >>> f.write("1312 foo\\n1534  bar\\n444   qux")
    >>> f.close()
    >>> np.fromregex('test.dat', r"(\\d+)\\s+(...)",
    ...              [('num', np.int64), ('key', 'S3')])
    array([(1312L, 'foo'), (1534L, 'bar'), (444L, 'qux')],
          dtype=[('num', '<i8'), ('key', '|S3')])

    """
    if not hasattr(file, "read"):
        file = open(file,'r')
    if not hasattr(regexp, 'match'):
        regexp = re.compile(regexp)
    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)

    seq = regexp.findall(file.read())
    if seq and not isinstance(seq[0], tuple):
        # Only one group is in the regexp.
        # Create the new array as a single data-type and then
        #   re-interpret as a single-field structured array. 
        newdtype = np.dtype(dtype[dtype.names[0]])
        output = np.array(seq, dtype=newdtype)
        output.dtype = dtype
    else:
        output = np.array(seq, dtype=dtype)

    return output




#####--------------------------------------------------------------------------
#---- --- ASCII functions ---
#####--------------------------------------------------------------------------



def genfromtxt(fname, dtype=float, comments='#', delimiter=None, skiprows=0,
               converters=None, missing='', missing_values=None, usecols=None,
               names=None, excludelist=None, deletechars=None,
               case_sensitive=True, unpack=None, usemask=False, loose=True):
    """
    Load data from a text file.

    Each line past the first `skiprows` ones is split at the `delimiter`
    character, and characters following the `comments` character are discarded.

    Parameters
    ----------
    fname : {file, string}
        File or filename to read.  If the filename extension is `.gz` or
        `.bz2`, the file is first decompressed.
    dtype : dtype
        Data type of the resulting array.  If this is a flexible data-type,
        the resulting array will be 1-dimensional, and each row will be
        interpreted as an element of the array. In this case, the number
        of columns used must match the number of fields in the data-type,
        and the names of each field will be set by the corresponding name
        of the dtype.
        If None, the dtypes will be determined by the contents of each
        column, individually.
    comments : string, optional
        The character used to indicate the start of a comment.
        All the characters occurring on a line after a comment are discarded
    delimiter : string, optional
        The string used to separate values.  By default, any consecutive
        whitespace act as delimiter.
    skiprows : int, optional
        Numbers of lines to skip at the beginning of the file.
    converters : {None, dictionary}, optional
        A dictionary mapping column number to a function that will convert
        values in the column to a number. Converters can also be used to
        provide a default value for missing data:
        ``converters = {3: lambda s: float(s or 0)}``.
    missing : string, optional
        A string representing a missing value, irrespective of the column where
        it appears (e.g., `'missing'` or `'unused'`).
    missing_values : {None, dictionary}, optional
        A dictionary mapping a column number to a string indicating whether the
        corresponding field should be masked.
    usecols : {None, sequence}, optional
        Which columns to read, with 0 being the first.  For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
    names : {None, True, string, sequence}, optional
        If `names` is True, the field names are read from the first valid line
        after the first `skiprows` lines.
        If `names` is a sequence or a single-string of comma-separated names,
        the names will be used to define the field names in a flexible dtype.
        If `names` is None, the names of the dtype fields will be used, if any.
    excludelist : sequence, optional
        A list of names to exclude. This list is appended to the default list
        ['return','file','print']. Excluded names are appended an underscore:
        for example, `file` would become `file_`.
    deletechars : string, optional
        A string combining invalid characters that must be deleted from the
        names.
    case_sensitive : {True, False, 'upper', 'lower'}, optional
        If True, field names are case_sensitive.
        If False or 'upper', field names are converted to upper case.
        If 'lower', field names are converted to lower case.
    unpack : bool, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = loadtxt(...)``
    usemask : bool, optional
        If True, returns a masked array.
        If False, return a regular standard array.

    Returns
    -------
    out : MaskedArray
        Data read from the text file.

    See Also
    --------
    numpy.loadtxt : equivalent function when no data is missing.

    Notes
    -----
    * When spaces are used as delimiters, or when no delimiter has been given
      as input, there should not be any missing data between two fields.
    * When the variable are named (either by a flexible dtype or with `names`,
      there must not be any header in the file (else a :exc:ValueError
      exception is raised).
    * Individual values are not stripped of spaces by default.
      When using a custom converter, make sure the function does remove spaces.

    """
    #
    if usemask:
        from numpy.ma import MaskedArray, make_mask_descr
    # Check the input dictionary of converters
    user_converters = converters or {}
    if not isinstance(user_converters, dict):
        errmsg = "The input argument 'converter' should be a valid dictionary "\
                 "(got '%s' instead)"
        raise TypeError(errmsg % type(user_converters))
    # Check the input dictionary of missing values
    user_missing_values = missing_values or {}
    if not isinstance(user_missing_values, dict):
        errmsg = "The input argument 'missing_values' should be a valid "\
                 "dictionary (got '%s' instead)"
        raise TypeError(errmsg % type(missing_values))
    defmissing = [_.strip() for _ in missing.split(',')] + ['']

    # Initialize the filehandle, the LineSplitter and the NameValidator
#    fhd = _to_filehandle(fname)
    if isinstance(fname, basestring):
        fhd = np.lib._datasource.open(fname)
    elif not hasattr(fname, 'read'):
        raise TypeError("The input should be a string or a filehandle. "\
                        "(got %s instead)" % type(fname))
    else:
        fhd = fname
    split_line = LineSplitter(delimiter=delimiter, comments=comments, 
                              autostrip=False)._handyman
    validate_names = NameValidator(excludelist=excludelist,
                                   deletechars=deletechars,
                                   case_sensitive=case_sensitive)

    # Get the first valid lines after the first skiprows ones
    for i in xrange(skiprows):
        fhd.readline()
    first_values = None
    while not first_values:
        first_line = fhd.readline()
        if first_line == '':
            raise IOError('End-of-file reached before encountering data.')
        if names is True:
            first_values = first_line.strip().split(delimiter)
        else:
            first_values = split_line(first_line)
    if names is True:
        fval = first_values[0].strip()
        if fval in comments:
            del first_values[0]

    # Check the columns to use
    if usecols is not None:
        usecols = list(usecols)
    nbcols = len(usecols or first_values)

    # Check the names and overwrite the dtype.names if needed
    if dtype is not None:
        dtype = np.dtype(dtype)
    dtypenames = getattr(dtype, 'names', None)
    if names is True:
        names = validate_names([_.strip() for _ in first_values])
        first_line =''
    elif _is_string_like(names):
        names = validate_names([_.strip() for _ in names.split(',')])
    elif names:
        names = validate_names(names)
    elif dtypenames:
        dtype.names = validate_names(dtypenames)
    if names and dtypenames:
        dtype.names = names

    # If usecols is a list of names, convert to a list of indices
    if usecols:
        for (i, current) in enumerate(usecols):
            if _is_string_like(current):
                usecols[i] = names.index(current)

    # If user_missing_values has names as keys, transform them to indices
    missing_values = {}
    for (key, val) in user_missing_values.iteritems():
        # If val is a list, flatten it. In any case, add missing &'' to the list
        if isinstance(val, (list, tuple)):
            val = [str(_) for _ in val]
        else:
            val = [str(val),]
        val.extend(defmissing)
        if _is_string_like(key):
            try:
                missing_values[names.index(key)] = val
            except ValueError:
                pass
        else:
            missing_values[key] = val


    # Initialize the default converters
    if dtype is None:
        # Note: we can't use a [...]*nbcols, as we would have 3 times the same
        # ... converter, instead of 3 different converters.
        converters = [StringConverter(None,
                              missing_values=missing_values.get(_, defmissing))
                      for _ in range(nbcols)]
    else:
        dtype_flat = flatten_dtype(dtype, flatten_base=True)
        # Initialize the converters
        if len(dtype_flat) > 1:
            # Flexible type : get a converter from each dtype
            converters = [StringConverter(dt,
                              missing_values=missing_values.get(i, defmissing),
                              locked=True)
                          for (i, dt) in enumerate(dtype_flat)]
        else:
            # Set to a default converter (but w/ different missing values)
            converters = [StringConverter(dtype,
                              missing_values=missing_values.get(_, defmissing),
                              locked=True)
                          for _ in range(nbcols)]
    missing_values = [_.missing_values for _ in converters]

    # Update the converters to use the user-defined ones
    uc_update = []
    for (i, conv) in user_converters.iteritems():
        # If the converter is specified by column names, use the index instead
        if _is_string_like(i):
            i = names.index(i)
        if usecols:
            try:
                i = usecols.index(i)
            except ValueError:
                # Unused converter specified
                continue
        converters[i].update(conv, default=None, 
                             missing_values=missing_values[i],
                             locked=True)
        uc_update.append((i, conv))
    # Make sure we have the corrected keys in user_converters...
    user_converters.update(uc_update)

    # Reset the names to match the usecols
    if (not first_line) and usecols:
        names = [names[_] for _ in usecols]

    rows = []
    append_to_rows = rows.append
    if usemask:
        masks = []
        append_to_masks = masks.append
    # Parse each line
    for line in itertools.chain([first_line,], fhd):
        values = split_line(line)
        # Skip an empty line
        if len(values) == 0:
            continue
        # Select only the columns we need
        if usecols:
            values = [values[_] for _ in usecols]
        # Check whether we need to update the converter
        if dtype is None:
            for (converter, item) in zip(converters, values):
                converter.upgrade(item)
        # Store the values
        append_to_rows(tuple(values))
        if usemask:
            append_to_masks(tuple([val.strip() in mss 
                                   for (val, mss) in zip(values,
                                                         missing_values)]))

    # Convert each value according to the converter:
    # We want to modify the list in place to avoid creating a new one...
    if loose:
        conversionfuncs = [conv._loose_call for conv in converters]
    else:
        conversionfuncs = [conv._strict_call for conv in converters]
    for (i, vals) in enumerate(rows):
        rows[i] = tuple([convert(val)
                         for (convert, val) in zip(conversionfuncs, vals)])

    # Reset the dtype
    data = rows
    if dtype is None:
        # Get the dtypes from the types of the converters
        coldtypes = [conv.type for conv in converters]
        # Find the columns with strings...
        strcolidx = [i for (i, v) in enumerate(coldtypes)
                     if v in (type('S'), np.string_)]
        # ... and take the largest number of chars.
        for i in strcolidx:
            coldtypes[i] = "|S%i" % max(len(row[i]) for row in data)
        #
        if names is None:
            # If the dtype is uniform, don't define names, else use ''
            base = set([c.type for c in converters if c._checked])
            
            if len(base) == 1:
                (ddtype, mdtype) = (list(base)[0], np.bool)
            else:
                ddtype = [('', dt) for dt in coldtypes]
                mdtype = [('', np.bool) for dt in coldtypes]
        else:
            ddtype = zip(names, coldtypes)
            mdtype = zip(names, [np.bool] * len(coldtypes))
        output = np.array(data, dtype=ddtype)
        if usemask:
            outputmask = np.array(masks, dtype=mdtype)
    else:
        # Overwrite the initial dtype names if needed
        if names and dtype.names:
            dtype.names = names
        # Case 1. We have a structured type
        if len(dtype_flat) > 1:
            # Nested dtype, eg  [('a', int), ('b', [('b0', int), ('b1', 'f4')])]
            # First, create the array using a flattened dtype:
            # [('a', int), ('b1', int), ('b2', float)]
            # Then, view the array using the specified dtype.
            if 'O' in (_.char for _ in dtype_flat):
                if has_nested_fields(dtype):
                    errmsg = "Nested fields involving objects "\
                             "are not supported..."
                    raise NotImplementedError(errmsg)
                else:
                    output = np.array(data, dtype=dtype)
            else:
                rows = np.array(data, dtype=[('', _) for _ in dtype_flat])
                output = rows.view(dtype)
            # Now, process the rowmasks the same way
            if usemask:
                rowmasks = np.array(masks,
                                    dtype=np.dtype([('', np.bool)
                                                    for t in dtype_flat]))
                # Construct the new dtype
                mdtype = make_mask_descr(dtype)
                outputmask = rowmasks.view(mdtype)
        # Case #2. We have a basic dtype
        else:
            # We used some user-defined converters
            if user_converters:
                ishomogeneous = True
                descr = []
                for (i, ttype) in enumerate([conv.type for conv in converters]):
                    # Keep the dtype of the current converter
                    if i in user_converters:
                        ishomogeneous &= (ttype == dtype.type)
                        if ttype == np.string_:
                            ttype = "|S%i" % max(len(row[i]) for row in data)
                        descr.append(('', ttype))
                    else:
                        descr.append(('', dtype))
                # So we changed the dtype ?
                if not ishomogeneous:
                    # We have more than one field
                    if len(descr) > 1:
                        dtype = np.dtype(descr)
                    # We have only one field: drop the name if not needed.
                    else:
                        dtype = np.dtype(ttype)
            #
            output = np.array(data, dtype)
            if usemask:
                if dtype.names:
                    mdtype = [(_, np.bool) for _ in dtype.names]
                else:
                    mdtype = np.bool
                outputmask = np.array(masks, dtype=mdtype)
    # Try to take care of the missing data we missed
    if usemask and output.dtype.names:
        for (name, conv) in zip(names or (), converters):
            missing_values = [conv(_) for _ in conv.missing_values if _ != '']
            for mval in missing_values:
                outputmask[name] |= (output[name] == mval)
    # Construct the final array
    if usemask:
        output = output.view(MaskedArray)
        output._mask = outputmask
    if unpack:
        return output.squeeze().T
    return output.squeeze()



def ndfromtxt(fname, dtype=float, comments='#', delimiter=None, skiprows=0,
             converters=None, missing='', missing_values=None,
             usecols=None, unpack=None, names=None,
             excludelist=None, deletechars=None, case_sensitive=True,):
    """
    Load ASCII data stored in fname and returns a ndarray.
    
    Complete description of all the optional input parameters is available in
    the docstring of the `genfromtxt` function.
    
    See Also
    --------
    numpy.genfromtxt : generic function.
    
    """
    kwargs = dict(dtype=dtype, comments=comments, delimiter=delimiter, 
                  skiprows=skiprows, converters=converters,
                  missing=missing, missing_values=missing_values,
                  usecols=usecols, unpack=unpack, names=names, 
                  excludelist=excludelist, deletechars=deletechars,
                  case_sensitive=case_sensitive, usemask=False)
    return genfromtxt(fname, **kwargs)

def mafromtxt(fname, dtype=float, comments='#', delimiter=None, skiprows=0,
              converters=None, missing='', missing_values=None,
              usecols=None, unpack=None, names=None,
              excludelist=None, deletechars=None, case_sensitive=True,):
    """
    Load ASCII data stored in fname and returns a MaskedArray.
    
    Complete description of all the optional input parameters is available in
    the docstring of the `genfromtxt` function.
    
    See Also
    --------
    numpy.genfromtxt : generic function.
    """
    kwargs = dict(dtype=dtype, comments=comments, delimiter=delimiter, 
                  skiprows=skiprows, converters=converters,
                  missing=missing, missing_values=missing_values,
                  usecols=usecols, unpack=unpack, names=names, 
                  excludelist=excludelist, deletechars=deletechars,
                  case_sensitive=case_sensitive,
                  usemask=True)
    return genfromtxt(fname, **kwargs)


def recfromtxt(fname, dtype=None, comments='#', delimiter=None, skiprows=0,
               converters=None, missing='', missing_values=None,
               usecols=None, unpack=None, names=None,
               excludelist=None, deletechars=None, case_sensitive=True,
               usemask=False):
    """
    Load ASCII data stored in fname and returns a standard recarray (if
    `usemask=False`) or a MaskedRecords (if `usemask=True`).

    Complete description of all the optional input parameters is available in
    the docstring of the `genfromtxt` function.

    See Also
    --------
    numpy.genfromtxt : generic function

    Notes
    -----
    * by default, `dtype=None`, which means that the dtype of the output array
      will be determined from the data.

    """
    kwargs = dict(dtype=dtype, comments=comments, delimiter=delimiter, 
                  skiprows=skiprows, converters=converters,
                  missing=missing, missing_values=missing_values,
                  usecols=usecols, unpack=unpack, names=names, 
                  excludelist=excludelist, deletechars=deletechars,
                  case_sensitive=case_sensitive, usemask=usemask)
    output = genfromtxt(fname, **kwargs)
    if usemask:
        from numpy.ma.mrecords import MaskedRecords
        output = output.view(MaskedRecords)
    else:
        output = output.view(np.recarray)
    return output


def recfromcsv(fname, dtype=None, comments='#', skiprows=0,
               converters=None, missing='', missing_values=None,
               usecols=None, unpack=None, names=True,
               excludelist=None, deletechars=None, case_sensitive='lower',
               usemask=False):
    """
    Load ASCII data stored in comma-separated file and returns a recarray (if 
    `usemask=False`) or a MaskedRecords (if `usemask=True`).
    
    Complete description of all the optional input parameters is available in
    the docstring of the `genfromtxt` function.
    
    See Also
    --------
    numpy.genfromtxt : generic function
    """
    kwargs = dict(dtype=dtype, comments=comments, delimiter=",", 
                  skiprows=skiprows, converters=converters,
                  missing=missing, missing_values=missing_values,
                  usecols=usecols, unpack=unpack, names=names, 
                  excludelist=excludelist, deletechars=deletechars,
                  case_sensitive=case_sensitive, usemask=usemask)
    output = genfromtxt(fname, **kwargs)
    if usemask:
        from numpy.ma.mrecords import MaskedRecords
        output = output.view(MaskedRecords)
    else:
        output = output.view(np.recarray)
    return output

