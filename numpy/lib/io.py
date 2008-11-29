__all__ = ['savetxt', 'loadtxt',
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

_file = file

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

def load(file, mmap_mode=None):
    """
    Load a pickled, ``.npy``, or ``.npz`` binary file.

    Parameters
    ----------
    file : file-like object or string
        The file to read.  It must support ``seek()`` and ``read()`` methods.
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
    if isinstance(file, basestring):
        fid = _file(file,"rb")
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
    Save an array to a binary file in NumPy format.

    Parameters
    ----------
    f : file or string
        File or filename to which the data is saved.  If the filename
        does not already have a ``.npy`` extension, it is added.
    x : array_like
        Array data.

    Examples
    --------
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()

    >>> x = np.arange(10)
    >>> np.save(outfile, x)

    >>> outfile.seek(0)
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
    Save several arrays into an .npz file format which is a zipped-archive
    of arrays

    If keyword arguments are given, then filenames are taken from the keywords.
    If arguments are passed in with no keywords, then stored file names are
    arr_0, arr_1, etc.

    Parameters
    ----------
    file : string
        File name of .npz file.
    args : Arguments
        Function arguments.
    kwds : Keyword arguments
        Keywords.

    """

    # Import is postponed to here since zipfile depends on gzip, an optional
    # component of the so-called standard library.
    import zipfile

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

    # Place to write temporary .npy files
    #  before storing them in the zip
    import tempfile
    direc = tempfile.gettempdir()
    todel = []

    for key, val in namedict.iteritems():
        fname = key + '.npy'
        filename = os.path.join(direc, fname)
        todel.append(filename)
        fid = open(filename,'wb')
        format.write_array(fid, np.asanyarray(val))
        fid.close()
        zip.write(filename, arcname=fname)

    zip.close()
    for name in todel:
        os.remove(name)

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


def _string_like(obj):
    try: obj + ''
    except (TypeError, ValueError): return 0
    return 1

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

    if _string_like(fname):
        if fname.endswith('.gz'):
            import gzip
            fh = gzip.open(fname)
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
            return [dt]
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

    if len(dtype_types) > 1:
        # We're dealing with a structured array, with a dtype such as
        # [('x', int), ('y', [('s', int), ('t', float)])]
        #
        # First, create the array using a flattened dtype:
        # [('x', int), ('s', int), ('t', float)]
        #
        # Then, view the array using the specified dtype.
        X = np.array(X, dtype=np.dtype([('', t) for t in dtype_types]))
        X = X.view(dtype)
    else:
        X = np.array(X, dtype)

    X = np.squeeze(X)
    if unpack:
        return X.T
    else:
        return X


def savetxt(fname, X, fmt='%.18e',delimiter=' '):
    """
    Save an array to file.

    Parameters
    ----------
    fname : filename or a file handle
        If the filename ends in .gz, the file is automatically saved in
        compressed gzip format.  The load() command understands gzipped
        files transparently.
    X : array_like
        Data.
    fmt : string or sequence of strings
        A single format (%10.5f), a sequence of formats, or a
        multi-format string, e.g. 'Iteration %d -- %10.5f', in which
        case delimiter is ignored.
    delimiter : str
        Character separating columns.

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

    This is not an exhaustive specification.



    Examples
    --------
    >>> savetxt('test.out', x, delimiter=',') # X is an array
    >>> savetxt('test.out', (x,y,z)) # x,y,z equal sized 1D arrays
    >>> savetxt('test.out', x, fmt='%1.4e') # use exponential notation

    """

    if _string_like(fname):
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
        # make sure np.array doesn't interpret strings as binary data
        # by always producing a list of tuples
        seq = [(x,) for x in seq]
    output = np.array(seq, dtype=dtype)
    return output
