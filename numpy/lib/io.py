__all__ = ['savetxt', 'loadtxt',
           'load', 'loads',
           'save', 'savez',
           'packbits', 'unpackbits',
           'fromregex',
           'DataSource']

import numpy as np
import format
import cStringIO
import tempfile
import os

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

def load(file, memmap=False):
    """Load a binary file.

    Read a binary file (either a pickle, or a binary .npy/.npz file) and
    return the result.

    Parameters
    ----------
    file : file-like object or string
        the file to read.  It must support seek and read methods
    memmap : bool
        If true, then memory-map the .npy file or unzip the .npz file into
           a temporary directory and memory-map each component
        This has no effect for a pickle.

    Returns
    -------
    result : array, tuple, dict, etc.
        data stored in the file.
        If file contains pickle data, then whatever is stored in the pickle is
          returned.
        If the file is .npy file, then an array is returned.
        If the file is .npz file, then a dictionary-like object is returned
          which has a filename:array key:value pair for every file in the zip.

    Raises
    ------
    IOError
    """
    if isinstance(file, type("")):
        fid = _file(file,"rb")
    else:
        fid = file

    if memmap:
        raise NotImplementedError

    # Code to distinguish from NumPy binary files and pickles.
    _ZIP_PREFIX = 'PK\x03\x04'
    N = len(format.MAGIC_PREFIX)
    magic = fid.read(N)
    fid.seek(-N,1) # back-up
    if magic.startswith(_ZIP_PREFIX):  # zip-file (assume .npz)
        return NpzFile(fid)
    elif magic == format.MAGIC_PREFIX: # .npy file
        return format.read_array(fid)
    else:  # Try a pickle
        try:
            return _cload(fid)
        except:
            raise IOError, \
                "Failed to interpret file %s as a pickle" % repr(file)

def save(file, arr):
    """Save an array to a binary file (a string or file-like object).

    If the file is a string, then if it does not have the .npy extension,
        it is appended and a file open.

    Data is saved to the open file in NumPy-array format

    Examples
    --------
    import numpy as np
    ...
    np.save('myfile', a)
    a = np.load('myfile.npy')
    """
    if isinstance(file, str):
        if not file.endswith('.npy'):
            file = file + '.npy'
        fid = open(file, "wb")
    else:
        fid = file

    arr = np.asanyarray(arr)
    format.write_array(fid, arr)

def savez(file, *args, **kwds):
    """Save several arrays into an .npz file format which is a zipped-archive
    of arrays

    If keyword arguments are given, then filenames are taken from the keywords.
    If arguments are passed in with no keywords, then stored file names are
    arr_0, arr_1, etc.
    """

    # Import is postponed to here since zipfile depends on gzip, an optional
    # component of the so-called standard library.
    import zipfile

    if isinstance(file, str):
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
    Load ASCII data from fname into an array and return the array.

    The data must be regular, same number of values in every row

    Parameters
    ----------
    fname : filename or a file handle.
      Support for gzipped files is automatic, if the filename ends in .gz

    dtype : data-type
      Data type of the resulting array.  If this is a record data-type, the
      resulting array will be 1-d and each row will be interpreted as an
      element of the array. The number of columns used must match the number
      of fields in the data-type in this case.

    comments : str
      The character used to indicate the start of a comment in the file.

    delimiter : str
      A string-like character used to separate values in the file. If delimiter
      is unspecified or none, any whitespace string is a separator.

    converters : {}
      A dictionary mapping column number to a function that will convert that
      column to a float.  Eg, if column 0 is a date string:
      converters={0:datestr2num}. Converters can also be used to provide
      a default value for missing data: converters={3:lambda s: float(s or 0)}.

    skiprows : int
      The number of rows from the top to skip.

    usecols : sequence
      A sequence of integer column indexes to extract where 0 is the first
      column, eg. usecols=(1,4,5) will extract the 2nd, 5th and 6th columns.

    unpack : bool
      If True, will transpose the matrix allowing you to unpack into named
      arguments on the left hand side.

    Examples
    --------
      >>> X = loadtxt('test.dat')  # data in two columns
      >>> x,y,z = load('somefile.dat', usecols=(3,5,7), unpack=True)
      >>> r = np.loadtxt('record.dat', dtype={'names':('gender','age','weight'),
                'formats': ('S1','i4', 'f4')})

    SeeAlso: scipy.io.loadmat to read and write matfiles.
    """

    if _string_like(fname):
        if fname.endswith('.gz'):
            import gzip
            fh = gzip.open(fname)
        else:
            fh = file(fname)
    elif hasattr(fname, 'seek'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')
    X = []

    dtype = np.dtype(dtype)
    defconv = _getconv(dtype)
    converterseq = None
    if converters is None:
        converters = {}
        if dtype.names is not None:
            if usecols is None:
                converterseq = [_getconv(dtype.fields[name][0]) \
                                for name in dtype.names]
            else:
                converters.update([(col,_getconv(dtype.fields[name][0])) \
                                    for col,name in zip(usecols, dtype.names)])

    for i,line in enumerate(fh):
        if i<skiprows: continue
        comment_start = line.find(comments)
        if comment_start != -1:
            line = line[:comment_start].strip()
        else:
            line = line.strip()
        if not len(line): continue
        vals = line.split(delimiter)
        if converterseq is None:
            converterseq = [converters.get(j,defconv) \
                            for j in xrange(len(vals))]
        if usecols is not None:
            row = [converterseq[j](vals[j]) for j in usecols]
        else:
            row = [converterseq[j](val) for j,val in enumerate(vals)]
        if dtype.names is not None:
            row = tuple(row)
        X.append(row)

    X = np.array(X, dtype)
    X = np.squeeze(X)
    if unpack: return X.T
    else:  return X



def savetxt(fname, X, fmt='%.18e',delimiter=' '):
    """
    Save the data in X to file fname using fmt string to convert the
    data to strings

    Parameters
    ----------
    fname : filename or a file handle
        If the filename ends in .gz, the file is automatically saved in
        compressed gzip format.  The load() command understands gzipped
        files transparently.
    X : array or sequence
        Data to write to file.
    fmt : string or sequence of strings
        A single format (%10.5f), a sequence of formats, or a
        multi-format string, e.g. 'Iteration %d -- %10.5f', in which
        case delimiter is ignored.
    delimiter : str
        Character separating columns.

    Examples
    --------
    >>> np.savetxt('test.out', x, delimiter=',') # X is an array
    >>> np.savetxt('test.out', (x,y,z)) # x,y,z equal sized 1D arrays
    >>> np.savetxt('test.out', x, fmt='%1.4e') # use exponential notation

    Notes on fmt
    ------------
    flags:
        - : left justify
        + : Forces to preceed result with + or -.
        0 : Left pad the number with zeros instead of space (see width).

    width:
        Minimum number of characters to be printed. The value is not truncated.

    precision:
        - For integer specifiers (eg. d,i,o,x), the minimum number of
          digits.
        - For e, E and f specifiers, the number of digits to print
          after the decimal point.
        - For g and G, the maximum number of significant digits.
        - For s, the maximum number of charac ters.

    specifiers:
        c : character
        d or i : signed decimal integer
        e or E : scientific notation with e or E.
        f : decimal floating point
        g,G : use the shorter of e,E or f
        o : signed octal
        s : string of characters
        u : unsigned decimal integer
        x,X : unsigned hexadecimal integer

    This is not an exhaustive specification.

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
    Construct an array from a text file, using regular-expressions
    parsing.

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
