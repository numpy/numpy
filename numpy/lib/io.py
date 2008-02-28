
__all__ = ['savetxt', 'loadtxt',
           'load', 'loads',
           'save', 'savez',
           'packbits', 'unpackbits',
           'DataSource']

import numpy as np
import format
import zipfile
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
        return int
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

    fname can be a filename or a file handle.  Support for gzipped files is
    automatic, if the filename ends in .gz

    See scipy.io.loadmat to read and write matfiles.

    Example usage:

      X = loadtxt('test.dat')  # data in two columns
      t = X[:,0]
      y = X[:,1]

    Alternatively, you can do the same with "unpack"; see below

      X = loadtxt('test.dat')    # a matrix of data
      x = loadtxt('test.dat')    # a single column of data


    dtype - the data-type of the resulting array.  If this is a
    record data-type, the the resulting array will be 1-d and each row will
    be interpreted as an element of the array. The number of columns
    used must match the number of fields in the data-type in this case.

    comments - the character used to indicate the start of a comment
    in the file

    delimiter is a string-like character used to seperate values in the
    file. If delimiter is unspecified or none, any whitespace string is
    a separator.

    converters, if not None, is a dictionary mapping column number to
    a function that will convert that column to a float.  Eg, if
    column 0 is a date string: converters={0:datestr2num}

    skiprows is the number of rows from the top to skip

    usecols, if not None, is a sequence of integer column indexes to
    extract where 0 is the first column, eg usecols=(1,4,5) to extract
    just the 2nd, 5th and 6th columns

    unpack, if True, will transpose the matrix allowing you to unpack
    into named arguments on the left hand side

        t,y = load('test.dat', unpack=True) # for  two column data
        x,y,z = load('somefile.dat', usecols=(3,5,7), unpack=True)

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
            converterseq = [_getconv(dtype.fields[name][0]) \
                            for name in dtype.names]

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
    r,c = X.shape
    if r==1 or c==1:
        X.shape = max([r,c]),
    if unpack: return X.T
    else:  return X


# adjust so that fmt can change across columns if desired.

def savetxt(fname, X, fmt='%.18e',delimiter=' '):
    """
    Save the data in X to file fname using fmt string to convert the
    data to strings

    fname can be a filename or a file handle.  If the filename ends in .gz,
    the file is automatically saved in compressed gzip format.  The load()
    command understands gzipped files transparently.

    Example usage:

    save('test.out', X)         # X is an array
    save('test1.out', (x,y,z))  # x,y,z equal sized 1D arrays
    save('test2.out', x)        # x is 1D
    save('test3.out', x, fmt='%1.4e')  # use exponential notation

    delimiter is used to separate the fields, eg delimiter ',' for
    comma-separated values
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
    origShape = None
    if len(X.shape)==1:
        origShape = X.shape
        X.shape = len(X), 1
    for row in X:
        fh.write(delimiter.join([fmt%val for val in row]) + '\n')

    if origShape is not None:
        X.shape = origShape
