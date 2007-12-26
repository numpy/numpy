
__all__ = ['savetxt', 'loadtxt',
           'loads', 'load',
           'save', 'packbits', 'unpackbits',
           'DataSource',
          ]

import numpy as np

from cPickle import load as _cload, loads
from _datasource import DataSource
from _compiled_base import packbits, unpackbits

_file = file

def load(file):
    """Load a binary file.

    Read a binary file (either a pickle or a binary NumPy array file .npy) and
    return the resulting arrays. 

    Parameters:
    -----------
    file - the file to read. This can be a string, or any file-like object

    Returns:
    --------
    result - array or tuple of arrays stored in the file.  If file contains 
             pickle data, then whatever is stored in the pickle is returned.
    """
    if isinstance(file, type("")):
        file = _file(file,"rb")
    # Code to distinguish from pickle and NumPy binary

    # if pickle:
        return _cload(file)



class _bagobj(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

class _npz_obj(dict):
    pass

def save(file, arr):
    """Save an array to a binary file (specified as a string or file-like object).

    If the file is a string, then if it does not have the .npy extension, it is appended
        and a file open. 

    Data is saved to the open file in NumPy-array format

    Example:
    --------
    import numpy as np
    ...
    np.save('myfile', a)
    a = np.load('myfile.npy')
    """    
    # code to save to numpy binary here...

    

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
        line = line[:line.find(comments)].strip()
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




