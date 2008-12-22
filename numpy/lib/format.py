"""
Define a simple format for saving numpy arrays to disk with the full
information about them.

WARNING: Due to limitations in the interpretation of structured dtypes, dtypes
with fields with empty names will have the names replaced by 'f0', 'f1', etc.
Such arrays will not round-trip through the format entirely accurately. The
data is intact; only the field names will differ. We are working on a fix for
this.  This fix will not require a change in the file format. The arrays with
such structures can still be saved and restored, and the correct dtype may be
restored by using the `loadedarray.view(correct_dtype)` method.

Format Version 1.0
------------------

The first 6 bytes are a magic string: exactly "\\\\x93NUMPY".

The next 1 byte is an unsigned byte: the major version number of the file
format, e.g. \\\\x01.

The next 1 byte is an unsigned byte: the minor version number of the file
format, e.g. \\\\x00. Note: the version of the file format is not tied to the
version of the numpy package.

The next 2 bytes form a little-endian unsigned short int: the length of the
header data HEADER_LEN.

The next HEADER_LEN bytes form the header data describing the array's format.
It is an ASCII string which contains a Python literal expression of a
dictionary. It is terminated by a newline ('\\\\n') and padded with spaces
('\\\\x20') to make the total length of the magic string + 4 + HEADER_LEN be
evenly divisible by 16 for alignment purposes.

The dictionary contains three keys:

    "descr" : dtype.descr
        An object that can be passed as an argument to the numpy.dtype()
        constructor to create the array's dtype.
    "fortran_order" : bool
        Whether the array data is Fortran-contiguous or not. Since
        Fortran-contiguous arrays are a common form of non-C-contiguity, we
        allow them to be written directly to disk for efficiency.
    "shape" : tuple of int
        The shape of the array.

For repeatability and readability, the dictionary keys are sorted in alphabetic
order. This is for convenience only. A writer SHOULD implement this if
possible. A reader MUST NOT depend on this.

Following the header comes the array data. If the dtype contains Python objects
(i.e. dtype.hasobject is True), then the data is a Python pickle of the array.
Otherwise the data is the contiguous (either C- or Fortran-, depending on
fortran_order) bytes of the array. Consumers can figure out the number of bytes
by multiplying the number of elements given by the shape (noting that shape=()
means there is 1 element) by dtype.itemsize.

"""

import cPickle

import numpy
from numpy.lib.utils import safe_eval


MAGIC_PREFIX = '\x93NUMPY'
MAGIC_LEN = len(MAGIC_PREFIX) + 2

def magic(major, minor):
    """ Return the magic string for the given file format version.

    Parameters
    ----------
    major : int in [0, 255]
    minor : int in [0, 255]

    Returns
    -------
    magic : str

    Raises
    ------
    ValueError if the version cannot be formatted.
    """
    if major < 0 or major > 255:
        raise ValueError("major version must be 0 <= major < 256")
    if minor < 0 or minor > 255:
        raise ValueError("minor version must be 0 <= minor < 256")
    return '%s%s%s' % (MAGIC_PREFIX, chr(major), chr(minor))

def read_magic(fp):
    """ Read the magic string to get the version of the file format.

    Parameters
    ----------
    fp : filelike object

    Returns
    -------
    major : int
    minor : int
    """
    magic_str = fp.read(MAGIC_LEN)
    if len(magic_str) != MAGIC_LEN:
        msg = "could not read %d characters for the magic string; got %r"
        raise ValueError(msg % (MAGIC_LEN, magic_str))
    if magic_str[:-2] != MAGIC_PREFIX:
        msg = "the magic string is not correct; expected %r, got %r"
        raise ValueError(msg % (MAGIC_PREFIX, magic_str[:-2]))
    major, minor = map(ord, magic_str[-2:])
    return major, minor

def dtype_to_descr(dtype):
    """
    Get a serializable descriptor from the dtype.

    The .descr attribute of a dtype object cannot be round-tripped through
    the dtype() constructor. Simple types, like dtype('float32'), have
    a descr which looks like a record array with one field with '' as
    a name. The dtype() constructor interprets this as a request to give
    a default name.  Instead, we construct descriptor that can be passed to
    dtype().

    Parameters
    ----------
    dtype : dtype
        The dtype of the array that will be written to disk.

    Returns
    -------
    descr : object
        An object that can be passed to `numpy.dtype()` in order to
        replicate the input dtype.

    """
    if dtype.names is not None:
        # This is a record array. The .descr is fine.
        # XXX: parts of the record array with an empty name, like padding bytes,
        # still get fiddled with. This needs to be fixed in the C implementation
        # of dtype().
        return dtype.descr
    else:
        return dtype.str

def header_data_from_array_1_0(array):
    """ Get the dictionary of header metadata from a numpy.ndarray.

    Parameters
    ----------
    array : numpy.ndarray

    Returns
    -------
    d : dict
        This has the appropriate entries for writing its string representation
        to the header of the file.
    """
    d = {}
    d['shape'] = array.shape
    if array.flags.c_contiguous:
        d['fortran_order'] = False
    elif array.flags.f_contiguous:
        d['fortran_order'] = True
    else:
        # Totally non-contiguous data. We will have to make it C-contiguous
        # before writing. Note that we need to test for C_CONTIGUOUS first
        # because a 1-D array is both C_CONTIGUOUS and F_CONTIGUOUS.
        d['fortran_order'] = False

    d['descr'] = dtype_to_descr(array.dtype)
    return d

def write_array_header_1_0(fp, d):
    """ Write the header for an array using the 1.0 format.

    Parameters
    ----------
    fp : filelike object
    d : dict
        This has the appropriate entries for writing its string representation
        to the header of the file.
    """
    import struct
    header = ["{"]
    for key, value in sorted(d.items()):
        # Need to use repr here, since we eval these when reading
        header.append("'%s': %s, " % (key, repr(value)))
    header.append("}")
    header = "".join(header)
    # Pad the header with spaces and a final newline such that the magic
    # string, the header-length short and the header are aligned on a 16-byte
    # boundary.  Hopefully, some system, possibly memory-mapping, can take
    # advantage of our premature optimization.
    current_header_len = MAGIC_LEN + 2 + len(header) + 1  # 1 for the newline
    topad = 16 - (current_header_len % 16)
    header = '%s%s\n' % (header, ' '*topad)
    if len(header) >= (256*256):
        raise ValueError("header does not fit inside %s bytes" % (256*256))
    header_len_str = struct.pack('<H', len(header))
    fp.write(header_len_str)
    fp.write(header)

def read_array_header_1_0(fp):
    """
    Read an array header from a filelike object using the 1.0 file format
    version.

    This will leave the file object located just after the header.

    Parameters
    ----------
    fp : filelike object
        A file object or something with a `.read()` method like a file.

    Returns
    -------
    shape : tuple of int
        The shape of the array.
    fortran_order : bool
        The array data will be written out directly if it is either C-contiguous
        or Fortran-contiguous. Otherwise, it will be made contiguous before
        writing it out.
    dtype : dtype
        The dtype of the file's data.

    Raises
    ------
    ValueError :
        If the data is invalid.

    """
    # Read an unsigned, little-endian short int which has the length of the
    # header.
    import struct
    hlength_str = fp.read(2)
    if len(hlength_str) != 2:
        msg = "EOF at %s before reading array header length"
        raise ValueError(msg % fp.tell())
    header_length = struct.unpack('<H', hlength_str)[0]
    header = fp.read(header_length)
    if len(header) != header_length:
        raise ValueError("EOF at %s before reading array header" % fp.tell())

    # The header is a pretty-printed string representation of a literal Python
    # dictionary with trailing newlines padded to a 16-byte boundary. The keys
    # are strings.
    #   "shape" : tuple of int
    #   "fortran_order" : bool
    #   "descr" : dtype.descr
    try:
        d = safe_eval(header)
    except SyntaxError, e:
        msg = "Cannot parse header: %r\nException: %r"
        raise ValueError(msg % (header, e))
    if not isinstance(d, dict):
        msg = "Header is not a dictionary: %r"
        raise ValueError(msg % d)
    keys = d.keys()
    keys.sort()
    if keys != ['descr', 'fortran_order', 'shape']:
        msg = "Header does not contain the correct keys: %r"
        raise ValueError(msg % (keys,))

    # Sanity-check the values.
    if (not isinstance(d['shape'], tuple) or
        not numpy.all([isinstance(x, (int,long)) for x in d['shape']])):
        msg = "shape is not valid: %r"
        raise ValueError(msg % (d['shape'],))
    if not isinstance(d['fortran_order'], bool):
        msg = "fortran_order is not a valid bool: %r"
        raise ValueError(msg % (d['fortran_order'],))
    try:
        dtype = numpy.dtype(d['descr'])
    except TypeError, e:
        msg = "descr is not a valid dtype descriptor: %r"
        raise ValueError(msg % (d['descr'],))

    return d['shape'], d['fortran_order'], dtype

def write_array(fp, array, version=(1,0)):
    """
    Write an array to an NPY file, including a header.

    If the array is neither C-contiguous or Fortran-contiguous AND if the
    filelike object is not a real file object, then this function will have
    to copy data in memory.

    Parameters
    ----------
    fp : filelike object
        An open, writable file object or similar object with a `.write()`
        method.
    array : numpy.ndarray
        The array to write to disk.
    version : (int, int), optional
        The version number of the format.

    Raises
    ------
    ValueError
        If the array cannot be persisted.
    Various other errors
        If the array contains Python objects as part of its dtype, the
        process of pickling them may raise arbitrary errors if the objects
        are not picklable.

    """
    if version != (1, 0):
        msg = "we only support format version (1,0), not %s"
        raise ValueError(msg % (version,))
    fp.write(magic(*version))
    write_array_header_1_0(fp, header_data_from_array_1_0(array))
    if array.dtype.hasobject:
        # We contain Python objects so we cannot write out the data directly.
        # Instead, we will pickle it out with version 2 of the pickle protocol.
        cPickle.dump(array, fp, protocol=2)
    elif array.flags.f_contiguous and not array.flags.c_contiguous:
        # Use a suboptimal, possibly memory-intensive, but correct way to
        # handle Fortran-contiguous arrays.
        fp.write(array.data)
    else:
        if isinstance(fp, file):
            array.tofile(fp)
        else:
            # XXX: We could probably chunk this using something like
            # arrayterator.
            fp.write(array.tostring('C'))

def read_array(fp):
    """
    Read an array from an NPY file.

    Parameters
    ----------
    fp : filelike object
        If this is not a real file object, then this may take extra memory and
        time.

    Returns
    -------
    array : numpy.ndarray
        The array from the data on disk.

    Raises
    ------
    ValueError
        If the data is invalid.

    """
    version = read_magic(fp)
    if version != (1, 0):
        msg = "only support version (1,0) of file format, not %r"
        raise ValueError(msg % (version,))
    shape, fortran_order, dtype = read_array_header_1_0(fp)
    if len(shape) == 0:
        count = 1
    else:
        count = numpy.multiply.reduce(shape)

    # Now read the actual data.
    if dtype.hasobject:
        # The array contained Python objects. We need to unpickle the data.
        array = cPickle.load(fp)
    else:
        if isinstance(fp, file):
            # We can use the fast fromfile() function.
            array = numpy.fromfile(fp, dtype=dtype, count=count)
        else:
            # This is not a real file. We have to read it the memory-intensive
            # way.
            # XXX: we can probably chunk this to avoid the memory hit.
            data = fp.read(count * dtype.itemsize)
            array = numpy.fromstring(data, dtype=dtype, count=count)

        if fortran_order:
            array.shape = shape[::-1]
            array = array.transpose()
        else:
            array.shape = shape

    return array


def open_memmap(filename, mode='r+', dtype=None, shape=None,
                fortran_order=False, version=(1,0)):
    """
    Open a .npy file as a memory-mapped array.

    This may be used to read an existing file or create a new one.

    Parameters
    ----------
    filename : str
        The name of the file on disk. This may not be a file-like object.
    mode : str, optional
        The mode to open the file with. In addition to the standard file modes,
        'c' is also accepted to mean "copy on write". See `numpy.memmap` for
        the available mode strings.
    dtype : dtype, optional
        The data type of the array if we are creating a new file in "write"
        mode.
    shape : tuple of int, optional
        The shape of the array if we are creating a new file in "write"
        mode.
    fortran_order : bool, optional
        Whether the array should be Fortran-contiguous (True) or
        C-contiguous (False) if we are creating a new file in "write" mode.
    version : tuple of int (major, minor)
        If the mode is a "write" mode, then this is the version of the file
        format used to create the file.

    Returns
    -------
    marray : numpy.memmap
        The memory-mapped array.

    Raises
    ------
    ValueError
        If the data or the mode is invalid.
    IOError
        If the file is not found or cannot be opened correctly.

    See Also
    --------
    numpy.memmap

    """
    if not isinstance(filename, basestring):
        raise ValueError("Filename must be a string.  Memmap cannot use" \
                         " existing file handles.")

    if 'w' in mode:
        # We are creating the file, not reading it.
        # Check if we ought to create the file.
        if version != (1, 0):
            msg = "only support version (1,0) of file format, not %r"
            raise ValueError(msg % (version,))
        # Ensure that the given dtype is an authentic dtype object rather than
        # just something that can be interpreted as a dtype object.
        dtype = numpy.dtype(dtype)
        if dtype.hasobject:
            msg = "Array can't be memory-mapped: Python objects in dtype."
            raise ValueError(msg)
        d = dict(
            descr=dtype_to_descr(dtype),
            fortran_order=fortran_order,
            shape=shape,
        )
        # If we got here, then it should be safe to create the file.
        fp = open(filename, mode+'b')
        try:
            fp.write(magic(*version))
            write_array_header_1_0(fp, d)
            offset = fp.tell()
        finally:
            fp.close()
    else:
        # Read the header of the file first.
        fp = open(filename, 'rb')
        try:
            version = read_magic(fp)
            if version != (1, 0):
                msg = "only support version (1,0) of file format, not %r"
                raise ValueError(msg % (version,))
            shape, fortran_order, dtype = read_array_header_1_0(fp)
            if dtype.hasobject:
                msg = "Array can't be memory-mapped: Python objects in dtype."
                raise ValueError(msg)
            offset = fp.tell()
        finally:
            fp.close()

    if fortran_order:
        order = 'F'
    else:
        order = 'C'

    # We need to change a write-only mode to a read-write mode since we've
    # already written data to the file.
    if mode == 'w+':
        mode = 'r+'

    marray = numpy.memmap(filename, dtype=dtype, shape=shape, order=order,
        mode=mode, offset=offset)

    return marray
