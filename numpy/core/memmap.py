__all__ = ['memmap']

import mmap
import warnings
from numeric import uint8, ndarray, dtype

dtypedescr = dtype
valid_filemodes = ["r", "c", "r+", "w+"]
writeable_filemodes = ["r+","w+"]

mode_equivalents = {
    "readonly":"r",
    "copyonwrite":"c",
    "readwrite":"r+",
    "write":"w+"
    }

class memmap(ndarray):
    """Create a memory-map to an array stored in a file on disk.

    Memory-mapped files are used for accessing small segments of large files
    on disk, without reading the entire file into memory.  Numpy's memmaps are
    array-like objects.  This differs from python's mmap module which are
    file-like objects.

    Parameters
    ----------
    filename : string or file-like object
        The file name or file object to be used as the array data
        buffer.
    dtype : data-type, optional
        The data-type used to interpret the file contents.
        Default is uint8
    mode : {'r', 'r+', 'w+', 'c'}, optional
        The mode to open the file.
        'r',  open existing file for read-only
        'r+', open existing file for read-write
        'w+', create or overwrite existing file and open for read-write
        'c',  copy-on-write, assignments effect data in memory, but changes
              are not saved to disk.  File on disk is read-only.
        Default is 'r+'
    offset : integer, optional
        Byte offset into the file to start the array data. Should be a
        multiple of the data-type of the data.  Requires shape=None.
        Default is 0
    shape : tuple, optional
        The desired shape of the array. If None, the returned array will be 1-D
        with the number of elements determined by file size and data-type.
        Default is None
    order : {'C', 'F'}, optional
        Specify the order of the N-D array, C or Fortran ordered. This only
        has an effect if the shape is greater than 2-D.
        Default is 'C'

    Methods
    -------
    close : close the memmap file
    flush : flush any changes in memory to file on disk
        When you delete a memmap object, flush is called first to write
        changes to disk before removing the object.

    Returns
    -------
    memmap : array-like memmap object
        The memmap object can be used anywhere an ndarray is accepted.
        If fp is a memmap, isinstance(fp, numpy.ndarray) will return True.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.arange(12, dtype='float32')
    >>> data.resize((3,4))

    >>> # Using a tempfile so doctest doesn't write files to your directory.
    >>> # You would use a 'normal' filename.
    >>> from tempfile import mkdtemp
    >>> import os.path as path
    >>> filename = path.join(mkdtemp(), 'newfile.dat')

    >>> # Create a memmap with dtype and shape that matches our data
    >>> fp = np.memmap(filename, dtype='float32', mode='w+', shape=(3,4))
    >>> fp
    memmap([[ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.]], dtype=float32)

    >>> # Write data to memmap array
    >>> fp[:] = data[:]
    >>> fp
    memmap([[  0.,   1.,   2.,   3.],
           [  4.,   5.,   6.,   7.],
           [  8.,   9.,  10.,  11.]], dtype=float32)

    >>> # Deletion flushes memory changes to disk before removing the object.
    >>> del fp
    >>> # Load the memmap and verify data was stored
    >>> newfp = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
    >>> newfp
    memmap([[  0.,   1.,   2.,   3.],
           [  4.,   5.,   6.,   7.],
           [  8.,   9.,  10.,  11.]], dtype=float32)

    >>> # read-only memmap
    >>> fpr = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
    >>> fpr.flags.writeable
    False
    >>> # Cannot assign to read-only, obviously
    >>> fpr[0, 3] = 56
    Traceback (most recent call last):
        ...
    RuntimeError: array is not writeable

    >>> # copy-on-write memmap
    >>> fpc = np.memmap(filename, dtype='float32', mode='c', shape=(3,4))
    >>> fpc.flags.writeable
    True
    >>> # Can assign to copy-on-write array, but values are only written
    >>> # into the memory copy of the array, and not written to disk.
    >>> fpc
    memmap([[  0.,   1.,   2.,   3.],
           [  4.,   5.,   6.,   7.],
           [  8.,   9.,  10.,  11.]], dtype=float32)
    >>> fpc[0,:] = 0
    >>> fpc
    memmap([[  0.,   0.,   0.,   0.],
           [  4.,   5.,   6.,   7.],
           [  8.,   9.,  10.,  11.]], dtype=float32)
    >>> # file on disk is unchanged
    >>> fpr
    memmap([[  0.,   1.,   2.,   3.],
           [  4.,   5.,   6.,   7.],
           [  8.,   9.,  10.,  11.]], dtype=float32)

    >>> # offset into a memmap
    >>> fpo = np.memmap(filename, dtype='float32', mode='r', offset=16)
    >>> fpo
    memmap([  4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.], dtype=float32)

    """

    __array_priority__ = -100.0
    def __new__(subtype, filename, dtype=uint8, mode='r+', offset=0,
                shape=None, order='C'):
        try:
            mode = mode_equivalents[mode]
        except KeyError:
            if mode not in valid_filemodes:
                raise ValueError("mode must be one of %s" % \
                                 (valid_filemodes + mode_equivalents.keys()))

        if hasattr(filename,'read'):
            fid = filename
        else:
            fid = file(filename, (mode == 'c' and 'r' or mode)+'b')

        if (mode == 'w+') and shape is None:
            raise ValueError, "shape must be given"

        fid.seek(0,2)
        flen = fid.tell()
        descr = dtypedescr(dtype)
        _dbytes = descr.itemsize

        if shape is None:
            bytes = flen-offset
            if (bytes % _dbytes):
                fid.close()
                raise ValueError, "Size of available data is not a "\
                      "multiple of data-type size."
            size = bytes // _dbytes
            shape = (size,)
        else:
            if not isinstance(shape, tuple):
                shape = (shape,)
            size = 1
            for k in shape:
                size *= k

        bytes = long(offset + size*_dbytes)

        if mode == 'w+' or (mode == 'r+' and flen < bytes):
            fid.seek(bytes-1,0)
            fid.write(chr(0))
            fid.flush()

        if mode == 'c':
            acc = mmap.ACCESS_COPY
        elif mode == 'r':
            acc = mmap.ACCESS_READ
        else:
            acc = mmap.ACCESS_WRITE

        mm = mmap.mmap(fid.fileno(), bytes, access=acc)

        self = ndarray.__new__(subtype, shape, dtype=descr, buffer=mm,
                               offset=offset, order=order)
        self._mmap = mm
# Should get rid of these...  Are they used?
        self._offset = offset
        self._mode = mode
        self._size = size
        self._name = filename
        return self

    def __array_finalize__(self, obj):
        if hasattr(obj, '_mmap'):
            self._mmap = obj._mmap
        else:
            self._mmap = None

    def flush(self):
        """Flush any changes in the array to the file on disk."""
        if self._mmap is not None:
            self._mmap.flush()

    def sync(self):
        """Flush any changes in the array to the file on disk."""
        warnings.warn("Use ``flush``.", DeprecationWarning)
        self.flush()

    def _close(self):
        """Close the memmap file.  Only do this when deleting the object."""
        if self.base is self._mmap:
            self._mmap.close()
            self._mmap = None

        # DEV NOTE: This error is raised on the deletion of each row
        # in a view of this memmap.  Python traps exceptions in
        # __del__ and prints them to stderr.  Suppressing this for now
        # until memmap code is cleaned up and and better tested for
        # numpy v1.1 Objects that do not have a python mmap instance
        # as their base data array, should not do anything in the
        # close anyway.
        #elif self._mmap is not None:
            #raise ValueError, "Cannot close a memmap that is being used " \
            #      "by another object."

    def close(self):
        """Close the memmap file. Does nothing."""
        warnings.warn("``close`` is deprecated on memmap arrays.  Use del",
                      DeprecationWarning)

    def __del__(self):
        if self._mmap is not None:
            try:
                # First run tell() to see whether file is open
                self._mmap.tell()
            except ValueError:
                pass
            else:
                # flush any changes to disk, even if it's a view
                self.flush()
                self._close()
