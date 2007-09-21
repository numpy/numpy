__all__ = ['memmap']

import mmap
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
    __array_priority__ = -100.0
    def __new__(subtype, name, dtype=uint8, mode='r+', offset=0,
                shape=None, order='C'):
        try:
            mode = mode_equivalents[mode]
        except KeyError:
            if mode not in valid_filemodes:
                raise ValueError("mode must be one of %s" % \
                                 (valid_filemodes + mode_equivalents.keys()))

        fid = file(name, (mode == 'c' and 'r' or mode)+'b')

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
        self._offset = offset
        self._mode = mode
        self._size = size
        self._name = name
        fid.close()
        return self

    def __array_finalize__(self, obj):
        if obj is not None:
            if not isinstance(obj, memmap):
                raise ValueError, "Cannot create a memmap array that way"
            self._mmap = obj._mmap
        else:
            self._mmap = None

    def sync(self):
        self._mmap.flush()

    def close(self):
        if (self.base is self._mmap):
            self._mmap.close()
        else:
            raise ValueError, "Cannot close a memmap that is being used " \
                  "by another object."

    def __del__(self):
        if self._mmap is not None:
            self._mmap.flush()
            try:
                self.close()
            except:
                pass
