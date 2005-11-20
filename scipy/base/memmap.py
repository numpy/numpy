import mmap
from numeric import uint8, ndarray

class memmap(ndarray):
    def __new__(self, name, dtype=uint8, mode='r', offset=0,
                size=-1, shape=None, swap=0, fortran=0):
        fid = file(name, mode+'b')

        if (mode == 'w') and size == -1:
            raise ValueError, "size must be given"
        
        if size == -1:
            fid.seek(0,2)
            size = fid.tell()

        if (mode == 'w'):
            fid.seek(size-1,0)
            fid.write(chr(0))
            fid.flush()

        if mode == 'r':
            acc = mmap.ACCESS_READ
        elif mode == 'w':
            acc = mmap.ACCESS_WRITE
        else:
            return NotImplemented

        mm = mmap.mmap(fid.fileno(), size, access=acc)

        if shape is None:
            shape = (size,)

        self = ndarray.__new__(self, shape, dtype=dtype, buffer=mm,
                               offset=offset, swap=swap, fortran=fortran)
        self._mmap = mm
        self._offset = offset
        self._mode = mode
        self._size = size
        self._name = name

        return self

    def sync(self):
        self._mmap.flush()

    
                     
