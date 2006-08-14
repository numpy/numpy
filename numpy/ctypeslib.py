__all__ = ['ctypes_load_library', 'ndpointer']


# Adapted from Albert Strasheim
def ctypes_load_library(libname, loader_path):
    if '.' not in libname:
        if sys.platform == 'win32':
            libname = '%s.dll' % libname
        elif sys.platform == 'darwin':
            libname = '%s.dylib' % libname
        else:
            libname = '%s.so' % libname
    loader_path = os.path.abspath(loader_path)
    if not os.path.isdir(loader_path):
        libdir = os.path.dirname(loader_path)
    else:
        libdir = loader_path
    import ctypes
    libpath = os.path.join(libdir, libname)
    return ctypes.cdll[libpath]

def _num_fromflags(flaglist):
    num = 0
    for val in flaglist:
        num += _flagdict[val]
    return num

def _flags_fromnum(num):
    res = []
    for key, value in _flagdict.items():
        if (num & value):
            res.append(key)
    return res

class _ndptr(object):
    def from_param(cls, obj):
        if not isinstance(obj, ndarray):
            raise TypeError, "argument must be an ndarray"
        if cls._dtype_ is not None \
               and obj.dtype != cls._dtype_:
            raise TypeError, "array must have data type %s" % cls._dtype_
        if cls._ndim_ is not None \
               and obj.ndim != cls._ndim_:
            raise TypeError, "array must have %d dimension(s)" % cls._ndim_
        if cls._shape_ is not None \
               and obj.shape != cls._shape_:
            raise TypeError, "array must have shape %s" % str(cls._shape_)
        if cls._flags_ is not None \
               and ((obj.flags.num & cls._flags_) != cls._flags_):
            raise TypeError, "array must have flags %s" % \
                  _flags_fromnum(cls._flags_)
        return obj.ctypes
    from_param = classmethod(from_param)


# Factory for an array-checking class with from_param defined for
#  use with ctypes argtypes mechanism
_pointer_type_cache = {}
def ndpointer(dtype=None, ndim=None, shape=None, flags=None):
    if dtype is not None:
        dtype = _dtype(dtype)
    num = None
    if flags is not None:
        if isinstance(flags, str):
            flags = flags.split(',')
        elif isinstance(flags, (int, integer)):
            num = flags
            flags = _flags_fromnum(num)
        elif isinstance(flags, flagsobj):
            num = flags.num
            flags = _flags_fromnum(num)
        if num is None:
            try:
                flags = [x.strip().upper() for x in flags]
            except:
                raise TypeError, "invalid flags specification"
            num = _num_fromflags(flags)
    try:
        return _pointer_type_cache[(dtype, ndim, shape, num)]
    except KeyError:
        pass
    if dtype is None:
        name = 'any'
    elif dtype.names:
        name = str(id(dtype))
    else:
        name = dtype.str
    if ndim is not None:
        name += "_%dd" % ndim
    if shape is not None:
        try:
            strshape = [str(x) for x in shape]
        except TypeError:
            strshape = [str(shape)]
            shape = (shape,)
        shape = tuple(shape)
        name += "_"+"x".join(strshape)
    if flags is not None:
        name += "_"+"_".join(flags)
    else:
        flags = []
    klass = type("ndpointer_%s"%name, (_ndptr,),
                 {"_dtype_": dtype,
                  "_shape_" : shape,
                  "_ndim_" : ndim,
                  "_flags_" : num})
    _pointer_type_cache[dtype] = klass
    return klass
