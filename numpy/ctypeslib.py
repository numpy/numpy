__all__ = ['load_library', 'ndpointer', 'test', 'ctypes_load_library',
           'c_intp', 'as_ctypes', 'as_array']

import sys, os
from numpy import integer, ndarray, dtype as _dtype, deprecate, array
from numpy.core.multiarray import _flagdict, flagsobj

try:
    import ctypes
except ImportError:
    ctypes = None

if ctypes is None:
    def _dummy(*args, **kwds):
        raise ImportError, "ctypes is not available."
    ctypes_load_library = _dummy
    load_library = _dummy
    as_ctypes = _dummy
    as_array = _dummy
    from numpy import intp as c_intp
else:
    import numpy.core._internal as nic
    c_intp = nic._getintp_ctype()
    del nic

    # Adapted from Albert Strasheim
    def load_library(libname, loader_path):
        if ctypes.__version__ < '1.0.1':
            import warnings
            warnings.warn("All features of ctypes interface may not work " \
                          "with ctypes < 1.0.1")
        if '.' not in libname:
            # Try to load library with platform-specific name, otherwise
            # default to libname.so.  Sometimes, .so files are built
            # erroneously on non-linux platforms.
            libname_ext = ['%s.so' % libname]
            if sys.platform == 'win32':
                libname_ext.insert(0, '%s.dll' % libname)
            elif sys.platform == 'darwin':
                libname_ext.insert(0, '%s.dylib' % libname)

        loader_path = os.path.abspath(loader_path)
        if not os.path.isdir(loader_path):
            libdir = os.path.dirname(loader_path)
        else:
            libdir = loader_path

        for ln in libname_ext:
            try:
                libpath = os.path.join(libdir, ln)
                return ctypes.cdll[libpath]
            except OSError, e:
                pass

        raise e

    ctypes_load_library = deprecate(load_library, 'ctypes_load_library',
                                    'load_library')

def _num_fromflags(flaglist):
    num = 0
    for val in flaglist:
        num += _flagdict[val]
    return num

_flagnames = ['C_CONTIGUOUS', 'F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE',
              'OWNDATA', 'UPDATEIFCOPY']
def _flags_fromnum(num):
    res = []
    for key in _flagnames:
        value = _flagdict[key]
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
    """Array-checking restype/argtypes.

    An ndpointer instance is used to describe an ndarray in restypes
    and argtypes specifications.  This approach is more flexible than
    using, for example,

    POINTER(c_double)

    since several restrictions can be specified, which are verified
    upon calling the ctypes function.  These include data type
    (dtype), number of dimensions (ndim), shape and flags (e.g.
    'C_CONTIGUOUS' or 'F_CONTIGUOUS').  If a given array does not satisfy the
    specified restrictions, a TypeError is raised.

    Example:

        clib.somefunc.argtypes = [ndpointer(dtype=float64,
                                            ndim=1,
                                            flags='C_CONTIGUOUS')]
        clib.somefunc(array([1,2,3],dtype=float64))

    """

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

if ctypes is not None:
    ct = ctypes
    ################################################################
    # simple types

    # maps the numpy typecodes like '<f8' to simple ctypes types like
    # c_double. Filled in by prep_simple.
    _typecodes = {}

    def prep_simple(simple_type, typestr):
        """Given a ctypes simple type, construct and attach an
        __array_interface__ property to it if it does not yet have one.
        """
        try: simple_type.__array_interface__
        except AttributeError: pass
        else: return

        _typecodes[typestr] = simple_type

        def __array_interface__(self):
            return {'descr': [('', typestr)],
                    '__ref': self,
                    'strides': None,
                    'shape': (),
                    'version': 3,
                    'typestr': typestr,
                    'data': (ct.addressof(self), False),
                    }

        simple_type.__array_interface__ = property(__array_interface__)

    if sys.byteorder == "little":
        TYPESTR = "<%c%d"
    else:
        TYPESTR = ">%c%d"

    simple_types = [
        ((ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong), "i"),
        ((ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong, ct.c_ulonglong), "u"),
        ((ct.c_float, ct.c_double), "f"),
    ]

    # Prep that numerical ctypes types:
    for types, code in simple_types:
        for tp in types:
            prep_simple(tp, TYPESTR % (code, ct.sizeof(tp)))

    ################################################################
    # array types

    _ARRAY_TYPE = type(ct.c_int * 1)

    def prep_array(array_type):
        """Given a ctypes array type, construct and attach an
        __array_interface__ property to it if it does not yet have one.
        """
        try: array_type.__array_interface__
        except AttributeError: pass
        else: return

        shape = []
        ob = array_type
        while type(ob) == _ARRAY_TYPE:
            shape.append(ob._length_)
            ob = ob._type_
        shape = tuple(shape)
        ai = ob().__array_interface__
        descr = ai['descr']
        typestr = ai['typestr']

        def __array_interface__(self):
            return {'descr': descr,
                    '__ref': self,
                    'strides': None,
                    'shape': shape,
                    'version': 3,
                    'typestr': typestr,
                    'data': (ct.addressof(self), False),
                    }

        array_type.__array_interface__ = property(__array_interface__)

    ################################################################
    # public functions

    def as_array(obj):
        """Create a numpy array from a ctypes array.  The numpy array
        shares the memory with the ctypes object."""
        tp = type(obj)
        try: tp.__array_interface__
        except AttributeError: prep_array(tp)
        return array(obj, copy=False)

    def as_ctypes(obj):
        """Create and return a ctypes object from a numpy array.  Actually
        anything that exposes the __array_interface__ is accepted."""
        ai = obj.__array_interface__
        if ai["strides"]:
            raise TypeError("strided arrays not supported")
        if ai["version"] != 3:
            raise TypeError("only __array_interface__ version 3 supported")
        addr, readonly = ai["data"]
        if readonly:
            raise TypeError("readonly arrays unsupported")
        tp = _typecodes[ai["typestr"]]
        for dim in ai["shape"][::-1]:
            tp = tp * dim
        result = tp.from_address(addr)
        result.__keep = ai
        return result


def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)
