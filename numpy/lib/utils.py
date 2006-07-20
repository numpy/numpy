import sys, os
from numpy.core.numerictypes import obj2sctype
from numpy.core.multiarray import dtype

__all__ = ['issubclass_', 'get_numpy_include', 'issubsctype',
           'issubdtype', 'deprecate', 'get_numarray_include',
	   'get_include', 'ctypes_load_library']

def issubclass_(arg1, arg2):
    try:
        return issubclass(arg1, arg2)
    except TypeError:
        return False

def issubsctype(arg1, arg2):
    return issubclass(obj2sctype(arg1), obj2sctype(arg2))

def issubdtype(arg1, arg2):
    return issubclass(dtype(arg1).type, dtype(arg2).type)

def get_include():
    """Return the directory in the package that contains the numpy/*.h header
    files.

    Extension modules that need to compile against numpy should use this
    function to locate the appropriate include directory. Using distutils:

      import numpy
      Extension('extension_name', ...
                include_dirs=[numpy.get_include()])
    """
    from numpy.distutils.misc_util import get_numpy_include_dirs
    include_dirs = get_numpy_include_dirs()
    assert len(include_dirs)==1,`include_dirs`
    return include_dirs[0]

def get_numarray_include(type=None):
    """Return the directory in the package that contains the numpy/*.h header
    files.

    Extension modules that need to compile against numpy should use this
    function to locate the appropriate include directory. Using distutils:

      import numpy
      Extension('extension_name', ...
                include_dirs=[numpy.get_include()])
    """
    from numpy.numarray import get_numarray_include_dirs
    from numpy.distutils.misc_util import get_numpy_include_dirs
    include_dirs = get_numarray_include_dirs()
    if type is None:
        return include_dirs[0]
    else:
        return include_dirs + get_numpy_include_dirs()


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


if sys.version_info < (2, 4):
    # Can't set __name__ in 2.3
    import new
    def _set_function_name(func, name):
        func = new.function(func.func_code, func.func_globals,
                            name, func.func_defaults, func.func_closure)
        return func
else:
    def _set_function_name(func, name):
        func.__name__ = name
        return func

def deprecate(func, oldname, newname):
    import warnings
    def newfunc(*args,**kwds):
        warnings.warn("%s is deprecated, use %s" % (oldname, newname),
                      DeprecationWarning)
        return func(*args, **kwds)
    newfunc = _set_function_name(newfunc, oldname)
    doc = func.__doc__
    depdoc = '%s is DEPRECATED in numpy: use %s instead' % (oldname, newname,)
    if doc is None:
        doc = depdoc
    else:
        doc = '\n'.join([depdoc, doc])
    newfunc.__doc__ = doc
    try:
        d = func.__dict__
    except AttributeError:
        pass
    else:
        newfunc.__dict__.update(d)
    return newfunc


get_numpy_include = deprecate(get_include, 'get_numpy_include', 'get_include')


