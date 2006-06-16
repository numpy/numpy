import sys
from numpy.core.numerictypes import obj2sctype

__all__ = ['issubclass_', 'get_numpy_include', 'issubsctype', 'deprecate']

def issubclass_(arg1, arg2):
    try:
        return issubclass(arg1, arg2)
    except TypeError:
        return False

def issubsctype(arg1, arg2):
    return issubclass(obj2sctype(arg1), obj2sctype(arg2))

def get_numpy_include():
    """Return the directory in the package that contains the numpy/*.h header
    files.

    Extension modules that need to compile against numpy should use this
    function to locate the appropriate include directory. Using distutils:

      import numpy
      Extension('extension_name', ...
                include_dirs=[numpy.get_numpy_include()])
    """
    from numpy.distutils.misc_util import get_numpy_include_dirs
    include_dirs = get_numpy_include_dirs()
    assert len(include_dirs)==1,`include_dirs`
    return include_dirs[0]

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
