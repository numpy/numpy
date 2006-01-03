from numerictypes import obj2dtype

__all__ = ['issubclass_', 'get_scipy_include', 'issubdtype']

def issubclass_(arg1, arg2):
    try:
        return issubclass(arg1, arg2)
    except TypeError:
        return False

def issubdtype(arg1, arg2):
    return issubclass(obj2dtype(arg1), obj2dtype(arg2))
    
def get_scipy_include():
    """Return the directory in the package that contains the scipy/*.h header 
    files.
    
    Extension modules that need to compile against scipy.base should use this
    function to locate the appropriate include directory. Using distutils:
    
      import scipy
      Extension('extension_name', ...
                include_dirs=[scipy.get_scipy_include()])
    """
    from scipy.distutils.misc_util import get_scipy_include_dirs
    include_dirs = get_scipy_include_dirs()
    assert len(include_dirs)==1,`include_dirs`
    return include_dirs[0]
