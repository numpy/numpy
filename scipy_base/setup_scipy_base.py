#!/usr/bin/env python
import os, sys
from glob import glob
import shutil

class _CleanUpFile:
    """CleanUpFile deletes the specified filename when self is destroyed."""
    def __init__(self, name):
        self.name = name
    def __del__(self):
        os.remove(self.name)
        # pass # leave source around for debugging

def _temp_copy(_from, _to):
    """temp_copy copies a named file into a named temporary file.
    The temporary will be deleted when the setupext module is destructed.
    """
    # Copy the file data from _from to _to
    s = open(_from).read()
    open(_to,"w+").write(s)
    # Suppress object rebuild by preserving time stamps.
    stats = os.stat(_from)
    os.utime(_to, (stats.st_atime, stats.st_mtime))
    # Make an object to eliminate the temporary file at exit time.
    globals()["_cleanup_"+_to] = _CleanUpFile(_to)

def _config_compiled_base(package, local_path, numerix_prefix, macro, info):
    """_config_compiled_base returns the Extension object for an
    Numeric or numarray specific version of _compiled_base.
    """
    from scipy_distutils.system_info import dict_append
    from scipy_distutils.core import Extension
    from scipy_distutils.misc_util import dot_join
    module = numerix_prefix + "_compiled_base"
    source = module + '.c'
    _temp_copy(os.path.join(local_path, "_compiled_base.c"),
               os.path.join(local_path, source))
    sources = [source]
    sources = [os.path.join(local_path,x) for x in sources]
    depends = sources
    ext_args = {'name':dot_join(package, module),
                'sources':sources,
                'depends':depends,
                'define_macros':[(macro,1)],
                }
    dict_append(ext_args,**info)
    return Extension(**ext_args)
    
def configuration(parent_package='',parent_path=None):
    from scipy_distutils.system_info import get_info, dict_append
    from scipy_distutils.core import Extension
    from scipy_distutils.misc_util import get_path,default_config_dict,dot_join
    from scipy_distutils.misc_util import get_path,default_config_dict,\
         dot_join,SourceGenerator

    package = 'scipy_base'
    local_path = get_path(__name__,parent_path)
    config = default_config_dict(package,parent_package)

    numpy_info = get_info('numpy',notfound_action=2)

    # extra_compile_args -- trying to find something that is binary compatible
    #                       with msvc for returning Py_complex from functions
    extra_compile_args=[]
    
    # fastumath module
    # scipy_base.fastumath module
    umath_c_sources = ['fastumathmodule.c',
                       'fastumath_unsigned.inc',
                       'fastumath_nounsigned.inc',
                       '_scipy_mapping.c',
                       '_scipy_number.c']
    depends = umath_c_sources  # ????
    depends = [os.path.join(local_path,x) for x in depends]
    umath_c_sources = [os.path.join(local_path,x) for x in umath_c_sources]
    umath_c = SourceGenerator(func = None,
                              target = os.path.join(local_path,'fastumathmodule.c'),
                              sources = umath_c_sources)
    sources = [umath_c, os.path.join(local_path,'isnan.c')]
    define_macros = []
    undef_macros = []
    libraries = []
    if sys.byteorder == "little":
        define_macros.append(('USE_MCONF_LITE_LE',None))
    else:
        define_macros.append(('USE_MCONF_LITE_BE',None))
    if sys.platform in ['win32']:
        undef_macros.append('HAVE_INVERSE_HYPERBOLIC')
    else:
        libraries.append('m')
        define_macros.append(('HAVE_INVERSE_HYPERBOLIC',None))

    ext_args = {'name':dot_join(package,'fastumath'),
                 'sources':sources,
                 'define_macros': define_macros,
                 'undef_macros': undef_macros,
                 'libraries': libraries,
                 'extra_compile_args': extra_compile_args,
                 'depends': umath_c_sources}
    dict_append(ext_args,**numpy_info)
    config['ext_modules'].append(Extension(**ext_args))
 
    # _compiled_base module for Numeric: _nc_compiled_base
    _nc_compiled_base_ext = _config_compiled_base(
        package, local_path, "_nc", "NUMERIC", numpy_info)
    config['ext_modules'].append(_nc_compiled_base_ext)

    # _compiled_base module for numarray: _na_compiled_base
    numarray_info = get_info('numarray')
    if numarray_info:
        _na_compiled_base_ext = _config_compiled_base(
            package, local_path, "_na", "NUMARRAY", numarray_info)
        config['ext_modules'].append(_na_compiled_base_ext)
        
    # display_test module
    sources = [os.path.join(local_path,'src','display_test.c')]
    x11 = get_info('x11')
    if x11:
        x11['define_macros'] = [('HAVE_X11',None)]
    ext = Extension(dot_join(package,'display_test'), sources, **x11)
    config['ext_modules'].append(ext)

    return config

if __name__ == '__main__':
    from scipy_base_version import scipy_base_version
    print 'scipy_base Version',scipy_base_version
    from scipy_distutils.core import setup

    setup(version = scipy_base_version,
          maintainer = "SciPy Developers",
          maintainer_email = "scipy-dev@scipy.org",
          description = "SciPy base module",
          url = "http://www.scipy.org",
          license = "SciPy License (BSD Style)",
          **configuration(parent_path='')
          )
