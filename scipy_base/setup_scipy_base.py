#!/usr/bin/env python

import os, sys
from glob import glob
import shutil

def configuration(parent_package='',parent_path=None):
    from scipy_distutils.system_info import get_info, NumericNotFoundError
    from scipy_distutils.core import Extension
    from scipy_distutils.misc_util import get_path,default_config_dict,dot_join
    from scipy_distutils.misc_util import get_path,default_config_dict,\
         dot_join,SourceGenerator

    package = 'scipy_base'
    local_path = get_path(__name__,parent_path)
    config = default_config_dict(package,parent_package)

    numpy_info = get_info('numpy')
    if not numpy_info:
        raise NumericNotFoundError, NumericNotFoundError.__doc__

    # extra_compile_args -- trying to find something that is binary compatible
    #                       with msvc for returning Py_complex from functions
    extra_compile_args=[]
    
    # fastumath module
    # scipy_base.fastumath module
    umath_c_sources = ['fastumathmodule.c',
                       'fastumath_unsigned.inc','fastumath_nounsigned.inc']
    umath_c_sources = [os.path.join(local_path,x) for x in umath_c_sources]
    umath_c = SourceGenerator(func = None,
                              target = os.path.join(local_path,'fastumathmodule.c'),
                              sources = umath_c_sources)
    sources = [umath_c, os.path.join(local_path,'isnan.c')]
    define_macros = []
    if sys.byteorder == "little":
        define_macros.append(('USE_MCONF_LITE_LE',None))
    else:
        define_macros.append(('USE_MCONF_LITE_BE',None))
    ext = Extension(dot_join(package,'fastumath'),sources,
                    define_macros = define_macros,
                    extra_compile_args=extra_compile_args,
                    depends = umath_c_sources)
    config['ext_modules'].append(ext)
 
    # _compiled_base module
    sources = ['_compiled_base.c']
    sources = [os.path.join(local_path,x) for x in sources]
    depends = ['_scipy_mapping.c','_scipy_number.c']
    depends = [os.path.join(local_path,x) for x in depends]

    ext = Extension(dot_join(package,'_compiled_base'),sources,
                    depends = depends)
    config['ext_modules'].append(ext)

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
          **configuration()
          )
