#!/usr/bin/env python

import os, sys
from glob import glob
from scipy_distutils.core import Extension
from scipy_distutils.misc_util import get_path,default_config_dict,dot_join
import shutil

def configuration(parent_package=''):
    package = 'scipy_base'
    local_path = get_path(__name__)
    config = default_config_dict(package,parent_package)

    # extra_compile_args -- trying to find something that is binary compatible
    #                       with msvc for returning Py_complex from functions
    extra_compile_args=[]
    
    # fastumath module
    # scipy_base.fastumath module
    sources = ['fastumathmodule.c','isnan.c']
    sources = [os.path.join(local_path,x) for x in sources]
    define_macros = []
    if sys.byteorder != "little":
        define_macros.append(('USE_MCONF_LITE_BE',1))
    ext = Extension(dot_join(package,'fastumath'),sources,
                    define_macros = define_macros,
                    extra_compile_args=extra_compile_args)
    config['ext_modules'].append(ext)
 
    # _compiled_base module
    sources = ['_compiled_base.c']
    sources = [os.path.join(local_path,x) for x in sources]
    ext = Extension(dot_join(package,'_compiled_base'),sources)
    config['ext_modules'].append(ext)

    return config

if __name__ == '__main__':
    from scipy_base_version import scipy_base_version
    print 'scipy_base Version',scipy_base_version
    if sys.platform == 'win32':
        from scipy_distutils.mingw32_support import *
    from scipy_distutils.core import setup
    setup(version = scipy_base_version,
          maintainer = "SciPy Developers",
          maintainer_email = "scipy-dev@scipy.org",
          description = "SciPy base module",
          url = "http://www.scipy.org",
          license = "SciPy License (BSD Style)",
          **configuration()
          )
