#!/usr/bin/env python

import os, sys
from glob import glob
from scipy_distutils.core import Extension
from scipy_distutils.misc_util import get_path, default_config_dict,dot_join
import shutil

def configuration(parent_package=''):
    parent_path = parent_package
    if parent_package:
        parent_package += '.'
    local_path = get_path(__name__)

    config = default_config_dict()
    config['packages'].append(parent_package+'scipy_base')
    config['package_dir'][parent_package+'scipy_base'] = local_path

    config['packages'].append(dot_join(parent_package,'scipy_base.tests'))
    test_path = os.path.join(local_path,'tests')
    config['package_dir']['scipy_base.tests'] = test_path

    # extra_compile_args -- trying to find something that is binary compatible
    #                       with msvc for returning Py_complex from functions
    extra_compile_args=[]
    
    # fastumath module
    # scipy_base.fastumath module
    sources = ['fastumathmodule.c','isnan.c']
    sources = [os.path.join(local_path,x) for x in sources]
    ext = Extension('scipy_base.fastumath',sources,libraries=[],
                    extra_compile_args=extra_compile_args)
    config['ext_modules'].append(ext)

    # Test to see if big or little-endian machine and get correct default
    #   mconf.h module.
    if sys.byteorder == "little":
        print "### Little Endian detected ####"
        shutil.copy2(os.path.join(local_path,'mconf_lite_LE.h'),os.path.join(local_path,'mconf_lite.h'))
    else:
        print "### Big Endian detected ####"
        shutil.copy2(os.path.join(local_path,'mconf_lite_BE.h'),os.path.join(local_path,'mconf_lite.h'))

    return config

if __name__ == '__main__':    
    if sys.platform == 'win32':
        from scipy_distutils.mingw32_support import *
    from scipy_distutils.core import setup
    setup(**configuration())
