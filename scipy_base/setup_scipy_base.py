#!/usr/bin/env python

import os
from glob import glob
from scipy_distutils.core import Extension
from scipy_distutils.misc_util import get_path, default_config_dict

def configuration(parent_package=''):
    parent_path = parent_package
    if parent_package:
        parent_package += '.'
    local_path = get_path(__name__)

    config = default_config_dict()
    config['packages'].append(parent_package+'scipy_base')
    config['package_dir'][parent_package+'scipy_base'] = local_path

    # fastumath module
    sources = ['fastumathmodule.c']
    sources = [os.path.join(local_path,x) for x in sources]
    ext = Extension('scipy_base.fastumath',sources,libraries=[])
    config['ext_modules'].append(ext)
    

    return config

if __name__ == '__main__':    
    from scipy_distutils.core import setup
    setup(**configuration())
