#!/usr/bin/env python

import os
from scipy_distutils.misc_util import get_path, default_config_dict,\
     dot_join

def configuration(parent_package=''):
    package = 'scipy_test'
    local_path = get_path(__name__)

    config = default_config_dict(package,parent_package)
    config['packages'].append(dot_join(parent_package,package))
    config['package_dir'][package] = local_path 
    return config

if __name__ == '__main__':    
    from scipy_distutils.core import setup
    setup(**configuration())
