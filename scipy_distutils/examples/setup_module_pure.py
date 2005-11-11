#!/usr/bin/env python
from scipy_distutils.core      import setup
from scipy_distutils.misc_util import default_config_dict

def configuration(parent_package='',parent_path=None):
    package_name = 'module'
    config = default_config_dict(package_name, parent_package)
    return config

if __name__ == '__main__':
    setup(**configuration(parent_path=''))
