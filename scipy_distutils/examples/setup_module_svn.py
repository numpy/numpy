#!/usr/bin/env python
from scipy_distutils.core      import setup, Extension
from scipy_distutils.misc_util import default_config_dict, \
     dot_join, generate_svn_version_py

def configuration(parent_package='',parent_path=None):
    package_name = 'module'
    config = default_config_dict(package_name, parent_package)

    ext = Extension(dot_join(config["name"],'__svn_version__'),
                    sources=[generate_svn_version_py])
    ext.local_path = local_path
    config['ext_modules'].append(ext)

    return config

if __name__ == '__main__':
    setup(**configuration(parent_path=''))
