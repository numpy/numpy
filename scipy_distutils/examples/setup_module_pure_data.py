#!/usr/bin/env python
from os.path import join
from glob import glob
from scipy_distutils.core      import setup
from scipy_distutils.misc_util import default_config_dict, get_path

def configuration(parent_package='',parent_path=None):
    package_name = 'module'
    config = default_config_dict(package_name, parent_package)

    local_path = get_path(__name__,parent_path)
    install_path = join(*config['name'].split('.'))

    config['data_files'].append((install_path,
                                 [join(local_path,'images.zip')]))

    config['data_files'].append((join(install_path,'images'),
                                 glob(join(local_path,'images','*.png'))))

    return config

if __name__ == '__main__':
    setup(**configuration(parent_path=''))
