#!/usr/bin/env python
from scipy_distutils.core      import setup
from scipy_distutils.misc_util import default_config_dict, get_path, \
     merge_config_dicts, get_subpackages

def configuration(parent_package='',parent_path=None):
    package_name = 'module'
    config = default_config_dict(package_name, parent_package)

    local_path = get_path(__name__,parent_path)
    install_path = join(*config['name'].split('.'))

    config_list = [config]

    config_list += get_subpackages(local_path,
                                   parent=config['name'],
                                   parent_path=parent_path,
                                   include_packages = ['subpackage1','subpackage2']
                                   )

    config = merge_config_dicts(config_list)

    return config

if __name__ == '__main__':
    setup(**configuration(parent_path=''))
