import os
from scipy_distutils.misc_util import get_path, default_config_dict

def configuration(parent_package=''):
    parent_path = parent_package
    if parent_package:
        parent_package += '.'
    local_path = get_path(__name__)

    config = default_config_dict()
    package = 'scipy_distutils'
    config['packages'].append(parent_package+package)
    config['package_dir'][package] = local_path 
    package = 'scipy_distutils.command'   
    config['packages'].append(parent_package+package),
    config['package_dir'][package] = os.path.join(local_path,'command')    
    return config
