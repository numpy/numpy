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

if __name__ == '__main__':
    from scipy_distutils_version import scipy_distutils_version
    print 'scipy_distutils Version',scipy_distutils_version
    from scipy_distutils.core import setup
    print 'scipy_distutils Version',scipy_distutils_version
    setup (name = "scipy_distutils",
           version = scipy_distutils_version,
           description = "Changes to distutils needed for SciPy -- mostly Fortran support",
           author = "Travis Oliphant, Eric Jones, and Pearu Peterson",
           author_email = "scipy-dev@scipy.org",
           license = "SciPy License (BSD Style)",
           url = 'http://www.scipy.org',
           **configuration()
           )
