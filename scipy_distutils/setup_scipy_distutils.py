#!/usr/bin/env python

import os
from misc_util import get_path, default_config_dict, dot_join

def configuration(parent_package=''):
    package = 'scipy_distutils'
    local_path = get_path(__name__)
    config = default_config_dict(package,parent_package)

    sub_package = dot_join(package,'command')
    config['packages'].append(dot_join(parent_package,sub_package))
    config['package_dir'][sub_package] = os.path.join(local_path,'command')
    return config

if __name__ == '__main__':
    from scipy_distutils_version import scipy_distutils_version
    print 'scipy_distutils Version',scipy_distutils_version
    from distutils.core import setup
    config = configuration()
    [config.__delitem__(k) for k,v in config.items() if not v]
    setup(version = scipy_distutils_version,
          description = "Changes to distutils needed for SciPy "\
          "-- mostly Fortran support",
          author = "Travis Oliphant, Eric Jones, and Pearu Peterson",
          author_email = "scipy-dev@scipy.org",
          license = "SciPy License (BSD Style)",
          url = 'http://www.scipy.org',
          **config
          )
