#!/usr/bin/env python

import os
from scipy_distutils.misc_util import default_config_dict

def configuration(parent_package='',parent_path=None):
    package = 'scipy_test'
    config = default_config_dict(package,parent_package)
    return config

if __name__ == '__main__':
    from scipy_test_version import scipy_test_version
    print 'scipy_test Version',scipy_test_version
    from scipy_distutils.core import setup
    setup(version = scipy_test_version,
          maintainer = "SciPy Developers",
          maintainer_email = "scipy-dev@scipy.org",
          description = "SciPy test module",
          url = "http://www.scipy.org",
          license = "SciPy License (BSD Style)",
          **configuration()
          )
