#!/usr/bin/env python
"""
Bundle of SciPy core modules:
  scipy_test, scipy_distutils

Usage:
   python setup_scipy_core.py install
   python setup_scipy_core.py sdist -f -t MANIFEST_scipy_core.in
"""

from scipy_distutils import scipy_distutils_version as v1
from scipy_test import scipy_test_version as v2

major = max(v1.major,v2.major)
minor = max(v1.minor,v2.minor)
micro = max(v1.micro,v2.micro)
release_level = min(v1.release_level,v2.release_level)
cvs_minor = v1.cvs_minor + v2.cvs_minor
cvs_serial = v1.cvs_serial + v2.cvs_serial

scipy_core_version = '%(major)d.%(minor)d.%(micro)d_%(release_level)s'\
                     '_%(cvs_minor)d.%(cvs_serial)d' % (locals ())

if __name__ == "__main__":
    import os,sys
    from distutils.core import setup
    print 'SciPy core Version %s' % scipy_core_version
    setup (name = "SciPy_core",
           version = scipy_core_version,
           maintainer = "SciPy Developers",
           maintainer_email = "scipy-dev@scipy.org",
           description = "SciPy core modules: scipy_test and scipy_distutils",
           license = "SciPy License (BSD Style)",
           url = "http://www.scipy.org",
           packages=['scipy_distutils',
                     'scipy_distutils.command',
                     'scipy_test'],
           package_dir = {'scipy_distutils':'scipy_distutils',
                          'scipy_test':'scipy_test',
                          },
           )
