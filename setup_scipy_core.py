#!/usr/bin/env python
"""
Bundle of SciPy core modules:
  scipy_test
  scipy_distutils
  scipy_base

Usage:
   python setup_scipy_core.py install
   python setup_scipy_core.py sdist -f -t MANIFEST_scipy_core.in
"""

import os
import sys

from scipy_distutils.misc_util import default_config_dict
from scipy_distutils.misc_util import get_path, merge_config_dicts

bundle_packages = ['scipy_distutils','scipy_test','scipy_base']

def get_package_config(name):
    sys.path.insert(0,name)
    try:
        mod = __import__('setup_'+name)
        config = mod.configuration()
    finally:
        del sys.path[0]
    return config

def get_package_version(name):
    sys.path.insert(0,name)
    try:
        mod = __import__(name+'_version')
    finally:
        del sys.path[0]
    return mod

def setup_package():
    old_path = os.getcwd()
    path = get_path(__name__)
    os.chdir(path)
    sys.path.insert(0,path)

    try:
        config = map(get_package_config,bundle_packages)
        config_dict = merge_config_dicts(config)

        versions = map(get_package_version,bundle_packages)
        major = max([v.major for v in versions])
        minor = max([v.minor for v in versions])
        micro = max([v.micro for v in versions])
        release_level = min([v.release_level for v in versions])
        cvs_minor = reduce(lambda a,b:a+b,[v.cvs_minor for v in versions],0)
        cvs_serial = reduce(lambda a,b:a+b,[v.cvs_serial for v in versions],0)

        scipy_core_version = '%(major)d.%(minor)d.%(micro)d'\
                             '_%(release_level)s'\
                             '_%(cvs_minor)d.%(cvs_serial)d' % (locals ())

        print 'SciPy Core Version %s' % scipy_core_version
        from scipy_distutils.core import setup
        setup (name = "Scipy_core",
               version = scipy_core_version,
               maintainer = "SciPy Developers",
               maintainer_email = "scipy-dev@scipy.org",
               description = "SciPy core modules: scipy_{distutils,test,base}",
               license = "SciPy License (BSD Style)",
               url = "http://www.scipy.org",
               **config_dict
               )

    finally:
        del sys.path[0]
        os.chdir(old_path)

if __name__ == "__main__":
    setup_package()
