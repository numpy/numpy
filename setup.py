#!/usr/bin/env python
"""
Bundle of SciPy core modules:
  scipy_test
  scipy_distutils
  scipy_base
  weave

Usage:
   python setup.py install
   python setup.py sdist -f
"""

import os
import sys

from scipy_distutils.core import setup
from scipy_distutils.misc_util import default_config_dict
from scipy_distutils.misc_util import get_path, merge_config_dicts

bundle_packages = ['scipy_distutils','scipy_test','scipy_base','weave']

def setup_package():
    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0, local_path)

    try:
        configs = [{'name':'Scipy_core'}]
        versions = []
        for n in bundle_packages:
            sys.path.insert(0,os.path.join(local_path,n))
            try:
                mod = __import__('setup_'+n)
                configs.append(mod.configuration(parent_path=local_path))
                mod = __import__(n+'_version')
                versions.append(mod)
            finally:
                del sys.path[0]
   
        config_dict = merge_config_dicts(configs)

        major = max([v.major for v in versions])
        minor = max([v.minor for v in versions])
        micro = max([v.micro for v in versions])
        release_level = min([v.release_level for v in versions])
        release_level = ''
        cvs_minor = reduce(lambda a,b:a+b,[v.cvs_minor for v in versions],0)
        cvs_serial = reduce(lambda a,b:a+b,[v.cvs_serial for v in versions],0)

        if cvs_minor or cvs_serial:
            if release_level:
                scipy_core_version = '%(major)d.%(minor)d.%(micro)d'\
                                     '_%(release_level)s'\
                                     '_%(cvs_minor)d.%(cvs_serial)d' % (locals ())
            else:
                scipy_core_version = '%(major)d.%(minor)d.%(micro)d'\
                                     '_%(cvs_minor)d.%(cvs_serial)d' % (locals ())
        else:
            if release_level:
                scipy_core_version = '%(major)d.%(minor)d.%(micro)d'\
                                     '_%(release_level)s'\
                                     % (locals ())
            else:
                scipy_core_version = '%(major)d.%(minor)d.%(micro)d'\
                                     % (locals ())

        print 'SciPy Core Version %s' % scipy_core_version
        setup( version = scipy_core_version,
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
