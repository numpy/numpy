#!/usr/bin/env python
import os,sys
from scipy_distutils.core import setup
from scipy_distutils.misc_util import get_path, merge_config_dicts
from scipy_distutils.misc_util import package_config

# Enough changes to bump the number.  We need a global method for
# versioning
version = "0.3.0"
   
def stand_alone_package(with_dependencies = 0):
    path = get_path(__name__)
    old_path = os.getcwd()
    os.chdir(path)
    try:
        primary =     ['weave']
        if with_dependencies:
            dependencies= ['scipy_distutils','scipy_test','scipy_base']       
        else:
            dependencies = []    
        
        print 'dep:', dependencies
        config_dict = package_config(primary,dependencies)
        config_dict['name'] = 'weave'
        setup (version = version,
               description = "Tools for inlining C/C++ in Python",
               author = "Eric Jones",
               author_email = "eric@enthought.com",
               licence = "SciPy License (BSD Style)",
               url = 'http://www.scipy.org',
               **config_dict
               )        
    finally:
        os.chdir(old_path)

if __name__ == '__main__':
    import sys
    if '--without-dependencies' in sys.argv:
        with_dependencies = 0
        sys.argv.remove('--without-dependencies')
    else:
        with_dependencies = 1    
    stand_alone_package(with_dependencies)
    
