#!/usr/bin/env python

import os
from glob import glob
from scipy.distutils.misc_util import get_path, Configuration, dot_join

def configuration(parent_package='',parent_path=None):
    parent_path2 = parent_path
    parent_path = parent_package
    local_path = get_path(__name__,parent_path2)
    config = Configuration('weave',parent_package)
    test_path = os.path.join(local_path,'tests')
    config.add_subpackage(dot_join(parent_package, 'tests'),test_path)
    scxx_files = glob(os.path.join(local_path,'scxx','*.*'))
    install_path = os.path.join(parent_path,'weave','scxx')
    config.add_data_dir(os.path.join(local_path,'scxx'))
    config.add_data_dir(os.path.join(local_path,'blitz','blitz'))
    config.add_data_dir(os.path.join(local_path,'blitz','blitz','array'))
    config.add_data_dir(os.path.join(local_path,'blitz','blitz','meta'))
    config.add_data_files(*glob(os.path.join(local_path,'doc','*.html')))
    config.add_data_files(*glob(os.path.join(local_path,'examples','*.py')))    
    return config

if __name__ == '__main__':    
    from scipy.distutils.core import setup
    from weave_version import weave_version
    setup(version = weave_version,
          description = "Tools for inlining C/C++ in Python",
          author = "Eric Jones",
          author_email = "eric@enthought.com",
          licence = "SciPy License (BSD Style)",
          url = 'http://www.scipy.org',
          **configuration(parent_path=''))
