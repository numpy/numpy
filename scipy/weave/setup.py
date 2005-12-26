#!/usr/bin/env python
import os
def configuration(parent_package='',top_path=None):
    from scipy.distutils.misc_util import Configuration
    config = Configuration('weave',parent_package,top_path)
    config.add_data_dir('tests')
    config.add_data_dir('scxx')
    config.add_data_dir(os.path.join('blitz','blitz'))
    config.add_data_dir('doc')
    config.add_data_dir('examples')
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
          **configuration(top_path='').todict())
