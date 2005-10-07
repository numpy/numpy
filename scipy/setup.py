#!/usr/bin/env python
import os

def configuration(parent_package='',top_path=None):
    from scipy.distutils.misc_util import Configuration
    config = Configuration('scipy',parent_package,top_path)
    config.add_subpackage('distutils')
    config.add_subpackage('weave')
    config.add_subpackage('test')
    config.add_subpackage('f2py2e') # installed as scipy.f2py
    config.add_subpackage('base')
    config.add_subpackage('corelib') # installed as scipy.lib
    config.add_subpackage('fftpack')
    config.add_subpackage('linalg')
    config.make_svn_version_py() # installs __svn_version__.py
    return config.todict()

if __name__ == '__main__':
    # Remove current working directory from sys.path
    # to avoid importing scipy.distutils as Python std. distutils:
    import os, sys
    sys.path.remove(os.getcwd())

    from scipy.distutils.core import setup
    setup(**configuration(top_path=''))
