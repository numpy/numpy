#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('numpy',parent_package,top_path)
    config.add_subpackage('distutils')
    config.add_subpackage('testing')
    config.add_subpackage('f2py')
    config.add_subpackage('core')
    config.add_subpackage('lib')
    config.add_subpackage('dft')
    config.add_subpackage('linalg')
    config.add_subpackage('random')
    config.add_data_dir('doc')
    config.make_config_py() # installs __config__.py
    return config

if __name__ == '__main__':
    # Remove current working directory from sys.path
    # to avoid importing numpy.distutils as Python std. distutils:
    import os, sys
    sys.path.remove(os.getcwd())

    from numpy.distutils.core import setup
    setup(configuration=configuration)
