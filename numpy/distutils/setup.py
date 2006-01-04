#!/usr/bin/env python
from numpy.distutils.core      import setup
from numpy.distutils.misc_util import Configuration

def configuration(parent_package='',top_path=None):
    config = Configuration('distutils',parent_package,top_path)
    config.add_subpackage('command')
    config.add_subpackage('fcompiler')
    config.add_data_dir('tests')
    config.make_config_py()
    return config.todict()

if __name__ == '__main__':
    setup(**configuration(top_path=''))
