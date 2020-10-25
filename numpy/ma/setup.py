#!/usr/bin/env python3
def configuration(parent_package='',top_path=None):
    from numpy_distutils.misc_util import Configuration
    config = Configuration('ma', parent_package, top_path)
    config.add_subpackage('tests')
    config.add_data_files('*.pyi')
    return config

if __name__ == "__main__":
    from numpy_distutils.core import setup
    config = configuration(top_path='').todict()
    setup(**config)
