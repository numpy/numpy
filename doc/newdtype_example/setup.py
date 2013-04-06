from __future__ import division, print_function

from numpy.distutils.core import setup

def configuration(parent_package = '', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('floatint',parent_package,top_path)

    config.add_extension('floatint',
                         sources = ['floatint.c']);
    return config

setup(configuration=configuration)
