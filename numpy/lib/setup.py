import imp
import os
from os.path import join
from glob import glob
from distutils.dep_util import newer,newer_group

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration,dot_join
    from numpy.distutils.system_info import get_info

    config = Configuration('lib',parent_package,top_path)
    local_dir = config.local_path

    config.add_include_dirs(join('..','core','include'))


    config.add_extension('_compiled_base',
                         sources=[join('src','_compiled_base.c')]
                         )

    config.add_data_dir('tests')

    return config

if __name__=='__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
