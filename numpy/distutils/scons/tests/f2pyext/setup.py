import os
import os.path

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('f2pyext',parent_package,top_path)

    config.add_sconscript('SConstruct', source_files = ['spam.pyf.c'])
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
