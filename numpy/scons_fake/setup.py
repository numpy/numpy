import os
import os.path

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('scons_fake',parent_package,top_path)

    config.add_subpackage('pyext')
    config.add_subpackage('ctypesext')
    config.add_subpackage('checklib')
    config.add_subpackage('checkers')
    config.add_subpackage('hook')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
