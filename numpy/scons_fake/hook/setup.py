import os
import os.path

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('hook',parent_package,top_path)

    def foo():
        print "foo"
    config.add_sconscript('SConstruct', post_hook = foo)
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
