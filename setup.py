
import os
import sys

def setup_package():

    from distutils.core import setup
    from distutils.misc_util import Configuration
    from base.core_version import version

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0,local_path)

    try:
        config = Configuration(
            version = version,
            maintainer = "Travis Oliphant",
            maintainer_email = "oliphant.travis@ieee.org",
            description = "Core SciPy",
            url = "http://numpy.sourceforge.net",
            )
        config.add_subpackage('scipy')
        config.name = 'scipy_core'
        print config.name,'version',config.version
        setup( **config.todict() )
    finally:
        del sys.path[0]
        os.chdir(old_path)
    return

if __name__ == '__main__':
    setup_package()
