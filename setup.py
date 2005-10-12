
import os
import sys

def setup_package():

    from scipy.distutils.core import setup
    from scipy.distutils.misc_util import Configuration
    from scipy.core_version import version

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0,local_path)

    try:
        config = Configuration(
            version = version,
            maintainer = "SciPy Developers",
            maintainer_email = "scipy-dev@scipy.org",
            description = "SciPy Core",
            url = "http://numeric.scipy.org",
	    license = 'BSD',
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
