
import os
import sys

def setup_package():

    from scipy.distutils.core import setup
    from scipy.distutils.misc_util import Configuration

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0,local_path)

    try:
        config = Configuration(
            maintainer = "SciPy Developers",
            maintainer_email = "scipy-dev@scipy.org",
            description = "Core SciPy",
            url = "http://numeric.scipy.org",
            license = 'BSD',
            )
        config.add_subpackage('scipy')

        from scipy.core_version import version
        config.name = 'scipy_core'
        config.dict_append(version=version)

        print config.name,'version',config.version

        setup( **config.todict() )
    finally:
        del sys.path[0]
        os.chdir(old_path)
    return

if __name__ == '__main__':
    setup_package()
