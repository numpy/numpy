#!/usr/bin/env python
import os

from distutils.core import setup
from misc_util import get_path, update_version
from scipy_distutils_version import scipy_distutils_version

def install_package():
    """ Install the scipy_distutils.  The dance with the current directory is done
        to fool distutils into thinking it is run from the scipy_distutils directory
        even if it was invoked from another script located in a different location.
    """
    path = get_path(__name__)
    old_path = os.getcwd()
    os.chdir(path)
    try:
        print 'scipy_distutils Version',scipy_distutils_version
        setup (name = "scipy_distutils",
               version = scipy_distutils_version,
               description = "Changes to distutils needed for SciPy -- mostly Fortran support",
               author = "Travis Oliphant, Eric Jones, and Pearu Peterson",
               author_email = "scipy-dev@scipy.org",
               license = "SciPy License (BSD Style)",
               url = 'http://www.scipy.org',
               packages = ['scipy_distutils','scipy_distutils.command'],
               package_dir = {'scipy_distutils':path}
               )
    finally:
        os.chdir(old_path)
    
if __name__ == '__main__':
    install_package()
