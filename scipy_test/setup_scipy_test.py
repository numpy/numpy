#!/usr/bin/env python
import os
from distutils.core import setup
from scipy_distutils.misc_util import get_path 
   
def install_package():
    """ Install the scipy_test module.  The dance with the current directory 
        is done to fool distutils into thinking it is run from the 
        scipy_distutils directory even if it was invoked from another script
        located in a different location.
    """
    path = get_path(__name__)
    old_path = os.getcwd()
    os.chdir(path)
    try:
        setup (name = "scipy_test",
               version = "0.1",
               description = "Supports testing of SciPy and other heirarchical packages",
               author = "Eric Jones",
               licence = "BSD Style",
               url = 'http://www.scipy.org',
               py_modules = ['scipy_test']
               )
    finally:
        os.chdir(old_path)
    
if __name__ == '__main__':
    install_package()
