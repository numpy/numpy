"""
Back compatibility nosetester module. It will import the appropriate
set of tools

"""
import os

from ._private.nosetester import *


__all__ = ['get_package_name', 'run_module_suite', 'NoseTester',
           '_numpy_tester', 'get_package_name', 'import_nose',
           'suppress_warnings']
