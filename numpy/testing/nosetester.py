"""
Back compatibility nosetester module. It will import the appropriate
set of tools

"""
import os

if int(os.getenv('NPY_PYTEST', '0')):
    from .pytest_tools.nosetester import *
else:
    from .nose_tools.nosetester import *


__all__ = ['get_package_name', 'run_module_suite', 'NoseTester',
           '_numpy_tester', 'get_package_name', 'import_nose',
           'suppress_warnings']
