"""
Back compatibility nosetester module. It will import the appropriate
set of tools

"""
import warnings

warnings.warn("Import from numpy.testing, not numpy.testing.nosetester",
              ImportWarning)

from ._private.nosetester import *

__all__ = ['get_package_name', 'run_module_suite', 'NoseTester',
           '_numpy_tester', 'get_package_name', 'import_nose',
           'suppress_warnings']
