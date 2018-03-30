"""
Back compatibility decorators module. It will import the appropriate
set of tools

"""
import warnings

warnings.warn("Import from numpy.testing, not numpy.testing.decorators",
              ImportWarning)

from ._private.decorators import *
