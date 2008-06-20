"""Masked arrays add-ons.

A collection of utilities for maskedarray

:author: Pierre GF Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: __init__.py 3473 2007-10-29 15:18:13Z jarrod.millman $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: jarrod.millman $)"
__version__ = '1.0'
__revision__ = "$Revision: 3473 $"
__date__     = '$Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $'

import core
from core import *

import extras
from extras import *

__all__ = ['core', 'extras']
__all__ += core.__all__
__all__ += extras.__all__

from numpy.testing.pkgtester import Tester
test = Tester().test
bench = Tester().bench
