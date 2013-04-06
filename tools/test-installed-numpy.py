#!/usr/bin/env python
from __future__ import division, absolute_import, print_function

# A simple script to test the installed version of numpy by calling
# 'numpy.test()'. Key features:
#   -- convenient command-line syntax
#   -- sets exit status appropriately, useful for automated test environments

# It would be better to set this up as a module in the numpy namespace, so
# that it could be run as:
#   python -m numpy.run_tests <args>
# But, python2.4's -m switch only works with top-level modules, not modules
# that are inside packages. So, once we drop 2.4 support, maybe...

import sys, os
# In case we are run from the source directory, we don't want to import numpy
# from there, we want to import the installed version:
sys.path.pop(0)

from optparse import OptionParser
parser = OptionParser("usage: %prog [options] -- [nosetests options]")
parser.add_option("-v", "--verbose",
                  action="count", dest="verbose", default=1,
                  help="increase verbosity")
parser.add_option("--doctests",
                  action="store_true", dest="doctests", default=False,
                  help="Run doctests in module")
parser.add_option("--coverage",
                  action="store_true", dest="coverage", default=False,
                  help="report coverage of NumPy code (requires 'coverage' module")
parser.add_option("-m", "--mode",
                  action="store", dest="mode", default="fast",
                  help="'fast', 'full', or something that could be "
                       "passed to nosetests -A [default: %default]")
(options, args) = parser.parse_args()

import numpy

# Check that NPY_RELAXED_STRIDES_CHECKING is active when set.
# The same flags check is also used in the tests to switch behavior.
if (os.environ.get('NPY_RELAXED_STRIDES_CHECKING', "0") != "0"):
    if not numpy.ones((10,1), order='C').flags.f_contiguous:
        print('NPY_RELAXED_STRIDES_CHECKING set, but not active.')
        sys.exit(1)
elif numpy.ones((10,1), order='C').flags.f_contiguous:
    print('NPY_RELAXED_STRIDES_CHECKING not set, but active.')
    sys.exit(1)

result = numpy.test(options.mode,
                    verbose=options.verbose,
                    extra_argv=args,
                    doctests=options.doctests,
                    coverage=options.coverage)

if result.wasSuccessful():
    sys.exit(0)
else:
    sys.exit(1)
