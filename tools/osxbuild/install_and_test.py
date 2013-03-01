#!/usr/bin/env python
"""Install the built package and run the tests."""

import os

# FIXME: Should handle relative import better!
#from .build import DIST_DIR
from build import SRC_DIR, DIST_DIR, shellcmd

clrgreen = '\033[0;32m'
clrnull = '\033[0m'
# print '\033[0;32m foobar \033[0m'
def color_print(msg):
    """Add color to this print output."""
    clrmsg = clrgreen + msg + clrnull
    print clrmsg

distdir = os.path.join(SRC_DIR, DIST_DIR)

# Find the package and build abspath to it
pkg = None
filelist = os.listdir(distdir)
for fn in filelist:
    if fn.endswith('mpkg'):
        pkg = fn
        break
if pkg is None:
    raise IOError('Package is not found in directory %s' % distdir)

pkgpath = os.path.abspath(os.path.join(SRC_DIR, DIST_DIR, pkg))
color_print('Installing package: %s' % pkgpath)

# Run the installer
print
color_print('Installer requires admin rights, you will be prompted for sudo')
print
cmd = 'sudo installer -verbose -package %s -target /' % pkgpath
#color_print(cmd)
shellcmd(cmd)

# Null out the PYTHONPATH so we're sure to test the Installed version of numpy
os.environ['PYTHONPATH'] = '0'

print
color_print('Install successful!')
color_print('Running numpy test suite!')
print
import numpy
numpy.test()
