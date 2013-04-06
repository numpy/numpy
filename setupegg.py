#!/usr/bin/env python
"""
A setup.py script to use setuptools, which gives egg goodness, etc.

This is used to build installers for OS X through bdist_mpkg.

Notes
-----
Using ``python setupegg.py install`` directly results in file permissions being
set wrong, with nose refusing to run any tests. To run the tests anyway, use::

  >>> np.test(extra_argv=['--exe'])

"""
from __future__ import division, absolute_import, print_function

import sys
from setuptools import setup

if sys.version_info[0] >= 3:
    import imp
    setupfile = imp.load_source('setupfile', 'setup.py')
    setupfile.setup_package()
else:
    exec(compile(open('setup.py').read(), 'setup.py', 'exec'))
