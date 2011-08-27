#!/usr/bin/env python
"""NumPy: array processing for numbers, strings, records, and objects.

NumPy is a general-purpose array-processing package designed to
efficiently manipulate large multi-dimensional arrays of arbitrary
records without sacrificing too much speed for small multi-dimensional
arrays.  NumPy is built on the Numeric code base and adds features
introduced by numarray as well as an extended C-API and the ability to
create arrays of arbitrary type which also makes NumPy suitable for
interfacing with general-purpose data-base applications.

There are also basic facilities for discrete fourier transform,
basic linear algebra and random number generation.
"""

DOCLINES = __doc__.split("\n")

import __builtin__
import os
import sys
import warnings

warnings.warn("""\
Numscons scripts are being deprecated - most of numscons features should now be
supported by bento""")

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: C
Programming Language :: Python
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

sys.path.insert(0, os.path.dirname(__file__))
try:
    setup_py = __import__("setup")
    write_version_py = setup_py.write_version_py
finally:
    sys.path.pop(0)

# This is a bit hackish: we are setting a global variable so that the main
# numpy __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.  While ugly, it's
# a lot more robust than what was previously being used.
__builtin__.__NUMPY_SETUP__ = True

# DO NOT REMOVE numpy.distutils IMPORT ! This is necessary for numpy.distutils'
# monkey patching to work.
import numpy.distutils
from distutils.errors import DistutilsError
try:
    import numscons
except ImportError, e:
    msg = ["You cannot build numpy with scons without the numscons package "]
    msg.append("(Failure was: %s)" % e)
    raise DistutilsError('\n'.join(msg))

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path, setup_name = 'setupscons.py')
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('numpy')

    config.add_data_files(('numpy','*.txt'),
                          ('numpy','COMPATIBILITY'),
                          ('numpy','site.cfg.example'),
                          ('numpy','setup.py'))

    config.get_version('numpy/version.py') # sets config.version

    return config

def setup_package():

    from numpy.distutils.core import setup

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0,local_path)

    # Rewrite the version file everytime
    if os.path.exists('numpy/version.py'): os.remove('numpy/version.py')
    write_version_py()

    try:
        setup(
            name = 'numpy',
            maintainer = "NumPy Developers",
            maintainer_email = "numpy-discussion@lists.sourceforge.net",
            description = DOCLINES[0],
            long_description = "\n".join(DOCLINES[2:]),
            url = "http://numeric.scipy.org",
            download_url = "http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103",
            license = 'BSD',
            classifiers=filter(None, CLASSIFIERS.split('\n')),
            author = "Travis E. Oliphant, et.al.",
            author_email = "oliphant@ee.byu.edu",
            platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
            configuration=configuration )
    finally:
        del sys.path[0]
        os.chdir(old_path)
    return

if __name__ == '__main__':
    setup_package()
