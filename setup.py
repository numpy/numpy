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

import os
import sys

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

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('numpy')

    config.add_data_files(('numpy','*.txt'),
                          ('numpy','COMPATIBILITY'),
                          ('numpy','scipy_compatibility'),
                          ('numpy','site.cfg.example'))

    config.get_version('numpy/version.py') # sets config.version

    return config

def setup_package():

    from numpy.distutils.core import setup

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0,local_path)

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
