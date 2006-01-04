"""Scipy: array processing for numbers, strings, records, and objects.

Scipy is an array processing package designed to efficiently manipulate
large multi-dimensional arrays without sacrificing too much speed for small
multi-dimensional arrays.  Scipy is built on the Numeric code base and
adds features introduced by numarray as well as an extended C-API and the
ability to create arrays of arbitrary type.

There are also basic facilities for discrete fourier transform,
linear algebra and random numbers.
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

def setup_package():

    from scipy.distutils.core import setup
    from scipy.distutils.misc_util import Configuration

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0,local_path)

    try:
        config = Configuration(
            maintainer = "SciPy Developers",
            maintainer_email = "scipy-dev@scipy.org",
            description = DOCLINES[0],
            long_description = "\n".join(DOCLINES[2:]),
            url = "http://numeric.scipy.org",
            download_url = "http://sourceforge.net/projects/numpy",
            license = 'BSD',
            classifiers=filter(None, CLASSIFIERS.split('\n')),
            author = "Travis E. Oliphant",
            author_email = "oliphant.travis@ieee.org",
            platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
            )
        config.add_subpackage('scipy')

        from scipy.core_version import version
        config.name = 'scipy'
        config.dict_append(version=version)

        print config.name,'version',config.version

        setup( **config.todict() )
    finally:
        del sys.path[0]
        os.chdir(old_path)
    return

if __name__ == '__main__':
    setup_package()
