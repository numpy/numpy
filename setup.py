"""NumPy: array processing for numbers, strings, records, and objects.

NumPy is a general-purpose array-processing package designed to
efficiently manipulate large multi-dimensional arrays of arbitrary
records without sacrificing too much speed for small multi-dimensional
arrays.  NumPy is built on the Numeric code base and adds features
introduced by numarray as well as an extended C-API and the ability to
create arrays of arbitrary type.

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

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path,
        maintainer = "NumPy Developers",
        maintainer_email = "numpy-discussion@lists.sourceforge.net",
        description = DOCLINES[0],
        long_description = "\n".join(DOCLINES[2:]),
        url = "http://numeric.scipy.org",
        download_url = "http://sourceforge.net/projects/numpy",
        license = 'BSD',
        classifiers=filter(None, CLASSIFIERS.split('\n')),
        author = "Travis E. Oliphant, et.al.",
        author_email = "oliphant@ee.byu.edu",
        platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        )
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    
    config.add_subpackage('numpy')

    config.name = 'numpy' # used in generated file names
    
    from numpy.version import version
    config.dict_append(version=version)
    
    config.add_data_files(('numpy',['*.txt','COMPATIBILITY',
                                    'scipy_compatibility']))
    
    return config

def setup_package():

    from numpy.distutils.core import setup

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0,local_path)

    try:
        setup( configuration=configuration )
    finally:
        del sys.path[0]
        os.chdir(old_path)
    return

if __name__ == '__main__':
    setup_package()
