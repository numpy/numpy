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

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
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

NAME                = 'numpy'
MAINTAINER          = "NumPy Developers"
MAINTAINER_EMAIL    = "numpy-discussion@scipy.org"
DESCRIPTION         = DOCLINES[0]
LONG_DESCRIPTION    = "\n".join(DOCLINES[2:])
URL                 = "http://numpy.scipy.org"
DOWNLOAD_URL        = "http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103"
LICENSE             = 'BSD'
CLASSIFIERS         = filter(None, CLASSIFIERS.split('\n'))
AUTHOR              = "Travis E. Oliphant, et.al."
AUTHOR_EMAIL        = "oliphant@enthought.com"
PLATFORMS           = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"]
MAJOR               = 1
MINOR               = 3
MICRO               = 0
ISRELEASED          = False
VERSION             = '%d.%d.%db1' % (MAJOR, MINOR, MICRO)
    
FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'
    # If in git or something, bypass the svn rev
    if os.path.exists('.svn.'):
        FULLVERSION += svn_version()

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

# This is a bit hackish: we are setting a global variable so that the main
# numpy __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.  While ugly, it's
# a lot more robust than what was previously being used.
__builtin__.__NUMPY_SETUP__ = True

# Return the svn version as a string, raise a ValueError otherwise
def svn_version():
    out = subprocess.Popen(['svn', 'info'], stdout = subprocess.PIPE).communicate()[0]
    r = re.compile('Revision: ([0-9]+)')
    svnver = None
    for line in out.split('\n'):
        m = r.match(line)
        if m:
            svnver = m.group(1)

    if not svnver:
        raise ValueError("Error while parsing svn version ?")
    return svnver

def write_version_py(filename='numpy/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM NUMPY SETUP.PY
short_version='%(version)s'
version='%(version)s'
release=%(isrelease)s

if not release:
    version += '.dev'
    import os
    svn_version_file = os.path.join(os.path.dirname(__file__),
                                   'core','__svn_version__.py')
    if os.path.isfile(svn_version_file):
        import imp
        svn = imp.load_module('numpy.core.__svn_version__',
                              open(svn_version_file),
                              svn_version_file,
                              ('.py','U',1))
        version += svn.version
"""
    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION, 'isrelease': str(ISRELEASED)})
    finally:
        a.close()

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
                          ('numpy','site.cfg.example'))

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
            name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            configuration=configuration )
    finally:
        del sys.path[0]
        os.chdir(old_path)
    return

if __name__ == '__main__':
    setup_package()
