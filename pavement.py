r"""
This paver file is intended to help with the release process as much as
possible. It relies on virtualenv to generate 'bootstrap' environments as
independent from the user system as possible (e.g. to make sure the sphinx doc
is built against the built numpy, not an installed one).

Building changelog + notes
==========================

Assumes you have git and the binaries/tarballs in installers/::

    paver write_release
    paver write_note

This automatically put the checksum into README.rst, and write the Changelog
which can be uploaded to sourceforge.

TODO
====
    - the script is messy, lots of global variables
    - make it more easily customizable (through command line args)
    - missing targets: install & test, sdist test, debian packaging
    - fix bdist_mpkg: we build the same source twice -> how to make sure we use
      the same underlying python for egg install in venv and for bdist_mpkg
"""
from __future__ import division, print_function

import os
import sys
import shutil
import subprocess
import re
import hashlib

# The paver package needs to be installed to run tasks
import paver
from paver.easy import Bunch, options, task, sh


#-----------------------------------
# Things to be changed for a release
#-----------------------------------

# Path to the release notes
RELEASE_NOTES = 'doc/release/1.17.0-notes.rst'


#-------------------------------------------------------
# Hardcoded build/install dirs, virtualenv options, etc.
#-------------------------------------------------------

# Where to put the release installers
options(installers=Bunch(releasedir="release",
                         installersdir=os.path.join("release", "installers")),)


#-----------------------------
# Generate the release version
#-----------------------------

sys.path.insert(0, os.path.dirname(__file__))
try:
    setup_py = __import__("setup")
    FULLVERSION = setup_py.VERSION
    # This is duplicated from setup.py
    if os.path.exists('.git'):
        GIT_REVISION = setup_py.git_version()
    elif os.path.exists('numpy/version.py'):
        # must be a source distribution, use existing version file
        from numpy.version import git_revision as GIT_REVISION
    else:
        GIT_REVISION = "Unknown"

    if not setup_py.ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]
finally:
    sys.path.pop(0)


#--------------------------
# Source distribution stuff
#--------------------------
def tarball_name(ftype='gztar'):
    """Generate source distribution name

    Parameters
    ----------
    ftype : {'zip', 'gztar'}
        Type of archive, default is 'gztar'.

    """
    root = 'numpy-%s' % FULLVERSION
    if ftype == 'gztar':
        return root + '.tar.gz'
    elif ftype == 'zip':
        return root + '.zip'
    raise ValueError("Unknown type %s" % type)

@task
def sdist(options):
    """Make source distributions.

    Parameters
    ----------
    options :
        Set by ``task`` decorator.

    """
    # First clean the repo and update submodules (for up-to-date doc html theme
    # and Sphinx extensions)
    sh('git clean -xdf')
    sh('git submodule init')
    sh('git submodule update')

    # To be sure to bypass paver when building sdist... paver + numpy.distutils
    # do not play well together.
    # Cython is run over all Cython files in setup.py, so generated C files
    # will be included.
    sh('python setup.py sdist --formats=gztar,zip')

    # Copy the superpack into installers dir
    idirs = options.installers.installersdir
    if not os.path.exists(idirs):
        os.makedirs(idirs)

    for ftype in ['gztar', 'zip']:
        source = os.path.join('dist', tarball_name(ftype))
        target = os.path.join(idirs, tarball_name(ftype))
        shutil.copy(source, target)


#-------------
# README stuff
#-------------

def _compute_hash(idirs, hashfunc):
    """Hash files using given hashfunc.

    Parameters
    ----------
    idirs : directory path
        Directory containing files to be hashed.
    hashfunc : hash function
        Function to be used to hash the files.

    """
    released = paver.path.path(idirs).listdir()
    checksums = []
    for fpath in sorted(released):
        with open(fpath, 'rb') as fin:
            fhash = hashfunc(fin.read())
            checksums.append(
                '%s  %s' % (fhash.hexdigest(), os.path.basename(fpath)))
    return checksums


def compute_md5(idirs):
    """Compute md5 hash of files in idirs.

    Parameters
    ----------
    idirs : directory path
        Directory containing files to be hashed.

    """
    return _compute_hash(idirs, hashlib.md5)


def compute_sha256(idirs):
    """Compute sha256 hash of files in idirs.

    Parameters
    ----------
    idirs : directory path
        Directory containing files to be hashed.

    """
    # better checksum so gpg signed README.rst containing the sums can be used
    # to verify the binaries instead of signing all binaries
    return _compute_hash(idirs, hashlib.sha256)


def write_release_task(options, filename='README'):
    """Append hashes of release files to release notes.

    Parameters
    ----------
    options :
        Set by ``task`` decorator.
    filename : string
        Filename of the modified notes. The file is written
        in the release directory.

    """
    idirs = options.installers.installersdir
    source = paver.path.path(RELEASE_NOTES)
    target = paver.path.path(filename + '.rst')
    if target.exists():
        target.remove()

    tmp_target = paver.path.path(filename + '.md')
    source.copy(tmp_target)

    with open(str(tmp_target), 'a') as ftarget:
        ftarget.writelines("""
Checksums
=========

MD5
---

""")
        ftarget.writelines(['    %s\n' % c for c in compute_md5(idirs)])
        ftarget.writelines("""
SHA256
------

""")
        ftarget.writelines(['    %s\n' % c for c in compute_sha256(idirs)])

    # Sign release
    cmd = ['gpg', '--clearsign', '--armor']
    if hasattr(options, 'gpg_key'):
        cmd += ['--default-key', options.gpg_key]
    cmd += ['--output', str(target), str(tmp_target)]
    subprocess.check_call(cmd)
    print("signed %s" % (target,))

    # Change PR links for github posting, don't sign this
    # as the signing isn't markdown compatible.
    with open(str(tmp_target), 'r') as ftarget:
        mdtext = ftarget.read()
        mdtext = re.sub(r'^\* `(\#[0-9]*).*?`__', r'* \1', mdtext, flags=re.M)
    with open(str(tmp_target), 'w') as ftarget:
        ftarget.write(mdtext)


@task
def write_release(options):
    """Write the README files.

    Two README files are generated from the release notes, one in ``rst``
    markup for the general release, the other in ``md`` markup for the github
    release notes.

    Parameters
    ----------
    options :
        Set by ``task`` decorator.

    """
    rdir = options.installers.releasedir
    write_release_task(options, os.path.join(rdir, 'README'))
