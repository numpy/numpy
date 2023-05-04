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

This automatically put the checksum into README.rst, and writes the Changelog.

TODO
====
    - the script is messy, lots of global variables
    - make it more easily customizable (through command line args)
    - missing targets: install & test, sdist test, debian packaging
    - fix bdist_mpkg: we build the same source twice -> how to make sure we use
      the same underlying python for egg install in venv and for bdist_mpkg
"""
import os
import sys
import shutil
import hashlib
import textwrap

# The paver package needs to be installed to run tasks
import paver
from paver.easy import Bunch, options, task, sh


#-----------------------------------
# Things to be changed for a release
#-----------------------------------

# Path to the release notes
RELEASE_NOTES = 'doc/source/release/1.25.0-notes.rst'


#-------------------------------------------------------
# Hardcoded build/install dirs, virtualenv options, etc.
#-------------------------------------------------------

# Where to put the release installers
options(installers=Bunch(releasedir="release",
                         installersdir=os.path.join("release", "installers")),)


#------------------------
# Get the release version
#------------------------

sys.path.insert(0, os.path.dirname(__file__))
try:
    from setup import FULLVERSION
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
    root = f'numpy-{FULLVERSION}'
    if ftype == 'gztar':
        return root + '.tar.gz'
    elif ftype == 'zip':
        return root + '.zip'
    raise ValueError(f"Unknown type {type}")


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
    sh('python3 setup.py sdist --formats=gztar,zip')

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

    This appends file hashes to the release notes and creates
    four README files of the result in various formats:

    - README.rst
    - README.rst.gpg
    - README.md
    - README.md.gpg

    The md file are created using `pandoc` so that the links are
    properly updated. The gpg files are kept separate, so that
    the unsigned files may be edited before signing if needed.

    Parameters
    ----------
    options :
        Set by ``task`` decorator.
    filename : str
        Filename of the modified notes. The file is written
        in the release directory.

    """
    idirs = options.installers.installersdir
    notes = paver.path.path(RELEASE_NOTES)
    rst_readme = paver.path.path(filename + '.rst')
    md_readme = paver.path.path(filename + '.md')

    # append hashes
    with open(rst_readme, 'w') as freadme:
        with open(notes) as fnotes:
            freadme.write(fnotes.read())

        freadme.writelines(textwrap.dedent(
            """
            Checksums
            =========

            MD5
            ---
            ::

            """))
        freadme.writelines([f'    {c}\n' for c in compute_md5(idirs)])

        freadme.writelines(textwrap.dedent(
            """
            SHA256
            ------
            ::

            """))
        freadme.writelines([f'    {c}\n' for c in compute_sha256(idirs)])

    # generate md file using pandoc before signing
    sh(f"pandoc -s -o {md_readme} {rst_readme}")

    # Sign files
    if hasattr(options, 'gpg_key'):
        cmd = f'gpg --clearsign --armor --default_key {options.gpg_key}'
    else:
        cmd = 'gpg --clearsign --armor'

    sh(cmd + f' --output {rst_readme}.gpg {rst_readme}')
    sh(cmd + f' --output {md_readme}.gpg {md_readme}')


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
