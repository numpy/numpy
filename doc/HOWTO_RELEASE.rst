These instructions give an overview of what is necessary to build binary
releases for NumPy.

Current build and release info
==============================

Useful info can be found in the following locations:

* **Source tree**

  - `INSTALL.rst <https://github.com/numpy/numpy/blob/main/INSTALL.rst>`_
  - `pavement.py <https://github.com/numpy/numpy/blob/main/pavement.py>`_

* **NumPy docs**

  - `HOWTO_RELEASE.rst <https://github.com/numpy/numpy/blob/main/doc/HOWTO_RELEASE.rst>`_
  - `RELEASE_WALKTHROUGH.rst <https://github.com/numpy/numpy/blob/main/doc/RELEASE_WALKTHROUGH.rst>`_
  - `BRANCH_WALKTHROUGH.rst <https://github.com/numpy/numpy/blob/main/doc/BRANCH_WALKTHROUGH.rst>`_

Supported platforms and versions
================================
:ref:`NEP 29 <NEP29>` outlines which Python versions
are supported; For the first half of 2020, this will be Python >= 3.6. We test
NumPy against all these versions every time we merge code to main.  Binary
installers may be available for a subset of these versions (see below).

* **OS X**

  OS X versions >= 10.9 are supported, for Python version support see
  :ref:`NEP 29 <NEP29>`. We build binary wheels for OSX that are compatible with
  Python.org Python, system Python, homebrew and macports - see this
  `OSX wheel building summary <https://github.com/MacPython/wiki/wiki/Spinning-wheels>`_
  for details.

* **Windows**

  We build 32- and 64-bit wheels on Windows. Windows 7, 8 and 10 are supported.
  We build NumPy using the `mingw-w64 toolchain`_, `cibuildwheels`_ and GitHub
  actions.

.. _cibuildwheels: https://cibuildwheel.readthedocs.io/en/stable/

* **Linux**

  We build and ship `manylinux2014 <https://www.python.org/dev/peps/pep-0513>`_
  wheels for NumPy.  Many Linux distributions include their own binary builds
  of NumPy.

* **BSD / Solaris**

  No binaries are provided, but successful builds on Solaris and BSD have been
  reported.

Tool chain
==========
We build all our wheels on cloud infrastructure - so this list of compilers is
for information and debugging builds locally.  See the ``.travis.yml`` script
in the `numpy wheels`_ repo for an outdated source of the build recipes using
multibuild.

.. _numpy wheels : https://github.com/MacPython/numpy-wheels

Compilers
---------
The same gcc version is used as the one with which Python itself is built on
each platform. At the moment this means:

- OS X builds on travis currently use `clang`.  It appears that binary wheels
  for OSX >= 10.6 can be safely built from the travis-ci OSX 10.9 VMs
  when building against the Python from the Python.org installers;
- Windows builds use the `mingw-w64 toolchain`_;
- Manylinux2014 wheels use the gcc provided on the Manylinux docker images.

You will need Cython for building the binaries.  Cython compiles the ``.pyx``
files in the NumPy distribution to ``.c`` files.

.. _mingw-w64 toolchain : https://mingwpy.github.io

OpenBLAS
--------

All the wheels link to a version of OpenBLAS_ supplied via the openblas-libs_ repo.
The shared object (or DLL) is shipped with in the wheel, renamed to prevent name
collisions with other OpenBLAS shared objects that may exist in the filesystem.

.. _OpenBLAS: https://github.com/xianyi/OpenBLAS
.. _openblas-libs: https://github.com/MacPython/openblas-libs


Building source archives and wheels
-----------------------------------
The NumPy wheels and sdist are now built using cibuildwheel with
github actions.


Building docs
-------------
We are no longer building ``PDF`` files. All that will be needed is

- virtualenv (pip).

The other requirements will be filled automatically during the documentation
build process.


Uploading to PyPI
-----------------
The only application needed for uploading is

- twine (pip).

You will also need a PyPI token, which is best kept on a keyring. See the
twine keyring_  documentation for how to do that.

.. _keyring: https://twine.readthedocs.io/en/stable/#keyring-support


Generating author/PR lists
--------------------------
You will need a personal access token
`<https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line/>`_
so that scripts can access the github NumPy repository.

- gitpython (pip)
- pygithub (pip)


What is released
================

* **Wheels**
  We currently support Python 3.8-3.10 on Windows, OSX, and Linux.

  * Windows: 32-bit and 64-bit wheels built using Github actions;
  * OSX: x64_86 and arm64 OSX wheels built using Github actions;
  * Linux: x64_86 and aarch64 Manylinux2014 wheels built using Github actions.

* **Other**
  Release notes and changelog

* **Source distribution**
  We build source releases in the .tar.gz format.


Release process
===============

Agree on a release schedule
---------------------------
A typical release schedule is one beta, two release candidates and a final
release.  It's best to discuss the timing on the mailing list first, in order
for people to get their commits in on time, get doc wiki edits merged, etc.
After a date is set, create a new maintenance/x.y.z branch, add new empty
release notes for the next version in the main branch and update the Trac
Milestones.


Make sure current branch builds a package correctly
---------------------------------------------------
The CI builds wheels when a PR header begins with ``REL``. Your last
PR before releasing should be so marked and all the tests should pass.
You can also do::

    git clean -fxdq
    python setup.py bdist_wheel
    python setup.py sdist

For details of the build process itself, it is best to read the
Step-by-Step Directions below.

.. note:: The following steps are repeated for the beta(s), release
   candidates(s) and the final release.


Check deprecations
------------------
Before :ref:`the release branch is made <branching>`, it should be checked that
all deprecated code that should be removed is actually removed, and all new
deprecations say in the docstring or deprecation warning what version the code
will be removed.

Check the C API version number
------------------------------
The C API version needs to be tracked in three places

- numpy/_core/meson.build
- numpy/_core/code_generators/cversions.txt
- numpy/_core/include/numpy/numpyconfig.h

There are three steps to the process.

1. If the API has changed, increment the C_API_VERSION in
   numpy/core/meson.build. The API is unchanged only if any code compiled
   against the current API will be backward compatible with the last released
   NumPy version. Any changes to C structures or additions to the public
   interface will make the new API not backward compatible.

2. If the C_API_VERSION in the first step has changed, or if the hash of
   the API has changed, the cversions.txt file needs to be updated. To check
   the hash, run the script numpy/_core/cversions.py and note the API hash that
   is printed. If that hash does not match the last hash in
   numpy/_core/code_generators/cversions.txt the hash has changed. Using both
   the appropriate C_API_VERSION and hash, add a new entry to cversions.txt.
   If the API version was not changed, but the hash differs, you will need to
   comment out the previous entry for that API version. For instance, in NumPy
   1.9 annotations were added, which changed the hash, but the API was the
   same as in 1.8. The hash serves as a check for API changes, but it is not
   definitive.

   If steps 1 and 2 are done correctly, compiling the release should not give
   a warning "API mismatch detect at the beginning of the build".

3. The numpy/_core/include/numpy/numpyconfig.h will need a new
   NPY_X_Y_API_VERSION macro, where X and Y are the major and minor version
   numbers of the release. The value given to that macro only needs to be
   increased from the previous version if some of the functions or macros in
   the include files were deprecated.

The C ABI version number in numpy/_core/meson.build should only be updated
for a major release.


Check the release notes
-----------------------
Use `towncrier`_ to build the release note and
commit the changes. This will remove all the fragments from
``doc/release/upcoming_changes`` and add ``doc/release/<version>-note.rst``.::

    towncrier build --version "<version>"
    git commit -m"Create release note"

Check that the release notes are up-to-date.

Update the release notes with a Highlights section. Mention some of the
following:

- major new features
- deprecated and removed features
- supported Python versions
- for SciPy, supported NumPy version(s)
- outlook for the near future

.. _towncrier: https://pypi.org/project/towncrier/
