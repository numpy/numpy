.. _numpy-distutils-refguide:

*********
Packaging
*********

.. module:: numpy.distutils

.. warning::

   ``numpy.distutils`` is deprecated, and will be removed for
   Python >= 3.12. For more details, see :ref:`distutils-status-migration`

.. warning::

   Note that ``setuptools`` does major releases often and those may contain
   changes that break :mod:`numpy.distutils`, which will *not* be updated anymore
   for new ``setuptools`` versions. It is therefore recommended to set an
   upper version bound in your build configuration for the last known version
   of ``setuptools`` that works with your build.

NumPy provides enhanced distutils functionality to make it easier to
build and install sub-packages, auto-generate code, and extension
modules that use Fortran-compiled libraries. A useful :class:`Configuration
<misc_util.Configuration>` class is also provided in
:mod:`numpy.distutils.misc_util` that can make it easier to construct
keyword arguments to pass to the setup function (by passing the
dictionary obtained from the todict() method of the class). More
information is available in the :ref:`distutils-user-guide`.

The choice and location of linked libraries such as BLAS and LAPACK as well as
include paths and other such build options can be specified in a ``site.cfg``
file located in the NumPy root repository or a ``.numpy-site.cfg`` file in your
home directory. See the ``site.cfg.example`` example file included in the NumPy
repository or sdist for documentation.

.. index::
   single: distutils


Modules in :mod:`numpy.distutils`
=================================
.. toctree::
   :maxdepth: 2

   distutils/misc_util


.. currentmodule:: numpy.distutils

.. autosummary::
   :toctree: generated/

   ccompiler
   ccompiler_opt
   cpuinfo.cpu
   core.Extension
   exec_command
   log.set_verbosity
   system_info.get_info
   system_info.get_standard_file


Configuration class
===================

.. currentmodule:: numpy.distutils.misc_util

.. class:: Configuration(package_name=None, parent_name=None, top_path=None, package_path=None, **attrs)

    Construct a configuration instance for the given package name. If
    *parent_name* is not None, then construct the package as a
    sub-package of the *parent_name* package. If *top_path* and
    *package_path* are None then they are assumed equal to
    the path of the file this instance was created in. The setup.py
    files in the numpy distribution are good examples of how to use
    the :class:`Configuration` instance.

    .. automethod:: todict

    .. automethod:: get_distribution

    .. automethod:: get_subpackage

    .. automethod:: add_subpackage

    .. automethod:: add_data_files

    .. automethod:: add_data_dir

    .. automethod:: add_include_dirs

    .. automethod:: add_headers

    .. automethod:: add_extension

    .. automethod:: add_library

    .. automethod:: add_scripts

    .. automethod:: add_installed_library

    .. automethod:: add_npy_pkg_config

    .. automethod:: paths

    .. automethod:: get_config_cmd

    .. automethod:: get_build_temp_dir

    .. automethod:: have_f77c

    .. automethod:: have_f90c

    .. automethod:: get_version

    .. automethod:: make_svn_version_py

    .. automethod:: make_config_py

    .. automethod:: get_info

Building installable C libraries
================================

Conventional C libraries (installed through `add_library`) are not installed, and
are just used during the build (they are statically linked).  An installable C
library is a pure C library, which does not depend on the python C runtime, and
is installed such that it may be used by third-party packages. To build and
install the C library, you just use the method `add_installed_library` instead of
`add_library`, which takes the same arguments except for an additional
``install_dir`` argument::

  .. hidden in a comment so as to be included in refguide but not rendered documentation
    >>> import numpy.distutils.misc_util
    >>> config = np.distutils.misc_util.Configuration(None, '', '.')
    >>> with open('foo.c', 'w') as f: pass

  >>> config.add_installed_library('foo', sources=['foo.c'], install_dir='lib')

npy-pkg-config files
--------------------

To make the necessary build options available to third parties, you could use
the `npy-pkg-config` mechanism implemented in `numpy.distutils`. This mechanism is
based on a .ini file which contains all the options. A .ini file is very
similar to .pc files as used by the pkg-config unix utility::

  [meta]
  Name: foo
  Version: 1.0
  Description: foo library

  [variables]
  prefix = /home/user/local
  libdir = ${prefix}/lib
  includedir = ${prefix}/include

  [default]
  cflags = -I${includedir}
  libs = -L${libdir} -lfoo

Generally, the file needs to be generated during the build, since it needs some
information known at build time only (e.g. prefix). This is mostly automatic if
one uses the `Configuration` method `add_npy_pkg_config`. Assuming we have a
template file foo.ini.in as follows::

  [meta]
  Name: foo
  Version: @version@
  Description: foo library

  [variables]
  prefix = @prefix@
  libdir = ${prefix}/lib
  includedir = ${prefix}/include

  [default]
  cflags = -I${includedir}
  libs = -L${libdir} -lfoo

and the following code in setup.py::

  >>> config.add_installed_library('foo', sources=['foo.c'], install_dir='lib')
  >>> subst = {'version': '1.0'}
  >>> config.add_npy_pkg_config('foo.ini.in', 'lib', subst_dict=subst)

This will install the file foo.ini into the directory package_dir/lib, and the
foo.ini file will be generated from foo.ini.in, where each ``@version@`` will be
replaced by ``subst_dict['version']``. The dictionary has an additional prefix
substitution rule automatically added, which contains the install prefix (since
this is not easy to get from setup.py).

Reusing a C library from another package
----------------------------------------

Info are easily retrieved from the `get_info` function in
`numpy.distutils.misc_util`::

  >>> info = np.distutils.misc_util.get_info('npymath')
  >>> config.add_extension('foo', sources=['foo.c'], extra_info=info)
  <numpy.distutils.extension.Extension('foo') at 0x...>


An additional list of paths to look for .ini files can be given to `get_info`.

Conversion of ``.src`` files
============================

NumPy distutils supports automatic conversion of source files named
<somefile>.src. This facility can be used to maintain very similar
code blocks requiring only simple changes between blocks. During the
build phase of setup, if a template file named <somefile>.src is
encountered, a new file named <somefile> is constructed from the
template and placed in the build directory to be used instead. Two
forms of template conversion are supported. The first form occurs for
files named <file>.ext.src where ext is a recognized Fortran
extension (f, f90, f95, f77, for, ftn, pyf). The second form is used
for all other cases. See :ref:`templating`.
