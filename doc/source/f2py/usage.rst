===========
Using F2PY
===========

This page contains a reference to all command-line options for the ``f2py``
command, as well as a reference to internal functions of the ``numpy.f2py``
module.

Using ``f2py`` as a command-line tool
=====================================

When used as a command-line tool, ``f2py`` has three major modes, distinguished
by the usage of ``-c`` and ``-h`` switches.

1. Signature file generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To scan Fortran sources and generate a signature file, use

.. code-block:: sh

  f2py -h <filename.pyf> <options> <fortran files>   \
    [[ only: <fortran functions>  : ]                \
      [ skip: <fortran functions>  : ]]...           \
    [<fortran files> ...]

.. note::

  A Fortran source file can contain many routines, and it is often not
  necessary to allow all routines to be usable from Python. In such cases,
  either specify which routines should be wrapped (in the ``only: .. :`` part)
  or which routines F2PY should ignore (in the ``skip: .. :`` part).

  F2PY has no concept of a "per-file" ``skip`` or ``only`` list, so if functions
  are listed in ``only``, no other functions will be taken from any other files.

If ``<filename.pyf>`` is specified as ``stdout``, then signatures are written to
standard output instead of a file.

Among other options (see below), the following can be used in this mode:

``--overwrite-signature``
  Overwrites an existing signature file.

2. Extension module construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To construct an extension module, use

.. code-block:: sh

  f2py -m <modulename> <options> <fortran files>   \
    [[ only: <fortran functions>  : ]              \
      [ skip: <fortran functions>  : ]]...          \
    [<fortran files> ...]

The constructed extension module is saved as ``<modulename>module.c`` to the
current directory.

Here ``<fortran files>`` may also contain signature files. Among other options
(see below), the following options can be used in this mode:

``--debug-capi``
  Adds debugging hooks to the extension module. When using this extension
  module, various diagnostic information about the wrapper is written to the
  standard output, for example, the values of variables, the steps taken, etc.

``-include'<includefile>'``
  Add a CPP ``#include`` statement to the extension module source.
  ``<includefile>`` should be given in one of the following forms

  .. code-block:: cpp

    "filename.ext"
    <filename.ext>

  The include statement is inserted just before the wrapper functions. This
  feature enables using arbitrary C functions (defined in ``<includefile>``)
  in F2PY generated wrappers.

  .. note:: This option is deprecated. Use ``usercode`` statement to specify
    C code snippets directly in signature files.

``--[no-]wrap-functions``
  Create Fortran subroutine wrappers to Fortran functions.
  ``--wrap-functions`` is default because it ensures maximum portability and
  compiler independence.

``--[no-]freethreading-compatible``
  Create a module that declares it does or doesn't require the GIL. The default
  is ``--no-freethreading-compatible`` for backwards compatibility. Inspect the
  fortran code you are wrapping for thread safety issues before passing
  ``--freethreading-compatible``, as ``f2py`` does not analyze fortran code for
  thread safety issues.

``--include-paths "<path1>:<path2>..."``
  Search include files from given directories.

  .. note:: The paths are to be separated by the correct operating system
            separator :py:data:`~os.pathsep`, that is ``:`` on Linux / MacOS
            and ``;`` on Windows. In ``CMake`` this corresponds to using
            ``$<SEMICOLON>``.

``--help-link [<list of resources names>]``
  List system resources found by ``numpy_distutils/system_info.py``. For
  example, try ``f2py --help-link lapack_opt``.

3. Building a module
~~~~~~~~~~~~~~~~~~~~

To build an extension module, use

.. code-block:: sh

  f2py -c <options> <fortran files>       \
    [[ only: <fortran functions>  : ]     \
      [ skip: <fortran functions>  : ]]... \
    [ <fortran/c source files> ] [ <.o, .a, .so files> ]
 
If ``<fortran files>`` contains a signature file, then the source for an
extension module is constructed, all Fortran and C sources are compiled, and
finally all object and library files are linked to the extension module
``<modulename>.so`` which is saved into the current directory.

If ``<fortran files>`` does not contain a signature file, then an extension
module is constructed by scanning all Fortran source codes for routine
signatures, before proceeding to build the extension module.

.. warning::
   From Python 3.12 onwards, ``distutils`` has been removed. Use environment
   variables or native files to interact with ``meson`` instead. See its `FAQ
   <https://mesonbuild.com/howtox.html>`__ for more information.

Among other options (see below) and options described for previous modes, the following can be used.

.. note::

   .. versionchanged:: 1.26.0
      There are now two separate build backends which can be used, ``distutils``
      and ``meson``. Users are **strongly** recommended to switch to ``meson``
      since it is the default above Python ``3.12``.

Common build flags:

``--backend <backend_type>``
  Specify the build backend for the compilation process.  The supported backends
  are ``meson`` and ``distutils``.  If not specified, defaults to ``distutils``.
  On Python 3.12 or higher, the default is ``meson``.
``--f77flags=<string>``
  Specify F77 compiler flags
``--f90flags=<string>``
  Specify F90 compiler flags
``--debug``
  Compile with debugging information
``-l<libname>``
  Use the library ``<libname>`` when linking.
``-D<macro>[=<defn=1>]``
  Define macro ``<macro>`` as ``<defn>``.
``-U<macro>``
  Define macro ``<macro>``
``-I<dir>``
  Append directory ``<dir>`` to the list of directories searched for include
  files.
``-L<dir>``
  Add directory ``<dir>`` to the list of directories to be searched for
  ``-l``.

The ``meson`` specific flags are:

``--dep <dependency>`` **meson only**
  Specify a meson dependency for the module. This may be passed multiple times
  for multiple dependencies. Dependencies are stored in a list for further
  processing. Example: ``--dep lapack --dep scalapack`` This will identify
  "lapack" and "scalapack" as dependencies and remove them from argv, leaving a
  dependencies list containing ["lapack", "scalapack"].

The older ``distutils`` flags are:

``--help-fcompiler`` **no meson**
  List the available Fortran compilers.
``--fcompiler=<Vendor>`` **no meson**
  Specify a Fortran compiler type by vendor.
``--f77exec=<path>`` **no meson**
  Specify the path to a F77 compiler
``--f90exec=<path>`` **no meson**
  Specify the path to a F90 compiler
``--opt=<string>`` **no meson**
  Specify optimization flags
``--arch=<string>`` **no meson**
  Specify architecture specific optimization flags
``--noopt`` **no meson**
  Compile without optimization flags
``--noarch`` **no meson**
  Compile without arch-dependent optimization flags
``link-<resource>`` **no meson**
  Link the extension module with <resource> as defined by
  ``numpy_distutils/system_info.py``. E.g. to link with optimized LAPACK
  libraries (vecLib on MacOSX, ATLAS elsewhere), use ``--link-lapack_opt``.
  See also ``--help-link`` switch.

.. note::
  
  The ``f2py -c`` option must be applied either to an existing ``.pyf`` file
  (plus the source/object/library files) or one must specify the
  ``-m <modulename>`` option (plus the sources/object/library files). Use one of
  the following options:

  .. code-block:: sh
    
    f2py -c -m fib1 fib1.f

  or

  .. code-block:: sh

    f2py -m fib1 fib1.f -h fib1.pyf
    f2py -c fib1.pyf fib1.f

  For more information, see the `Building C and C++ Extensions`__ Python
  documentation for details.

  __ https://docs.python.org/3/extending/building.html


When building an extension module, a combination of the following macros may be
required for non-gcc Fortran compilers:

.. code-block:: sh

  -DPREPEND_FORTRAN
  -DNO_APPEND_FORTRAN
  -DUPPERCASE_FORTRAN
 
To test the performance of F2PY generated interfaces, use
``-DF2PY_REPORT_ATEXIT``. Then a report of various timings is printed out at the
exit of Python. This feature may not work on all platforms, and currently only
Linux is supported.
 
To see whether F2PY generated interface performs copies of array arguments, use
``-DF2PY_REPORT_ON_ARRAY_COPY=<int>``. When the size of an array argument is
larger than ``<int>``, a message about the copying is sent to ``stderr``.

Other options
~~~~~~~~~~~~~

``-m <modulename>``
  Name of an extension module. Default is ``untitled``.

.. warning::
   Don't use this option if a signature file (``*.pyf``) is used.

   .. versionchanged:: 1.26.3
      Will ignore ``-m`` if a ``pyf`` file is provided.

``--[no-]lower``
  Do [not] lower the cases in ``<fortran files>``. By default, ``--lower`` is
  assumed with ``-h`` switch, and ``--no-lower`` without the ``-h`` switch.
``-include<header>``
  Writes additional headers in the C wrapper, can be passed multiple times,
  generates #include <header> each time. Note that this is meant to be passed
  in single quotes and without spaces, for example ``'-include<stdbool.h>'``
``--build-dir <dirname>``
  All F2PY generated files are created in ``<dirname>``. Default is
  ``tempfile.mkdtemp()``.
``--f2cmap <filename>``
  Load Fortran-to-C ``KIND`` specifications from the given file.
``--quiet``
  Run quietly.
``--verbose``
  Run with extra verbosity.
``--skip-empty-wrappers``
  Do not generate wrapper files unless required by the inputs.
  This is a backwards compatibility flag to restore pre 1.22.4 behavior.
``-v``
  Print the F2PY version and exit.

Execute ``f2py`` without any options to get an up-to-date list of available
options.

.. _python-module-numpy.f2py:

Python module ``numpy.f2py``
============================

.. warning::

   .. versionchanged:: 2.0.0

      There used to be a ``f2py.compile`` function, which was removed, users
      may wrap ``python -m numpy.f2py`` via ``subprocess.run`` manually, and
      set environment variables to interact with ``meson`` as required.

When using ``numpy.f2py`` as a module, the following functions can be invoked.

.. automodule:: numpy.f2py
    :members:

Automatic extension module generation
=====================================

If you want to distribute your f2py extension module, then you only
need to include the .pyf file and the Fortran code. The distutils
extensions in NumPy allow you to define an extension module entirely
in terms of this interface file. A valid ``setup.py`` file allowing
distribution of the ``add.f`` module (as part of the package
``f2py_examples`` so that it would be loaded as ``f2py_examples.add``) is:

.. code-block:: python

    def configuration(parent_package='', top_path=None)
        from numpy.distutils.misc_util import Configuration
        config = Configuration('f2py_examples',parent_package, top_path)
        config.add_extension('add', sources=['add.pyf','add.f'])
        return config

    if __name__ == '__main__':
        from numpy.distutils.core import setup
        setup(**configuration(top_path='').todict())

Installation of the new package is easy using::

    pip install .

assuming you have the proper permissions to write to the main site-
packages directory for the version of Python you are using. For the
resulting package to work, you need to create a file named ``__init__.py``
(in the same directory as ``add.pyf``). Notice the extension module is
defined entirely in terms of the ``add.pyf`` and ``add.f`` files. The
conversion of the .pyf file to a .c file is handled by `numpy.distutils`.

Building with Meson (Examples)
==============================

Using f2py with Meson
~~~~~~~~~~~~~~~~~~~~~

Meson is a modern build system recommended for building Python extension
modules, especially starting with Python 3.12 and NumPy 2.x. Meson provides
a robust and maintainable way to build Fortran extensions with f2py.

To build a Fortran extension using f2py and Meson, you can use Meson's
``custom_target`` to invoke f2py and generate the extension module. The
following minimal example demonstrates how to do this:

This example shows how to build the ``add`` extension from the ``add.f`` and ``add.pyf``
files described in the :ref:`f2py-examples` (note that you do not always need
a ``.pyf`` file: in many cases ``f2py`` can figure out the annotations by itself).

Project layout:

  f2py_examples/
    meson.build
    add.f
    add.pyf (optional)
    __init__.py  (can be empty)

Example ``meson.build``:

.. code-block:: meson

   project('f2py_examples', 'fortran')

   py = import('python').find_installation()

   # List your Fortran source files
   sources = files('add.pyf', 'add.f')

   # Build the extension by invoking f2py via a custom target
   add_mod = custom_target(
     'add_extension',
     input: sources,
     output: ['add' + py.extension_suffix()],
     command: [
       py.full_path(), '-m', 'numpy.f2py',
       '-c', 'add.pyf', 'add.f',
       '-m', 'add'
     ],
     build_by_default: true
   )

   # Install into site-packages under the f2py_examples package
   install_subdir('.', install_dir: join_paths(py.site_packages_dir(), 'f2py_examples'),
                  strip_directory: false,
                  exclude_files: ['meson.build'])

   # Also install the built extension (place it beside __init__.py)
   install_data(add_mod, install_dir: join_paths(py.site_packages_dir(), 'f2py_examples'))

For more details and advanced usage, see the Meson build guide in the
user documentation or refer to SciPy's Meson build files for real-world
examples: https://github.com/scipy/scipy/tree/main/meson.build
