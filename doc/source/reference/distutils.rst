**********************************
Packaging (:mod:`numpy.distutils`)
**********************************

.. module:: numpy.distutils

NumPy provides enhanced distutils functionality to make it easier to
build and install sub-packages, auto-generate code, and extension
modules that use Fortran-compiled libraries. To use features of numpy
distutils, use the :func:`setup <core.setup>` command from
:mod:`numpy.distutils.core`. A useful :class:`Configuration
<misc_util.Configuration>` class is also provided in
:mod:`numpy.distutils.misc_util` that can make it easier to construct
keyword arguments to pass to the setup function (by passing the
dictionary obtained from the todict() method of the class). More
information is available in the NumPy Distutils Users Guide in
``<site-packages>/numpy/doc/DISTUTILS.txt``.

.. index::
   single: distutils


Modules in :mod:`numpy.distutils`
=================================

misc_util
---------

.. module:: numpy.distutils.misc_util

.. autosummary::
   :toctree: generated/

   Configuration
   get_numpy_include_dirs
   get_numarray_include_dirs
   dict_append
   appendpath
   allpath
   dot_join
   generate_config_py
   get_cmd
   terminal_has_colors
   red_text
   green_text
   yellow_text
   blue_text
   cyan_text
   cyg2win32
   all_strings
   has_f_sources
   has_cxx_sources
   filter_sources
   get_dependencies
   is_local_src_dir
   get_ext_source_files
   get_script_files


.. class:: Configuration(package_name=None, parent_name=None, top_path=None, package_path=None, **attrs)

    Construct a configuration instance for the given package name. If
    *parent_name* is not :const:`None`, then construct the package as a
    sub-package of the *parent_name* package. If *top_path* and
    *package_path* are :const:`None` then they are assumed equal to
    the path of the file this instance was created in. The setup.py
    files in the numpy distribution are good examples of how to use
    the :class:`Configuration` instance.

    .. method:: todict()

        Return a dictionary compatible with the keyword arguments of distutils
        setup function. Thus, this method may be used as
        setup(\**config.todict()).

    .. method:: get_distribution()

        Return the distutils distribution object for self.

    .. method:: get_subpackage(subpackage_name, subpackage_path=None)

        Return a Configuration instance for the sub-package given. If
        subpackage_path is None then the path is assumed to be the local path
        plus the subpackage_name. If a setup.py file is not found in the
        subpackage_path, then a default configuration is used.

    .. method:: add_subpackage(subpackage_name, subpackage_path=None)

        Add a sub-package to the current Configuration instance. This is
        useful in a setup.py script for adding sub-packages to a package. The
        sub-package is contained in subpackage_path / subpackage_name and this
        directory may contain a setup.py script or else a default setup
        (suitable for Python-code-only subpackages) is assumed. If the
        subpackage_path is None, then it is assumed to be located in the local
        path / subpackage_name.

    .. method:: self.add_data_files(*files)

        Add files to the list of data_files to be included with the package.
        The form of each element of the files sequence is very flexible
        allowing many combinations of where to get the files from the package
        and where they should ultimately be installed on the system. The most
        basic usage is for an element of the files argument sequence to be a
        simple filename. This will cause that file from the local path to be
        installed to the installation path of the self.name package (package
        path). The file argument can also be a relative path in which case the
        entire relative path will be installed into the package directory.
        Finally, the file can be an absolute path name in which case the file
        will be found at the absolute path name but installed to the package
        path.

        This basic behavior can be augmented by passing a 2-tuple in as the
        file argument. The first element of the tuple should specify the
        relative path (under the package install directory) where the
        remaining sequence of files should be installed to (it has nothing to
        do with the file-names in the source distribution). The second element
        of the tuple is the sequence of files that should be installed. The
        files in this sequence can be filenames, relative paths, or absolute
        paths. For absolute paths the file will be installed in the top-level
        package installation directory (regardless of the first argument).
        Filenames and relative path names will be installed in the package
        install directory under the path name given as the first element of
        the tuple. An example may clarify::

            self.add_data_files('foo.dat',
            ('fun', ['gun.dat', 'nun/pun.dat', '/tmp/sun.dat']),
            'bar/cat.dat',
            '/full/path/to/can.dat')

        will install these data files to::

            <package install directory>/
             foo.dat
             fun/
               gun.dat
               nun/
                 pun.dat
             sun.dat
             bar/
               car.dat
             can.dat

        where <package install directory> is the package (or sub-package)
        directory such as '/usr/lib/python2.4/site-packages/mypackage' ('C: \\Python2.4 \\Lib \\site-packages \\mypackage') or '/usr/lib/python2.4/site-
        packages/mypackage/mysubpackage' ('C: \\Python2.4 \\Lib \\site-packages \\mypackage \\mysubpackage').


        An additional feature is that the path to a data-file can actually be
        a function that takes no arguments and returns the actual path(s) to
        the data-files. This is useful when the data files are generated while
        building the package.

    .. method:: add_data_dir(data_path)

        Recursively add files under data_path to the list of data_files to be
        installed (and distributed). The data_path can be either a relative
        path-name, or an absolute path-name, or a 2-tuple where the first
        argument shows where in the install directory the data directory
        should be installed to. For example suppose the source directory
        contains fun/foo.dat and fun/bar/car.dat::

            self.add_data_dir('fun')
            self.add_data_dir(('sun', 'fun'))
            self.add_data_dir(('gun', '/full/path/to/fun'))

        Will install data-files to the locations::

            <package install directory>/
              fun/
                foo.dat
                bar/
                  car.dat
              sun/
                foo.dat
                bar/
                  car.dat
              gun/
                foo.dat
                car.dat

    .. method:: add_include_dirs(*paths)

        Add the given sequence of paths to the beginning of the include_dirs
        list. This list will be visible to all extension modules of the
        current package.

    .. method:: add_headers(*files)

        Add the given sequence of files to the beginning of the headers list.
        By default, headers will be installed under <python-
        include>/<self.name.replace('.','/')>/ directory. If an item of files
        is a tuple, then its first argument specifies the actual installation
        location relative to the <python-include> path.

    .. method:: add_extension(name, sources, **kw)

        Create and add an Extension instance to the ext_modules list. The
        first argument defines the name of the extension module that will be
        installed under the self.name package. The second argument is a list
        of sources. This method also takes the following optional keyword
        arguments that are passed on to the Extension constructor:
        include_dirs, define_macros, undef_macros, library_dirs, libraries,
        runtime_library_dirs, extra_objects, swig_opts, depends, language,
        f2py_options, module_dirs, and extra_info.

        The self.paths(...) method is applied to all lists that may contain
        paths. The extra_info is a dictionary or a list of dictionaries whose
        content will be appended to the keyword arguments. The depends list
        contains paths to files or directories that the sources of the
        extension module depend on. If any path in the depends list is newer
        than the extension module, then the module will be rebuilt.

        The list of sources may contain functions (called source generators)
        which must take an extension instance and a build directory as inputs
        and return a source file or list of source files or None. If None is
        returned then no sources are generated. If the Extension instance has
        no sources after processing all source generators, then no extension
        module is built.

    .. method:: add_library(name, sources, **build_info)

        Add a library to the list of libraries. Allowed keyword arguments are
        depends, macros, include_dirs, extra_compiler_args, and f2py_options.
        The name is the name of the library to be built and sources is a list
        of sources (or source generating functions) to add to the library.

    .. method:: add_scripts(*files)

        Add the sequence of files to the beginning of the scripts list.
        Scripts will be installed under the <prefix>/bin/ directory.

    .. method:: paths(*paths)

        Applies glob.glob(...) to each path in the sequence (if needed) and
        pre-pends the local_path if needed. Because this is called on all
        source lists, this allows wildcard characters to be specified in lists
        of sources for extension modules and libraries and scripts and allows
        path-names be relative to the source directory.

    .. method:: get_config_cmd()

        Returns the numpy.distutils config command instance.

    .. method:: get_build_temp_dir()

        Return a path to a temporary directory where temporary files should be
        placed.

    .. method:: have_f77c()

        True if a Fortran 77 compiler is available (because a simple Fortran
        77 code was able to be compiled successfully).

    .. method:: have_f90c()

        True if a Fortran 90 compiler is available (because a simple Fortran
        90 code was able to be compiled successfully)

    .. method:: get_version()

        Return a version string of the current package or None if the version
        information could not be detected. This method scans files named
        __version__.py, <packagename>_version.py, version.py, and
        __svn_version__.py for string variables version, __version\__, and
        <packagename>_version, until a version number is found.

    .. method:: make_svn_version_py()

        Appends a data function to the data_files list that will generate
        __svn_version__.py file to the current package directory. This file
        will be removed from the source directory when Python exits (so that
        it can be re-generated next time the package is built). This is
        intended for working with source directories that are in an SVN
        repository.

    .. method:: make_config_py()

        Generate a package __config__.py file containing system information
        used during the building of the package. This file is installed to the
        package installation directory.

    .. method:: get_info(*names)

        Return information (from system_info.get_info) for all of the names in
        the argument list in a single dictionary.


Other modules
-------------

.. currentmodule:: numpy.distutils

.. autosummary::
   :toctree: generated/

   system_info.get_info
   system_info.get_standard_file
   cpuinfo.cpu
   log.set_verbosity
   exec_command


Conversion of ``.src`` files
============================

NumPy distutils supports automatic conversion of source files named
<somefile>.src. This facility can be used to maintain very similar
code blocks requiring only simple changes between blocks. During the
build phase of setup, if a template file named <somefile>.src is
encountered, a new file named <somefile> is constructed from the
template and placed in the build directory to be used instead. Two
forms of template conversion are supported. The first form occurs for
files named named <file>.ext.src where ext is a recognized Fortran
extension (f, f90, f95, f77, for, ftn, pyf). The second form is used
for all other cases.

.. index::
   single: code generation

Fortran files
-------------

This template converter will replicate all **function** and
**subroutine** blocks in the file with names that contain '<...>'
according to the rules in '<...>'. The number of comma-separated words
in '<...>' determines the number of times the block is repeated. What
these words are indicates what that repeat rule, '<...>', should be
replaced with in each block. All of the repeat rules in a block must
contain the same number of comma-separated words indicating the number
of times that block should be repeated. If the word in the repeat rule
needs a comma, leftarrow, or rightarrow, then prepend it with a
backslash ' \'. If a word in the repeat rule matches ' \\<index>' then
it will be replaced with the <index>-th word in the same repeat
specification. There are two forms for the repeat rule: named and
short.


Named repeat rule
^^^^^^^^^^^^^^^^^

A named repeat rule is useful when the same set of repeats must be
used several times in a block. It is specified using <rule1=item1,
item2, item3,..., itemN>, where N is the number of times the block
should be repeated. On each repeat of the block, the entire
expression, '<...>' will be replaced first with item1, and then with
item2, and so forth until N repeats are accomplished. Once a named
repeat specification has been introduced, the same repeat rule may be
used **in the current block** by referring only to the name
(i.e. <rule1>.


Short repeat rule
^^^^^^^^^^^^^^^^^

A short repeat rule looks like <item1, item2, item3, ..., itemN>. The
rule specifies that the entire expression, '<...>' should be replaced
first with item1, and then with item2, and so forth until N repeats
are accomplished.


Pre-defined names
^^^^^^^^^^^^^^^^^

The following predefined named repeat rules are available:

- <prefix=s,d,c,z>

- <_c=s,d,c,z>

- <_t=real, double precision, complex, double complex>

- <ftype=real, double precision, complex, double complex>

- <ctype=float, double, complex_float, complex_double>

- <ftypereal=float, double precision, \\0, \\1>

- <ctypereal=float, double, \\0, \\1>


Other files
-----------

Non-Fortran files use a separate syntax for defining template blocks
that should be repeated using a variable expansion similar to the
named repeat rules of the Fortran-specific repeats. The template rules
for these files are:

1. "/\**begin repeat "on a line by itself marks the beginning of
   a segment that should be repeated.

2. Named variable expansions are defined using #name=item1, item2, item3,
   ..., itemN# and placed on successive lines. These variables are
   replaced in each repeat block with corresponding word. All named
   variables in the same repeat block must define the same number of
   words.

3. In specifying the repeat rule for a named variable, item*N is short-
   hand for item, item, ..., item repeated N times. In addition,
   parenthesis in combination with \*N can be used for grouping several
   items that should be repeated. Thus, #name=(item1, item2)*4# is
   equivalent to #name=item1, item2, item1, item2, item1, item2, item1,
   item2#

4. "\*/ "on a line by itself marks the end of the the variable expansion
   naming. The next line is the first line that will be repeated using
   the named rules.

5. Inside the block to be repeated, the variables that should be expanded
   are specified as @name@.

6. "/\**end repeat**/ "on a line by itself marks the previous line
   as the last line of the block to be repeated.
