Compiler selection and customizing a build
==========================================

Selecting a specific compiler
-----------------------------

Meson supports the standard environment variables ``CC``, ``CXX`` and ``FC`` to
select specific C, C++ and/or Fortran compilers. These environment variables are
documented in `the reference tables in the Meson docs
<https://mesonbuild.com/Reference-tables.html#compiler-and-linker-flag-environment-variables>`__.

Note that environment variables only get applied from a clean build, because
they affect the configure stage (i.e., ``meson setup``). An incremental rebuild
does not react to changes in environment variables - you have to run ``git
clean -xdf`` and do a full rebuild, or run ``meson setup --reconfigure``.


Adding a custom compiler or linker flag
---------------------------------------

Meson by design prefers builds being configured through command-line options
passed to ``meson setup``. It provides many built-in options:

- For enabling a debug build and the optimization level, see the next section
  on "build types",
- Enabling ``-Werror`` in a portable manner is done via ``-Dwerror=true``,
- Enabling warning levels is done via ``-Dwarning_level=<val>``, with ``<val>``
  one of ``{0, 1, 2, 3, everything}``,
- There are many other builtin options, from activating Visual Studio
  (``-Dvsenv=true``) and building with link time optimization (``-Db_lto``) to
  changing the default C++ language level (``-Dcpp_std='c++17'``) or linker
  flags (``-Dcpp_link_args='-Wl,-z,defs'``).

For a comprehensive overview of options, see `Meson's builtin options docs page
<https://mesonbuild.com/Builtin-options.html>`__.

Meson also supports the standard environment variables ``CFLAGS``,
``CXXFLAGS``, ``FFLAGS`` and ``LDFLAGS`` to inject extra flags - with the same
caveat as in the previous section about those environment variables being
picked up only for a clean build and not an incremental build.


Using different build types with Meson
--------------------------------------

Meson provides different build types while configuring the project. You can see
the available options for build types in
`the "core options" section of the Meson documentation <https://mesonbuild.com/Builtin-options.html#core-options>`__.

Assuming that you are building from scratch (do ``git clean -xdf`` if needed),
you can configure the build as following to use the ``debug`` build type::

    spin build -- -Dbuildtype=debug

Now, you can use the ``spin`` interface for further building, installing and
testing NumPy as normal::

    spin test -s linalg

This will work because after initial configuration, Meson will remember the
config options.


Controlling build parallelism
-----------------------------

By default, ``ninja`` will launch ``2*n_cpu + 2``, with ``n_cpu`` the number of
physical CPU cores, parallel build jobs. This is fine in the vast majority of
cases, and results in close to optimal build times. In some cases, on machines
with a small amount of RAM relative to the number of CPU cores, this leads to a
job running out of memory. In case that happens, lower the number of jobs ``N``
such that you have at least 2 GB RAM per job. For example, to launch 6 jobs::

    python -m pip install . -Ccompile-args="-j6"

or::

    spin build -j6

