.. _meson-introspection:

Introspecting build steps
=========================

When you have an issue with a particular Python extension module or other build
target, there are a number of ways to figure out what the build system is doing
exactly. Beyond looking at the ``meson.build`` content for the target of
interest, these include:

1. Reading the generated ``build.ninja`` file in the build directory,
2. Using ``meson introspect`` to learn more about build options, dependencies
   and flags used for the target,
3. Reading ``<build-dir>/meson-info/*.json`` for details on discovered
   dependencies, where Meson plans to install files to, etc.

These things are all available after the configure stage of the build (i.e.,
``meson setup``) has run. It is typically more effective to look at this
information, rather than running the build and reading the full build log.

For more details on this topic, see the
`SciPy doc page on build introspection <https://scipy.github.io/devdocs/building/introspecting_a_build.html>`__.
