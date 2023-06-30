.. _distutils-status-migration:

Status of ``numpy.distutils`` and migration advice
==================================================

`numpy.distutils` has been deprecated in NumPy ``1.23.0``. It will be removed
for Python 3.12; for Python <= 3.11 it will not be removed until 2 years after
the Python 3.12 release (Oct 2025).


.. warning::

   ``numpy.distutils`` is only tested with ``setuptools < 60.0``, newer
   versions may break. See :ref:`numpy-setuptools-interaction` for details.


Migration advice
----------------

There are several build systems which are good options to migrate to. Assuming
you have compiled code in your package (if not, you have several good options,
e.g. the build backends offered by Poetry, Hatch or PDM) and you want to be
using a well-designed, modern and reliable build system, we recommend:

1. Meson_, and the meson-python_ build backend
2. CMake_, and the scikit-build-core_ build backend

If you have modest needs (only simple Cython/C extensions; no need for Fortran,
BLAS/LAPACK, nested ``setup.py`` files, or other features of
``numpy.distutils``) and have been happy with ``numpy.distutils`` so far, you
can also consider switching to ``setuptools``. Note that most functionality of
``numpy.distutils`` is unlikely to be ported to ``setuptools``.

Moving to Meson
~~~~~~~~~~~~~~~

SciPy has moved to Meson and meson-python for its 1.9.0 release. During
this process, remaining issues with Meson's Python support and
feature parity with ``numpy.distutils`` were resolved. *Note: parity means a
large superset (because Meson is a good general-purpose build system); only
a few BLAS/LAPACK library selection niceties are missing*. SciPy uses almost
all functionality that ``numpy.distutils`` offers, so if SciPy has successfully
made a release with Meson as the build system, there should be no blockers left
to migrate, and SciPy will be a good reference for other packages who are
migrating. For more details about the SciPy migration, see:

- `RFC: switch to Meson as a build system <https://github.com/scipy/scipy/issues/13615>`__
- `Tracking issue for Meson support <https://github.com/rgommers/scipy/issues/22>`__

NumPy will migrate to Meson for the 1.26 release.


Moving to CMake / scikit-build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next generation of scikit-build is called scikit-build-core_. Where the
older ``scikit-build`` used ``setuptools`` underneath, the rewrite does not.
Like Meson, CMake is a good general-purpose build system.


Moving to ``setuptools``
~~~~~~~~~~~~~~~~~~~~~~~~

For projects that only use ``numpy.distutils`` for historical reasons, and do
not actually use features beyond those that ``setuptools`` also supports,
moving to ``setuptools`` is likely the solution which costs the least effort.
To assess that, there are the ``numpy.distutils`` features that are *not*
present in ``setuptools``:

- Nested ``setup.py`` files
- Fortran build support
- BLAS/LAPACK library support (OpenBLAS, MKL, ATLAS, Netlib LAPACK/BLAS, BLIS, 64-bit ILP interface, etc.)
- Support for a few other scientific libraries, like FFTW and UMFPACK
- Better MinGW support
- Per-compiler build flag customization (e.g. `-O3` and `SSE2` flags are default)
- a simple user build config system, see `site.cfg.example <https://github.com/numpy/numpy/blob/master/site.cfg.example>`__
- SIMD intrinsics support

The most widely used feature is nested ``setup.py`` files. This feature may
perhaps still be ported to ``setuptools`` in the future (it needs a volunteer
though, see `gh-18588 <https://github.com/numpy/numpy/issues/18588>`__ for
status). Projects only using that feature could move to ``setuptools`` after
that is done. In case a project uses only a couple of ``setup.py`` files, it
also could make sense to simply aggregate all the content of those files into a
single ``setup.py`` file and then move to ``setuptools``. This involves
dropping all ``Configuration`` instances, and using ``Extension`` instead.
E.g.,::

    from distutils.core import setup
    from distutils.extension import Extension
    setup(name='foobar',
          version='1.0',
          ext_modules=[
              Extension('foopkg.foo', ['foo.c']),
              Extension('barpkg.bar', ['bar.c']),
              ],
          )

For more details, see the
`setuptools documentation <https://setuptools.pypa.io/en/latest/setuptools.html>`__


.. _numpy-setuptools-interaction:

Interaction of ``numpy.distutils`` with ``setuptools``
------------------------------------------------------

It is recommended to use ``setuptools < 60.0``. Newer versions may work, but
are not guaranteed to. The reason for this is that ``setuptools`` 60.0 enabled
a vendored copy of ``distutils``, including backwards incompatible changes that
affect some functionality in ``numpy.distutils``.

If you are using only simple Cython or C extensions with minimal use of
``numpy.distutils`` functionality beyond nested ``setup.py`` files (its most
popular feature, see :class:`Configuration <numpy.distutils.misc_util.Configuration>`),
then latest ``setuptools`` is likely to continue working. In case of problems,
you can also try ``SETUPTOOLS_USE_DISTUTILS=stdlib`` to avoid the backwards
incompatible changes in ``setuptools``.

Whatever you do, it is recommended to put an upper bound on your ``setuptools``
build requirement in ``pyproject.toml`` to avoid future breakage - see
:ref:`for-downstream-package-authors`.


.. _CMake: https://cmake.org/
.. _Meson: https://mesonbuild.com/
.. _meson-python: https://meson-python.readthedocs.io
.. _scikit-build-core: https://scikit-build-core.readthedocs.io/en/latest/
