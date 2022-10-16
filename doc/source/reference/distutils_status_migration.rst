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

It is **not necessary** to migrate immediately - the release date for Python 3.12
is October 2023. It may be beneficial to wait with migrating until there are
examples from other projects to follow (see below).

There are several build systems which are good options to migrate to. Assuming
you have compiled code in your package (if not, we recommend using Flit_) and
you want to be using a well-designed, modern and reliable build system, we
recommend:

1. Meson_
2. CMake_ (or scikit-build_ as an interface to CMake)

If you have modest needs (only simple Cython/C extensions, and perhaps nested
``setup.py`` files) and have been happy with ``numpy.distutils`` so far, you
can also consider switching to ``setuptools``. Note that most functionality of
``numpy.distutils`` is unlikely to be ported to ``setuptools``.


Moving to Meson
~~~~~~~~~~~~~~~

SciPy is moving to Meson for its 1.9.0 release, planned for July 2022. During
this process, any remaining issues with Meson's Python support and achieving
feature parity with ``numpy.distutils`` will be resolved. *Note: parity means a
large superset, but right now some BLAS/LAPACK support is missing and there are
a few open issues related to Cython.* SciPy uses almost all functionality that
``numpy.distutils`` offers, so if SciPy has successfully made a release with
Meson as the build system, there should be no blockers left to migrate, and
SciPy will be a good reference for other packages who are migrating.
For more details about the SciPy migration, see:

- `RFC: switch to Meson as a build system <https://github.com/scipy/scipy/issues/13615>`__
- `Tracking issue for Meson support <https://github.com/rgommers/scipy/issues/22>`__

NumPy itself will very likely migrate to Meson as well, once the SciPy
migration is done.


Moving to CMake / scikit-build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the `scikit-build documentation <https://scikit-build.readthedocs.io/en/latest/>`__
for how to use scikit-build. Please note that as of Feb 2022, scikit-build
still relies on setuptools, so it's probably not quite ready yet for a
post-distutils world. How quickly this changes depends on funding, the current
(Feb 2022) estimate is that if funding arrives then a viable ``numpy.distutils``
replacement will be ready at the end of 2022, and a very polished replacement
mid-2023.  For more details on this, see
`this blog post by Henry Schreiner <https://iscinumpy.gitlab.io/post/scikit-build-proposal/>`__.


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
- a simple user build config system, see [site.cfg.example](https://github.com/numpy/numpy/blob/master/site.cfg.example)
- SIMD intrinsics support

The most widely used feature is nested ``setup.py`` files. This feature will
likely be ported to ``setuptools`` (see
`gh-18588 <https://github.com/numpy/numpy/issues/18588>`__ for status).
Projects only using that feature could move to ``setuptools`` after that is
done. In case a project uses only a couple of ``setup.py`` files, it also could
make sense to simply aggregate all the content of those files into a single
``setup.py`` file and then move to ``setuptools``. This involves dropping all
``Configuration`` instances, and using ``Extension`` instead. E.g.,::

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


.. _Flit: https://flit.readthedocs.io
.. _CMake: https://cmake.org/
.. _Meson: https://mesonbuild.com/
.. _scikit-build: https://scikit-build.readthedocs.io/

