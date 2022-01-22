.. _distutils-status-migration:

Status of ``numpy.distutils`` and migration advice
==================================================

`numpy.distutils` has been deprecated in NumPy ``1.23.0``. It will be removed
for Python 3.12; for Python <= 3.11 it will not be removed until 2 years after
the Python 3.12 release (Oct 2025).


Migration advice
----------------

It is **not necessary** to migrate immediately - the release date for Python 3.12
is October 2023, so you have a good amount of time left. It may be beneficial
to wait with migrating until there are examples from other projects to follow
(see below).

There are several build systems which are good options to migrate to. Assuming
you have compiled code in your package (if not, use Flit_) and you want to be
using a well-designed, modern and reliable build system, your two best options
are:

1. Meson_
2. CMake_ (or scikit-build_ as an interface to CMake)

If you have modest needs and have been happy with ``numpy.distutils`` so far,
you can also consider switching to ``setuptools``. Note that most functionality
of ``numpy.disutils`` is unlikely to be ported to ``setuptools``. The likely
exception is nested ``setup.py`` files, but this is not yet done (help with
this is very welcome!).


Moving to Meson
```````````````

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
``````````````````````````````

See the `scikit-build documentation <https://scikit-build.readthedocs.io/en/latest/>`__
for how to use scikit-build. Please note that as of Jan 2022, scikit-build
still relies on setuptools, so it's probably not quite ready yet for a
post-distutils world. For more details on this, see
`this blog post by Henry Schreiner <https://iscinumpy.gitlab.io/post/scikit-build-proposal/>`__.


Interaction of ``numpy.disutils`` with ``setuptools``
-----------------------------------------------------

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

