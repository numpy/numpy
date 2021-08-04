.. _for-downstream-package-authors:

For downstream package authors
==============================

This document aims to explain some best practices for authoring a package that
depends on NumPy.


Understanding NumPy's versioning and API/ABI stability
------------------------------------------------------

NumPy uses a standard, :pep:`440` compliant, versioning scheme:
``major.minor.bugfix``. A *major* release is highly unusual (NumPy is still at
version ``1.xx``) and if it happens it will likely indicate an ABI break.
*Minor* versions are released regularly, typically every 6 months. Minor
versions contain new features, deprecations, and removals of previously
deprecated code. *Bugfix* releases are made even more frequently; they do not
contain any new features or deprecations.

It is important to know that NumPy, like Python itself and most other
well known scientific Python projects, does **not** use semantic versioning.
Instead, backwards incompatible API changes require deprecation warnings for at
least two releases. For more details, see :ref:`NEP23`.

NumPy has both a Python API and a C API. The C API can be used directly or via
Cython, f2py, or other such tools. If your package uses the C API, then ABI
(application binary interface) stability of NumPy is important. NumPy's ABI is
forward but not backward compatible. This means: binaries compiled against a
given version of NumPy will still run correctly with newer NumPy versions, but
not with older versions.


Testing against the NumPy main branch or pre-releases
-----------------------------------------------------

For large, actively maintained packages that depend on NumPy, we recommend
testing against the development version of NumPy in CI. To make this easy,
nightly builds are provided as wheels at
https://anaconda.org/scipy-wheels-nightly/.
This helps detect regressions in NumPy that need fixing before the next NumPy
release.  Furthermore, we recommend to raise errors on warnings in CI for this
job, either all warnings or otherwise at least ``DeprecationWarning`` and
``FutureWarning``. This gives you an early warning about changes in NumPy to
adapt your code.


Adding a dependency on NumPy
----------------------------

Build-time dependency
`````````````````````

If a package either uses the NumPy C API directly or it uses some other tool
that depends on it like Cython or Pythran, NumPy is a *build-time* dependency
of the package. Because the NumPy ABI is only forward compatible, you must
build your own binaries (wheels or other package formats) against the lowest
NumPy version that you support (or an even older version).

Picking the correct NumPy version to build against for each Python version and
platform can get complicated. There are a couple of ways to do this.
Build-time dependencies are specified in ``pyproject.toml`` (see PEP 517),
which is the file used to build wheels by PEP 517 compliant tools (e.g.,
when using ``pip wheel``).

You can specify everything manually in ``pyproject.toml``, or you can instead
rely on the `oldest-supported-numpy <https://github.com/scipy/oldest-supported-numpy/>`__
metapackage. ``oldest-supported-numpy`` will specify the correct NumPy version
at build time for wheels, taking into account Python version, Python
implementation (CPython or PyPy), operating system and hardware platform. It
will specify the oldest NumPy version that supports that combination of
characteristics.  Note: for platforms for which NumPy provides wheels on PyPI,
it will be the first version with wheels (even if some older NumPy version
happens to build).

For conda-forge it's a little less complicated: there's dedicated handling for
NumPy in build-time and runtime dependencies, so typically this is enough
(see `here <https://conda-forge.org/docs/maintainer/knowledge_base.html#building-against-numpy>`__ for docs)::

    host:
      - numpy
    run:
      - {{ pin_compatible('numpy') }}

.. note::

    ``pip`` has ``--no-use-pep517`` and ``--no-build-isolation`` flags that may
    ignore ``pyproject.toml`` or treat it differently - if users use those
    flags, they are responsible for installing the correct build dependencies
    themselves.

    ``conda`` will always use ``-no-build-isolation``; dependencies for conda
    builds are given in the conda recipe (``meta.yaml``), the ones in
    ``pyproject.toml`` have no effect.

    Please do not use ``setup_requires`` (it is deprecated and may invoke
    ``easy_install``).

Because for NumPy you have to care about ABI compatibility, you
specify the version with ``==`` to the lowest supported version. For your other
build dependencies you can probably be looser, however it's still important to
set lower and upper bounds for each dependency. It's fine to specify either a
range or a specific version for a dependency like ``wheel`` or ``setuptools``.
It's recommended to set the upper bound of the range to the latest already
released version of ``wheel`` and ``setuptools`` - this prevents future
releases from breaking your packages on PyPI.


Runtime dependency & version ranges
```````````````````````````````````

NumPy itself and many core scientific Python packages have agreed on a schedule
for dropping support for old Python and NumPy versions: :ref:`NEP29`. We
recommend all packages depending on NumPy to follow the recommendations in NEP
29.

For *run-time dependencies*, you specify the range of versions in
``install_requires`` in ``setup.py`` (assuming you use ``numpy.distutils`` or
``setuptools`` to build). Getting the upper bound right for NumPy is slightly
tricky. If we don't set any bound, a too-new version will be pulled in a few
years down the line, and NumPy may have deprecated and removed some API that
your package depended on by then. On the other hand if you set the upper bound
to the newest already-released version, then as soon as a new NumPy version is
released there will be no matching version of your package that works with it.

What to do here depends on your release frequency. Given that NumPy releases
come in a 6-monthly cadence and that features that get deprecated in NumPy
should stay around for another two releases, a good upper bound is
``<1.(xx+3).0`` - where ``xx`` is the minor version of the latest
already-released NumPy. This is safe to do if you release at least once a year.
If your own releases are much less frequent, you may set the upper bound a
little further into the future - this is a trade-off between a future NumPy
version _maybe_ removing something you rely on, and the upper bound being
exceeded which _may_ lead to your package being hard to install in combination
with other packages relying on the latest NumPy.


.. note::


    SciPy has more documentation on how it builds wheels and deals with its
    build-time and runtime dependencies
    `here <https://scipy.github.io/devdocs/dev/core-dev/index.html#distributing>`__.

    NumPy and SciPy wheel build CI may also be useful as a reference, it can be
    found `here for NumPy <https://github.com/MacPython/numpy-wheels>`__ and
    `here for SciPy <https://github.com/MacPython/scipy-wheels>`__.
