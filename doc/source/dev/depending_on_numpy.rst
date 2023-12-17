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
https://anaconda.org/scientific-python-nightly-wheels/. Example install command::

    pip install -U --pre --only-binary :all: -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple numpy

This helps detect regressions in NumPy that need fixing before the next NumPy
release.  Furthermore, we recommend to raise errors on warnings in CI for this
job, either all warnings or otherwise at least ``DeprecationWarning`` and
``FutureWarning``. This gives you an early warning about changes in NumPy to
adapt your code.


.. _depending_on_numpy:

Adding a dependency on NumPy
----------------------------

Build-time dependency
~~~~~~~~~~~~~~~~~~~~~

.. note::

    Before NumPy 1.25, the NumPy C-API was *not* backwards compatible.  This
    means that when compiling with a NumPy version earlier than 1.25 you
    have to compile with the oldest version you wish to support.
    This can be done by using
    `oldest-supported-numpy <https://github.com/scipy/oldest-supported-numpy/>`__.
    Please see the `NumPy 1.24 documentation
    <https://numpy.org/doc/1.24/dev/depending_on_numpy.html>`__.


If a package either uses the NumPy C API directly or it uses some other tool
that depends on it like Cython or Pythran, NumPy is a *build-time* dependency
of the package. 

By default, NumPy will expose an API that is backwards compatible with the
oldest NumPy version that supports the currently oldest compatible Python
version.  NumPy 1.25.0 supports Python 3.9 and higher and NumPy 1.19 is the
first version to support Python 3.9.  Thus, we guarantee that, when using
defaults, NumPy 1.25 will expose a C-API compatible with NumPy 1.19.
(the exact version is set within NumPy-internal header files).

NumPy is also forward compatible for all minor releases, but a major release
will require recompilation.

The default behavior can be customized for example by adding::

    #define NPY_TARGET_VERSION NPY_1_22_API_VERSION

before including any NumPy headers (or the equivalent ``-D`` compiler flag) in
every extension module that requires the NumPy C-API.
This is mainly useful if you need to use newly added API at the cost of not
being compatible with older versions.

If for some reason you wish to compile for the currently installed NumPy
version by default you can add::

    #ifndef NPY_TARGET_VERSION
        #define NPY_TARGET_VERSION NPY_API_VERSION
    #endif

Which allows a user to override the default via ``-DNPY_TARGET_VERSION``.
This define must be consistent for each extension module (use of
``import_array()``) and also applies to the umath module.

When you compile against NumPy, you should add the proper version restrictions
to your ``pyproject.toml`` (see PEP 517).  Since your extension will not be
compatible with a new major release of NumPy and may not be compatible with
very old versions.

For conda-forge packages, please see
`here <https://conda-forge.org/docs/maintainer/knowledge_base.html#building-against-numpy>`__.

as of now, it is usually as easy as including::

    host:
      - numpy
    run:
      - {{ pin_compatible('numpy') }}

.. note::

    At the time of NumPy 1.25, NumPy 2.0 is expected to be the next release
    of NumPy.  The NumPy 2.0 release is expected to require a different pin,
    since NumPy 2+ will be needed in order to be compatible with both NumPy
    1.x and 2.x.


Runtime dependency & version ranges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NumPy itself and many core scientific Python packages have agreed on a schedule
for dropping support for old Python and NumPy versions: :ref:`NEP29`. We
recommend all packages depending on NumPy to follow the recommendations in NEP
29.

For *run-time dependencies*, specify version bounds using
``install_requires`` in ``setup.py`` (assuming you use ``numpy.distutils`` or
``setuptools`` to build).

Most libraries that rely on NumPy will not need to set an upper
version bound: NumPy is careful to preserve backward-compatibility.

That said, if you are (a) a project that is guaranteed to release
frequently, (b) use a large part of NumPy's API surface, and (c) is
worried that changes in NumPy may break your code, you can set an
upper bound of ``<MAJOR.MINOR + N`` with N no less than 3, and
``MAJOR.MINOR`` being the current release of NumPy [*]_. If you use the NumPy
C API (directly or via Cython), you can also pin the current major
version to prevent ABI breakage. Note that setting an upper bound on
NumPy may `affect the ability of your library to be installed
alongside other, newer packages
<https://iscinumpy.dev/post/bound-version-constraints/>`__.

.. [*] The reason for setting ``N=3`` is that NumPy will, on the
       rare occasion where it makes breaking changes, raise warnings
       for at least two releases. (NumPy releases about once every six
       months, so this translates to a window of at least a year;
       hence the subsequent requirement that your project releases at
       least on that cadence.)

.. note::


    SciPy has more documentation on how it builds wheels and deals with its
    build-time and runtime dependencies
    `here <https://scipy.github.io/devdocs/dev/core-dev/index.html#distributing>`__.

    NumPy and SciPy wheel build CI may also be useful as a reference, it can be
    found `here for NumPy <https://github.com/MacPython/numpy-wheels>`__ and
    `here for SciPy <https://github.com/MacPython/scipy-wheels>`__.
