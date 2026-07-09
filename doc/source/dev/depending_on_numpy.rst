.. _for-downstream-package-authors:

For downstream package authors
==============================

This document aims to explain some best practices for authoring a package that
depends on NumPy.


Understanding NumPy's versioning and API/ABI stability
------------------------------------------------------

NumPy uses a standard, :pep:`440` compliant, versioning scheme:
``major.minor.bugfix``. A *major* release is highly unusual and if it happens
it will most likely indicate an ABI break. NumPy 1.xx releases happened from
2006 to 2023; NumPy 2.0 in early 2024 is the first release which changed the
ABI (minor ABI breaks for corner cases may have happened a few times in minor
releases).
*Minor* versions are released regularly, typically every 6 months. Minor
versions contain new features, deprecations, and removals of previously
deprecated code. *Bugfix* releases are made even more frequently; they do not
contain any new features or deprecations.

It is important to know that NumPy, like Python itself and most other
well known scientific Python projects, does **not** use semantic versioning.
Instead, backward incompatible API changes require deprecation warnings for at
least two releases. For more details, see :ref:`NEP23`.

NumPy provides both a Python API and a C-API. The C-API can be accessed
directly or through tools like Cython or f2py. If your package uses the
NumPy C-API, it will generally be backward compatible with all relevant older
NumPy versions and forward compatible within the same major NumPy version.
For more details, for example if you wish to use API added in newer NumPy
versions, see :ref:`depending_on_numpy`.

Modules can also be safely built against NumPy 2.0 or later in
:ref:`CPython's abi3 mode <python:stable-abi>`, which allows
building against a single (minimum-supported) version of Python but be
forward compatible with higher versions in the same series (e.g., ``3.x``).
This can greatly reduce the number of wheels that need to be built and
distributed. For more information and examples, see the
`cibuildwheel docs <https://cibuildwheel.pypa.io/en/stable/faq/#abi3>`__.

.. _testing-prereleases:

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

If you want to test your own wheel builds against the latest NumPy nightly
build and you're using ``cibuildwheel``, you may need something like this in
your CI config file:

.. code::

    CIBW_ENVIRONMENT: "PIP_PRE=1 PIP_EXTRA_INDEX_URL=https://pypi.anaconda.org/scientific-python-nightly-wheels/simple"


.. _depending_on_numpy:

Adding a dependency on NumPy
----------------------------

Build-time dependency
~~~~~~~~~~~~~~~~~~~~~

.. note::

    Before NumPy 1.25, the NumPy C-API was *not* exposed in a backward
    compatible way by default. This means that when compiling with a NumPy
    version earlier than 1.25 you have to compile with the oldest version you
    wish to support. This can be done by using
    `oldest-supported-numpy <https://github.com/scipy/oldest-supported-numpy/>`__.
    Please see the `NumPy 1.24 documentation
    <https://numpy.org/doc/1.24/dev/depending_on_numpy.html>`__.


If a package either uses the NumPy C-API directly or it uses some other tool
that depends on it like Cython or Pythran, NumPy is a *build-time* dependency
of the package.

By default, NumPy exposes an API that is backward compatible with the earliest
NumPy version that supports the oldest Python version currently supported by
NumPy. For example, NumPy 1.25.0 supports Python 3.9 and above; and the
earliest NumPy version to support Python 3.9 was 1.19. Therefore we guarantee
NumPy 1.25 will, when using defaults, expose a C-API compatible with NumPy
1.19. (the exact version is set within NumPy-internal header files).

NumPy is also forward compatible for all minor releases, but a major release
is expected to require recompilation (see :ref:`numpy-2-abi-handling` further down).\
[#major-transition]_

The default behavior can be customized for example by adding::

    #define NPY_TARGET_VERSION NPY_1_22_API_VERSION

before including any NumPy headers (or the equivalent ``-D`` compiler flag) in
every extension module that requires the NumPy C-API.
This is mainly useful if you need to use newly added API at the cost of not
being compatible with older versions.\ [#future-api]_

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
`here <https://conda-forge.org/docs/maintainer/knowledge_base.html#building-against-numpy>`__
for instructions on how to declare a dependency on ``numpy`` when using the C
API.

.. [#major-transition] Improving on the NumPy 2 transition, if a NumPy 3.0
    release happens, we may release NumPy 2.x versions that can compile
    modules that are compatible with NumPy 3.0 when deprecated APIs are
    disabled.

.. [#future-api] We do not provide a mechanism to compile a single extension
    module that is compatible with old NumPy versions but uses new NumPy API
    when running with a newer NumPy version.
    This can be achieved via ``PyArray_RUNTIME_VERSION`` and manually
    backported API, although you may not receive fixes from NumPy headers when
    using a backport. We recommend only backporting when
    ``NPY_FEATURE_VERSION`` is lower than the version at which the API was
    introduced. That way you pick up official header definitions and fixes
    once they are available. Even without bugs, future NumPy versions could
    introduce incompatible changes eventually.



Runtime dependency & version ranges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NumPy itself and many core scientific Python packages have agreed on a schedule
for dropping support for old Python and NumPy versions: :ref:`NEP29`. We
recommend all packages depending on NumPy to follow the recommendations in NEP
29.

For *run-time dependencies*, specify version bounds in `pyproject.toml`.

Most libraries that rely on NumPy will not need to set an upper
version bound: NumPy is careful to preserve backward-compatibility.

That said, if you are (a) a project that is guaranteed to release
frequently, (b) use a large part of NumPy's API surface, and (c) is
worried that changes in NumPy may break your code, you can set an
upper bound of ``<MAJOR.MINOR + N`` with N no less than 3, and
``MAJOR.MINOR`` being the current release of NumPy.\ [#N]_ If you use the NumPy
C-API (directly or via Cython), you can also pin the current major
version to prevent ABI breakage. Note that setting an upper bound on
NumPy may `affect the ability of your library to be installed
alongside other, newer packages
<https://iscinumpy.dev/post/bound-version-constraints/>`__.

.. [#N] The reason for setting ``N=3`` is that NumPy will, on the
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


.. _numpy-2-abi-handling:

NumPy 2.0 ABI handling
~~~~~~~~~~~~~~~~~~~~~~

NumPy 2.0 changed the C ABI. The important rule for binary wheels is:

1. Wheels built against NumPy 1.xx **will not work** with NumPy 2.0 or later.
2. Wheels built against NumPy 2.x **will work** with NumPy 1.xx at runtime.
   How old NumPy versions are supported can be customized with ``NPY_TARGET_VERSION``,
   see :ref:`depending_on_numpy`.

If your package uses the NumPy C-API (directly or via Cython), you need to
rebuild and release wheels compiled against NumPy 2.x. Pure Python packages
may also need code updates; see :ref:`numpy-2-migration-guide`.

There are two common cases:

**Keep compatibility with NumPy 1.xx and 2.x**

Build against NumPy 2.x, but keep a lower runtime bound. For example, to
support NumPy 1.23.5 and up:

.. code:: ini

    [build-system]
    build-backend = ...
    requires = [
        "numpy>=2.0",
        ...
    ]

    [project]
    dependencies = [
        "numpy>=1.23.5",
    ]

**Support NumPy 2.x only**

This is simpler, but more restrictive for your users:

.. code:: ini

    [build-system]
    build-backend = ...
    requires = [
        "numpy>=2.0",
        ...
    ]

    [project]
    dependencies = [
        "numpy>=2.0",
    ]

We recommend at least one CI job that builds a wheel and then tests it against
the oldest NumPy version you support. For example:

.. code:: yaml

    - name: Build wheel, then install it
      run: |
        python -m build
        python -m pip install dist/*.whl

    - name: Test against oldest supported NumPy version
      run: |
        python -m pip install numpy==1.23.5
        # now run test suite

To test against unreleased NumPy versions, use a nightly build (see
:ref:`this section <testing-prereleases>`) or build NumPy from source.
