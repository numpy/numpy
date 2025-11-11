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
Instead, backwards incompatible API changes require deprecation warnings for at
least two releases. For more details, see :ref:`NEP23`.

NumPy provides both a Python API and a C-API. The C-API can be accessed
directly or through tools like Cython or f2py. If your package uses the
C-API, it's important to understand NumPy's application binary interface
(ABI) compatibility: NumPy's ABI is forward compatible but not backward
compatible. This means that binaries compiled against an older version of
NumPy will still work with newer versions, but binaries compiled against a
newer version will not necessarily work with older ones.

Modules can also be safely built against NumPy 2.0 or later in
:ref:`CPython's abi3 mode <python:stable-abi>`, which allows
building against a single (minimum-supported) version of Python but be
forward compatible higher versions in the same series (e.g., ``3.x``).
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

    Before NumPy 1.25, the NumPy C-API was *not* exposed in a backwards
    compatible way by default. This means that when compiling with a NumPy
    version earlier than 1.25 you have to compile with the oldest version you
    wish to support. This can be done by using
    `oldest-supported-numpy <https://github.com/scipy/oldest-supported-numpy/>`__.
    Please see the `NumPy 1.24 documentation
    <https://numpy.org/doc/1.24/dev/depending_on_numpy.html>`__.


If a package either uses the NumPy C-API directly or it uses some other tool
that depends on it like Cython or Pythran, NumPy is a *build-time* dependency
of the package.

By default, NumPy exposes a API that is backward compatible with the earliest
NumPy version that supports the oldest Python version currently supported by
NumPy. For example, NumPy 1.25.0 supports Python 3.9 and above; and the
earliest NumPy version to support Python 3.9 was 1.19. Therefore we guarantee
NumPy 1.25 will, when using defaults, expose a C-API compatible with NumPy
1.19. (the exact version is set within NumPy-internal header files).

NumPy is also forward compatible for all minor releases, but a major release
will require recompilation (see NumPy 2.0-specific advice further down).

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
`here <https://conda-forge.org/docs/maintainer/knowledge_base.html#building-against-numpy>`__
for instructions on how to declare a dependency on ``numpy`` when using the C
API.


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
C-API (directly or via Cython), you can also pin the current major
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


.. _numpy-2-abi-handling:

NumPy 2.0-specific advice
~~~~~~~~~~~~~~~~~~~~~~~~~

NumPy 2.0 is an ABI-breaking release, however it does contain support for
building wheels that work on both 2.0 and 1.xx releases. It's important to understand that:

1. When you build wheels for your package using a NumPy 1.xx version at build
   time, those **will not work** with NumPy 2.0.
2. When you build wheels for your package using a NumPy 2.x version at build
   time, those **will work** with NumPy 1.xx.

The first time the NumPy ABI for 2.0 is guaranteed to be stable will be the
release of the first release candidate for 2.0 (i.e., 2.0.0rc1). Our advice for
handling your dependency on NumPy is as follows:

1. In the main (development) branch of your package, do not add any constraints.
2. If you rely on the NumPy C-API (e.g. via direct use in C/C++, or via Cython
   code that uses NumPy), add a ``numpy<2.0`` requirement in your
   package's dependency metadata for releases / in release branches. Do this
   until numpy ``2.0.0rc1`` is released and you can target that.
   *Rationale: the NumPy C ABI will change in 2.0, so any compiled extension
   modules that rely on NumPy will break; they need to be recompiled.*
3. If you rely on a large API surface from NumPy's Python API, also consider
   adding the same ``numpy<2.0`` requirement to your metadata until you are
   sure your code is updated for changes in 2.0 (i.e., when you've tested
   things work against ``2.0.0rc1``).
   *Rationale: we will do a significant API cleanup, with many aliases and
   deprecated/non-recommended objects being removed (see, e.g.,*
   :ref:`numpy-2-migration-guide` *and* :ref:`NEP52`), *so unless you only use
   modern/recommended functions and objects, your code is likely to require at
   least some adjustments.*
4. Plan to do a release of your own packages which depend on ``numpy`` shortly
   after the first NumPy 2.0 release candidate is released (probably around 1
   Feb 2024).
   *Rationale: at that point, you can release packages that will work with both
   2.0 and 1.X, and hence your own end users will not be seeing much/any
   disruption (you want* ``pip install mypackage`` *to continue working on the
   day NumPy 2.0 is released).*
5. Once ``2.0.0rc1`` is available, you can adjust your metadata in
   ``pyproject.toml`` in the way outlined below.

There are two cases: you need to keep compatibility with numpy 1.xx while also
supporting 2.0, or you are able to drop numpy 1.xx support for new releases of
your package and support >=2.0 only. The latter is simpler, but may be more
restrictive for your users. In that case, simply add ``numpy>=2.0`` (or
``numpy>=2.0.0rc1``) to your build and runtime requirements and you're good to
go. We'll focus on the "keep compatibility with 1.xx and 2.x" now, which is a
little more involved.

*Example for a package using the NumPy C-API (via C/Cython/etc.) which wants to support
NumPy 1.23.5 and up*:

.. code:: ini

    [build-system]
    build-backend = ...
    requires = [
        # Note for packagers: this constraint is specific to wheels
        # for PyPI; it is also supported to build against 1.xx still.
        # If you do so, please ensure to include a `numpy<2.0`
        # runtime requirement for those binary packages.
        "numpy>=2.0.0rc1",
        ...
    ]

    [project]
    dependencies = [
        "numpy>=1.23.5",
    ]

We recommend that you have at least one CI job which builds/installs via a wheel,
and then runs tests against the oldest numpy version that the package supports.
For example:

.. code:: yaml

    - name: Build wheel via wheel, then install it
      run: |
        python -m build  # This will pull in numpy 2.0 in an isolated env
        python -m pip install dist/*.whl

    - name: Test against oldest supported numpy version
      run: |
        python -m pip install numpy==1.23.5
        # now run test suite

The above only works once NumPy 2.0 is available on PyPI. If you want to test
against a NumPy 2.0-dev wheel, you have to use a numpy nightly build (see
:ref:`this section <testing-prereleases>` higher up) or build numpy from source.
