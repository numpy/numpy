.. _building-from-source:

Building from source
====================

..
  This page is referenced from numpy/numpy/__init__.py. Please keep its
  location in sync with the link there.

.. note::

   If you are only trying to install NumPy, we recommend using binaries - see
   `Installation <https://numpy.org/install>`__ for details on that.

Building NumPy from source requires setting up system-level dependencies
(compilers, BLAS/LAPACK libraries, etc.) first, and then invoking a build. The
build may be done in order to install NumPy for local usage, develop NumPy
itself, or build redistributable binary packages. And it may be desired to
customize aspects of how the build is done. This guide will cover all these
aspects. In addition, it provides background information on how the NumPy build
works, and links to up-to-date guides for generic Python build & packaging
documentation that is relevant.

.. _system-level:

System-level dependencies
-------------------------

NumPy uses compiled code for speed, which means you need compilers and some
other system-level (i.e, non-Python / non-PyPI) dependencies to build it on
your system.

.. note::

    If you are using Conda, you can skip the steps in this section - with the
    exception of installing compilers for Windows or the Apple Developer Tools
    for macOS. All other dependencies will be installed automatically by the
    ``mamba env create -f environment.yml`` command.

.. tab-set::

  .. tab-item:: Linux
    :sync: linux

    If you want to use the system Python and ``pip``, you will need:

    * C and C++ compilers (typically GCC).

    * Python header files (typically a package named ``python3-dev`` or
      ``python3-devel``)

    * BLAS and LAPACK libraries. `OpenBLAS <https://github.com/OpenMathLib/OpenBLAS/>`__
      is the NumPy default; other variants include Apple Accelerate,
      `MKL <https://software.intel.com/en-us/intel-mkl>`__,
      `ATLAS <http://math-atlas.sourceforge.net/>`__ and
      `Netlib <https://www.netlib.org/lapack/index.html>`__ (or "Reference")
      BLAS and LAPACK.

    * ``pkg-config`` for dependency detection.

    * A Fortran compiler is needed only for running the ``f2py`` tests. The
      instructions below include a Fortran compiler, however you can safely
      leave it out.

    .. tab-set::

      .. tab-item:: Debian/Ubuntu Linux

        To install NumPy build requirements, you can do::

          sudo apt install -y gcc g++ gfortran libopenblas-dev liblapack-dev pkg-config python3-pip python3-dev

        Alternatively, you can do::

          sudo apt build-dep numpy

        This command installs whatever is needed to build NumPy, with the
        advantage that new dependencies or updates to required versions are
        handled by the package managers.

      .. tab-item:: Fedora

        To install NumPy build requirements, you can do::

          sudo dnf install gcc-gfortran python3-devel openblas-devel lapack-devel pkgconfig

        Alternatively, you can do::

          sudo dnf builddep numpy

        This command installs whatever is needed to build NumPy, with the
        advantage that new dependencies or updates to required versions are
        handled by the package managers.

      .. tab-item:: CentOS/RHEL

        To install NumPy build requirements, you can do::

          sudo yum install gcc-gfortran python3-devel openblas-devel lapack-devel pkgconfig

        Alternatively, you can do::

          sudo yum-builddep numpy

        This command installs whatever is needed to build NumPy, with the
        advantage that new dependencies or updates to required versions are
        handled by the package managers.

      .. tab-item:: Arch

        To install NumPy build requirements, you can do::

          sudo pacman -S gcc-fortran openblas pkgconf

  .. tab-item:: macOS
    :sync: macos

    Install Apple Developer Tools. An easy way to do this is to
    `open a terminal window <https://blog.teamtreehouse.com/introduction-to-the-mac-os-x-command-line>`_,
    enter the command::

        xcode-select --install

    and follow the prompts. Apple Developer Tools includes Git, the Clang C/C++
    compilers, and other development utilities that may be required.

    Do *not* use the macOS system Python. Instead, install Python
    with `the python.org installer <https://www.python.org/downloads/>`__ or
    with a package manager like Homebrew, MacPorts or Fink.

    On macOS >=13.3, the easiest build option is to use Accelerate, which is
    already installed and will be automatically used by default.

    On older macOS versions you need a different BLAS library, most likely
    OpenBLAS, plus pkg-config to detect OpenBLAS. These are easiest to install
    with `Homebrew <https://brew.sh/>`__::

        brew install openblas pkg-config gfortran

  .. tab-item:: Windows
    :sync: windows

    On Windows, the use of a Fortran compiler is more tricky than on other
    platforms, because MSVC does not support Fortran, and gfortran and MSVC
    can't be used together. If you don't need to run the ``f2py`` tests, simply
    using MSVC is easiest. Otherwise, you will need one of these sets of
    compilers:

    1. MSVC + Intel Fortran (``ifort``)
    2. Intel compilers (``icc``, ``ifort``)
    3. Mingw-w64 compilers (``gcc``, ``g++``, ``gfortran``)

    Compared to macOS and Linux, building NumPy on Windows is a little more
    difficult, due to the need to set up these compilers. It is not possible to
    just call a one-liner on the command prompt as you would on other
    platforms.

    First, install Microsoft Visual Studio - the 2019 Community Edition or any
    newer version will work (see the
    `Visual Studio download site <https://visualstudio.microsoft.com/downloads/>`__).
    This is needed even if you use the MinGW-w64 or Intel compilers, in order
    to ensure you have the Windows Universal C Runtime (the other components of
    Visual Studio are not needed when using Mingw-w64, and can be deselected if
    desired, to save disk space). The recommended version of the UCRT is
    >= 10.0.22621.0.

    .. tab-set::

      .. tab-item:: MSVC

        The MSVC installer does not put the compilers on the system path, and
        the install location may change. To query the install location, MSVC
        comes with a ``vswhere.exe`` command-line utility. And to make the
        C/C++ compilers available inside the shell you are using, you need to
        run a ``.bat`` file for the correct bitness and architecture (e.g., for
        64-bit Intel CPUs, use ``vcvars64.bat``).

        If using a Conda environment while a version of Visual Studio 2019+ is
        installed that includes the MSVC v142 package (VS 2019 C++ x86/x64
        build tools), activating the conda environment should cause Visual
        Studio to be found and the appropriate .bat file executed to set
        these variables.

        For detailed guidance, see `Use the Microsoft C++ toolset from the command line
        <https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170>`__.

      .. tab-item:: Intel

        Similar to MSVC, the Intel compilers are designed to be used with an
        activation script (``Intel\oneAPI\setvars.bat``) that you run in the
        shell you are using. This makes the compilers available on the path.
        For detailed guidance, see
        `Get Started with the Intel® oneAPI HPC Toolkit for Windows
        <https://www.intel.com/content/www/us/en/docs/oneapi-hpc-toolkit/get-started-guide-windows/2023-1/overview.html>`__.

      .. tab-item:: MinGW-w64

        There are several sources of binaries for MinGW-w64. We recommend the
        RTools versions, which can be installed with Chocolatey (see
        Chocolatey install instructions `here <https://chocolatey.org/install>`_)::

            choco install rtools -y --no-progress --force --version=4.0.0.20220206

    .. note::

        Compilers should be on the system path (i.e., the ``PATH`` environment
        variable should contain the directory in which the compiler executables
        can be found) in order to be found, with the exception of MSVC which
        will be found automatically if and only if there are no other compilers
        on the ``PATH``. You can use any shell (e.g., Powershell, ``cmd`` or
        Git Bash) to invoke a build. To check that this is the case, try
        invoking a Fortran compiler in the shell you use (e.g., ``gfortran
        --version`` or ``ifort --version``).

    .. warning::

        When using a conda environment it is possible that the environment
        creation will not work due to an outdated Fortran compiler. If that
        happens, remove the ``compilers`` entry from ``environment.yml`` and
        try again. The Fortran compiler should be installed as described in
        this section.

  .. tab-item:: Windows on ARM64
    :sync: Windows on ARM64

    In Windows on ARM64, the set of a compiler options that are available for
    building NumPy are limited. Compilers such as GCC and GFortran are not yet
    supported for Windows on ARM64. Currently, the NumPy build for Windows on ARM64
    is supported with MSVC and LLVM toolchains. The use of a Fortran compiler is
    more tricky than on other platforms, because MSVC does not support Fortran, and
    gfortran and MSVC can't be used together. If you don't need to run the ``f2py``
    tests, simply using MSVC is easiest. Otherwise, you will need the following
    set of compilers:

    1. MSVC + flang (``cl``, ``flang``)
    2. LLVM + flang (``clang-cl``, ``flang``)

    First, install Microsoft Visual Studio - the 2022 Community Edition will
    work (see the `Visual Studio download site <https://visualstudio.microsoft.com/downloads/>`__).
    Ensure that you have installed necessary Visual Studio components for building NumPy 
    on WoA from `here <https://gist.github.com/Mugundanmcw/c3bb93018d5da9311fb2b222f205ba19>`__.

    To use the flang compiler for Windows on ARM64, install Latest LLVM
    toolchain for WoA from `here <https://github.com/llvm/llvm-project/releases>`__.

    .. tab-set::

      .. tab-item:: MSVC

        The MSVC installer does not put the compilers on the system path, and
        the install location may change. To query the install location, MSVC
        comes with a ``vswhere.exe`` command-line utility. And to make the
        C/C++ compilers available inside the shell you are using, you need to
        run a ``.bat`` file for the correct bitness and architecture (e.g., for
        ARM64-based CPUs, use ``vcvarsarm64.bat``).

        For detailed guidance, see `Use the Microsoft C++ toolset from the command line
        <https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170>`__.

      .. tab-item:: LLVM

        Similar to MSVC, LLVM does not put the compilers on the system path.
        To set system path for LLVM compilers, users may need to use ``set``
        command to put compilers on the system path. To check compiler's path
        for LLVM's clang-cl, try invoking LLVM's clang-cl compiler in the shell you use
        (``clang-cl --version``).

    .. note::

        Compilers should be on the system path (i.e., the ``PATH`` environment
        variable should contain the directory in which the compiler executables
        can be found) in order to be found, with the exception of MSVC which
        will be found automatically if and only if there are no other compilers
        on the ``PATH``. You can use any shell (e.g., Powershell, ``cmd`` or
        Git Bash) to invoke a build. To check that this is the case, try
        invoking a Fortran compiler in the shell you use (e.g., ``flang
        --version``).

    .. warning::

        Currently, Conda environment is not yet supported officially on `Windows
        on ARM64 <https://github.com/conda-forge/conda-forge.github.io/issues/1940>`__.
        The present approach uses virtualenv for building NumPy from source on
        Windows on ARM64.

Building NumPy from source
--------------------------

If you want to only install NumPy from source once and not do any development
work, then the recommended way to build and install is to use ``pip``.
Otherwise, conda is recommended.

.. note::

    If you don't have a conda installation yet, we recommend using
    Miniforge_; any conda flavor will work though.

Building from source to use NumPy
`````````````````````````````````

.. tab-set::

  .. tab-item:: Conda env
    :sync: conda

    If you are using a conda environment, ``pip`` is still the tool you use to
    invoke a from-source build of NumPy. It is important to always use the
    ``--no-build-isolation`` flag to the ``pip install`` command, to avoid
    building against a ``numpy`` wheel from PyPI. In order for that to work you
    must first install the remaining build dependencies into the conda
    environment::

      # Either install all NumPy dev dependencies into a fresh conda environment
      mamba env create -f environment.yml

      # Or, install only the required build dependencies
      mamba install python numpy cython compilers openblas meson-python pkg-config

      # To build the latest stable release:
      pip install numpy --no-build-isolation --no-binary numpy

      # To build a development version, you need a local clone of the NumPy git repository:
      git clone https://github.com/numpy/numpy.git
      cd numpy
      git submodule update --init
      pip install . --no-build-isolation

    .. warning::

        On Windows, the AR, LD, and LDFLAGS environment variables may be set,
        which will cause the pip install command to fail. These variables are only
        needed for flang and can be safely unset prior to running pip install.

  .. tab-item:: Virtual env or system Python
    :sync: pip

    ::

      # To build the latest stable release:
      pip install numpy --no-binary numpy

      # To build a development version, you need a local clone of the NumPy git repository:
      git clone https://github.com/numpy/numpy.git
      cd numpy
      git submodule update --init
      pip install .



.. _the-spin-interface:

Building from source for NumPy development
``````````````````````````````````````````

If you want to build from source in order to work on NumPy itself, first clone
the NumPy repository::

      git clone https://github.com/numpy/numpy.git
      cd numpy
      git submodule update --init

Then you want to do the following:

1. Create a dedicated development environment (virtual environment or conda
   environment),
2. Install all needed dependencies (*build*, and also *test*, *doc* and
   *optional* dependencies),
3. Build NumPy with the ``spin`` developer interface.

Step (3) is always the same, steps (1) and (2) are different between conda and
virtual environments:

.. tab-set::

  .. tab-item:: Conda env
    :sync: conda

    To create a ``numpy-dev`` development environment with every required and
    optional dependency installed, run::

        mamba env create -f environment.yml
        mamba activate numpy-dev

  .. tab-item:: Virtual env or system Python
    :sync: pip

    .. note::

       There are many tools to manage virtual environments, like ``venv``,
       ``virtualenv``/``virtualenvwrapper``, ``pyenv``/``pyenv-virtualenv``,
       Poetry, PDM, Hatch, and more. Here we use the basic ``venv`` tool that
       is part of the Python stdlib. You can use any other tool; all we need is
       an activated Python environment.

    Create and activate a virtual environment in a new directory named ``venv`` (
    note that the exact activation command may be different based on your OS and shell
    - see `"How venvs work" <https://docs.python.org/3/library/venv.html#how-venvs-work>`__
    in the ``venv`` docs).

    .. tab-set::

      .. tab-item:: Linux
        :sync: linux

        ::

          python -m venv venv
          source venv/bin/activate

      .. tab-item:: macOS
        :sync: macos

        ::

          python -m venv venv
          source venv/bin/activate

      .. tab-item:: Windows
        :sync: windows

        ::

          python -m venv venv
          .\venv\Scripts\activate

      .. tab-item:: Windows on ARM64
        :sync: Windows on ARM64

        ::

          python -m venv venv
          .\venv\Scripts\activate

        .. note::

          Building NumPy with BLAS and LAPACK functions requires OpenBLAS
          library at Runtime. In Windows on ARM64, this can be done by setting
          up pkg-config for OpenBLAS dependency. The build steps for OpenBLAS
          for Windows on ARM64 can be found `here <http://www.openmathlib.org/OpenBLAS/docs/install/#windows-on-arm>`__.


    Then install the Python-level dependencies from PyPI with::

       python -m pip install -r requirements/build_requirements.txt

To build NumPy in an activated development environment, run::

    spin build

This will install NumPy inside the repository (by default in a
``build-install`` directory). You can then run tests (``spin test``),
drop into IPython (``spin ipython``), or take other development steps
like build the html documentation or running benchmarks. The ``spin``
interface is self-documenting, so please see ``spin --help`` and
``spin <subcommand> --help`` for detailed guidance.

.. warning::

    In an activated conda environment on Windows, the AR, LD, and LDFLAGS
    environment variables may be set, which will cause the build to fail.
    These variables are only needed for flang and can be safely unset
    for build.

.. _meson-editable-installs:

.. admonition:: IDE support & editable installs

    While the ``spin`` interface is our recommended way of working on NumPy,
    it has one limitation: because of the custom install location, NumPy
    installed using ``spin`` will not be recognized automatically within an
    IDE (e.g., for running a script via a "run" button, or setting breakpoints
    visually). This will work better with an *in-place build* (or "editable
    install").

    Editable installs are supported. It is important to understand that **you
    may use either an editable install or ``spin`` in a given repository clone,
    but not both**. If you use editable installs, you have to use ``pytest``
    and other development tools directly instead of using ``spin``.

    To use an editable install, ensure you start from a clean repository (run
    ``git clean -xdf`` if you've built with ``spin`` before) and have all
    dependencies set up correctly as described higher up on this page. Then
    do::

        # Note: the --no-build-isolation is important!
        pip install -e . --no-build-isolation

        # To run the tests for, e.g., the `numpy.linalg` module:
        pytest numpy/linalg

    When making changes to NumPy code, including to compiled code, there is no
    need to manually rebuild or reinstall. NumPy is automatically rebuilt each
    time NumPy is imported by the Python interpreter; see the meson-python_
    documentation on editable installs for more details on how that works under
    the hood.

    When you run ``git clean -xdf``, which removes the built extension modules,
    remember to also uninstall NumPy with ``pip uninstall numpy``.


    .. warning::

        Note that editable installs are fundamentally incomplete installs.
        Their only guarantee is that ``import numpy`` works - so they are
        suitable for working on NumPy itself, and for working on pure Python
        packages that depend on NumPy. Headers, entrypoints, and other such
        things may not be available from an editable install.


Customizing builds
------------------

.. toctree::
   :maxdepth: 1

   compilers_and_options
   blas_lapack
   cross_compilation
   redistributable_binaries


Background information
----------------------

.. toctree::
   :maxdepth: 1

   understanding_meson
   introspecting_a_build
   distutils_equivalents


.. _Miniforge: https://github.com/conda-forge/miniforge
.. _meson-python: https://mesonbuild.com/meson-python/
