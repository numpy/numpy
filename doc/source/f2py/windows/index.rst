.. _f2py-windows:

=================
F2PY and Windows
=================

.. warning::

   F2PY support for Windows is not always at par with Linux support

.. note::
   `SciPy's documentation`_ has some information on system-level dependencies
   which are well tested for Fortran as well.

Broadly speaking, there are two issues working with F2PY on Windows:

- the lack of actively developed FOSS Fortran compilers, and,
- the linking issues related to the C runtime library for building Python-C extensions.

The focus of this section is to establish a guideline for developing and
extending Fortran modules for Python natively, via F2PY on Windows.

Currently supported toolchains are:

- Mingw-w64 C/C++/Fortran compilers
- Intel compilers
- Clang-cl + Flang
- MSVC + Flang

Overview
========

From a user perspective, the most UNIX compatible Windows
development environment is through emulation, either via the Windows Subsystem
on Linux, or facilitated by Docker. In a similar vein, traditional
virtualization methods like VirtualBox are also reasonable methods to develop
UNIX tools on Windows.

Native Windows support is typically stunted beyond the usage of commercial compilers.
However, as of 2022, most commercial compilers have free plans which are sufficient for
general use. Additionally, the Fortran language features supported by ``f2py``
(partial coverage of Fortran 2003), means that newer toolchains are often not
required. Briefly, then, for an end user, in order of use:

Classic Intel Compilers (commercial)
   These are maintained actively, though licensing restrictions may apply as
   further detailed in :ref:`f2py-win-intel`.

   Suitable for general use for those building native Windows programs by
   building off of MSVC.

MSYS2 (FOSS)
   In conjunction with the ``mingw-w64`` project, ``gfortran`` and ``gcc``
   toolchains can be used to natively build Windows programs.

Windows Subsystem for Linux
   Assuming the usage of ``gfortran``, this can be used for cross-compiling
   Windows applications, but is significantly more complicated.

Conda
   Windows support for compilers in ``conda`` is facilitated by pulling MSYS2
   binaries, however these `are outdated`_, and therefore not recommended (as of 30-01-2022).

PGI Compilers (commercial)
   Unmaintained but sufficient if an existing license is present. Works
   natively, but has been superseded by the Nvidia HPC SDK, with no `native
   Windows support`_.

Cygwin (FOSS)
   Can also be used for ``gfortran``. However, the POSIX API compatibility layer provided by
   Cygwin is meant to compile UNIX software on Windows, instead of building
   native Windows programs. This means cross compilation is required.

The compilation suites described so far are compatible with the `now
deprecated`_ ``np.distutils`` build backend which is exposed by the F2PY CLI.
Additional build system usage (``meson``, ``cmake``) as described in
:ref:`f2py-bldsys` allows for a more flexible set of compiler
backends including:

Intel oneAPI
   The newer Intel compilers (``ifx``, ``icx``) are based on LLVM and can be
   used for native compilation. Licensing requirements can be onerous.

Classic Flang (FOSS)
   The backbone of the PGI compilers were cannibalized to form the "classic" or
   `legacy version of Flang`_. This may be compiled from source and used
   natively. `LLVM Flang`_ does not support Windows yet (30-01-2022).
   
LFortran (FOSS)
   One of two LLVM based compilers. Not all of F2PY supported Fortran can be
   compiled yet (30-01-2022) but uses MSVC for native linking.


Baseline
========

For this document we will assume the following basic tools:

- The IDE being considered is the community supported `Microsoft Visual Studio Code`_
- The terminal being used is the `Windows Terminal`_
- The shell environment is assumed to be `Powershell 7.x`_
- Python 3.10 from `the Microsoft Store`_ and this can be tested with
   ``Get-Command python.exe`` resolving to
   ``C:\Users\$USERNAME\AppData\Local\Microsoft\WindowsApps\python.exe``
- The Microsoft Visual C++ (MSVC) toolset

With this baseline configuration, we will further consider a configuration
matrix as follows:

.. _table-f2py-winsup-mat:

.. table:: Support matrix, exe implies a Windows installer 

  +----------------------+--------------------+-------------------+
  | **Fortran Compiler** | **C/C++ Compiler** | **Source**        |
  +======================+====================+===================+
  | Intel Fortran        | MSVC / ICC         | exe               |
  +----------------------+--------------------+-------------------+
  | GFortran             | MSVC               | MSYS2/exe         |
  +----------------------+--------------------+-------------------+
  | GFortran             | GCC                | WSL               |
  +----------------------+--------------------+-------------------+
  | Classic Flang        | MSVC               | Source / Conda    |
  +----------------------+--------------------+-------------------+
  | Anaconda GFortran    | Anaconda GCC       | exe               |
  +----------------------+--------------------+-------------------+

For an understanding of the key issues motivating the need for such a matrix
`Pauli Virtanen's in-depth post on wheels with Fortran for Windows`_ is an
excellent resource. An entertaining explanation of an application binary
interface (ABI) can be found in this post by `JeanHeyd Meneide`_. 

PowerShell and MSVC
====================

MSVC is installed either via the Visual Studio Bundle or the lighter (preferred)
`Build Tools for Visual Studio`_ with the ``Desktop development with C++``
setting.

.. note::
   
  This can take a significant amount of time as it includes a download of around
  2GB and requires a restart.

It is possible to use the resulting environment from a `standard command
prompt`_. However, it is more pleasant to use a `developer powershell`_,
with a `profile in Windows Terminal`_. This can be achieved by adding the
following block to the ``profiles->list`` section of the JSON file used to 
configure Windows Terminal (see ``Settings->Open JSON file``):

.. code-block:: json

  {
  "name": "Developer PowerShell for VS 2019",
  "commandline": "powershell.exe -noe -c \"$vsPath = (Join-Path ${env:ProgramFiles(x86)} -ChildPath 'Microsoft Visual Studio\\2019\\BuildTools'); Import-Module (Join-Path $vsPath 'Common7\\Tools\\Microsoft.VisualStudio.DevShell.dll'); Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation\"",
  "icon": "ms-appx:///ProfileIcons/{61c54bbd-c2c6-5271-96e7-009a87ff44bf}.png"
  }

Now, testing the compiler toolchain could look like:

.. code-block:: powershell

   # New Windows Developer Powershell instance / tab
   # or
   $vsPath = (Join-Path ${env:ProgramFiles(x86)} -ChildPath 'Microsoft Visual Studio\\2019\\BuildTools'); 
   Import-Module (Join-Path $vsPath 'Common7\\Tools\\Microsoft.VisualStudio.DevShell.dll');
   Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation
   **********************************************************************
   ** Visual Studio 2019 Developer PowerShell v16.11.9
   ** Copyright (c) 2021 Microsoft Corporation
   **********************************************************************
   cd $HOME
   echo "#include<stdio.h>" > blah.cpp; echo 'int main(){printf("Hi");return 1;}' >> blah.cpp
   cl blah.cpp
  .\blah.exe
   # Hi
   rm blah.cpp

It is also possible to check that the environment has been updated correctly
with ``$ENV:PATH``.


Microsoft Store Python paths
============================

The MS Windows version of Python discussed here installs to a non-deterministic
path using a hash. This needs to be added to the ``PATH`` variable.

.. code-block:: powershell

   $Env:Path += ";$env:LOCALAPPDATA\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\scripts"

.. toctree::
   :maxdepth: 2

   intel
   msys2
   conda
   pgi


.. _the Microsoft Store: https://www.microsoft.com/en-us/p/python-310/9pjpw5ldxlz5
.. _Microsoft Visual Studio Code: https://code.visualstudio.com/Download
.. _more complete POSIX environment: https://www.cygwin.com/
.. _This MSYS2 document: https://www.msys2.org/wiki/How-does-MSYS2-differ-from-Cygwin/
.. _Build Tools for Visual Studio: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019
.. _Windows Terminal: https://www.microsoft.com/en-us/p/windows-terminal/9n0dx20hk701?activetab=pivot:overviewtab
.. _Powershell 7.x: https://docs.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-windows?view=powershell-7.1
.. _standard command prompt: https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-160#developer_command_file_locations
.. _developer powershell: https://docs.microsoft.com/en-us/visualstudio/ide/reference/command-prompt-powershell?view=vs-2019
.. _profile in Windows Terminal: https://techcommunity.microsoft.com/t5/microsoft-365-pnp-blog/add-developer-powershell-and-developer-command-prompt-for-visual/ba-p/2243078
.. _Pauli Virtanen's in-depth post on wheels with Fortran for Windows: https://pav.iki.fi/blog/2017-10-08/pywingfortran.html#building-python-wheels-with-fortran-for-windows
.. _Nvidia HPC SDK: https://www.pgroup.com/index.html
.. _JeanHeyd Meneide: https://thephd.dev/binary-banshees-digital-demons-abi-c-c++-help-me-god-please
.. _legacy version of Flang: https://github.com/flang-compiler/flang
.. _native Windows support: https://developer.nvidia.com/nvidia-hpc-sdk-downloads#collapseFour
.. _are outdated: https://github.com/conda-forge/conda-forge.github.io/issues/1044
.. _now deprecated: https://github.com/numpy/numpy/pull/20875
.. _LLVM Flang: https://releases.llvm.org/11.0.0/tools/flang/docs/ReleaseNotes.html
.. _SciPy's documentation: https://scipy.github.io/devdocs/building/index.html#system-level-dependencies
