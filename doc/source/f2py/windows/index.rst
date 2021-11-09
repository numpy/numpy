.. _f2py-windows:

=================
F2PY and Windows
=================

.. note::

	F2PY support for Windows is not at par with Linux support, and 
	OS specific flags can be seen via ``python -m numpy.f2py``

Broadly speaking, there are two issues working with F2PY on Windows: the lack of actively
developed FOSS Fortran compilers, and the linking issues related to the C
runtime library for building Python-C extensions.

The focus of this section is to establish a guideline for developing and
extending Fortran modules for Python natively, via F2PY on Windows. 
For this document we will asume the following basic tools:

- The IDE being considered is the community supported `Microsoft Visual Studio Code`_
- The shell environment is assumed to be `Powershell 7.x`_
- Python 3.9 from `the Microsoft Store`_ and this can be tested with
   ``Get-Command python.exe`` resolving to
   ``C:\Users\$USERNAME\AppData\Local\Microsoft\WindowsApps\python.exe``
- The  Microsoft Visual C++ (MSVC) toolset

With this baseline configuration, we will further consider a configuration
matrix as follows:

.. _table-f2py-winsup-mat:

.. table:: Support matrix, exe implies a Windows installer 

  +----------------------+--------------------+------------+
  | **Fortran Compiler** | **C/C++ Compiler** | **Source** |
  +======================+====================+============+
  | GFortran             | MSVC               | MSYS2/exe  |
  +----------------------+--------------------+------------+
  | GFortran             | GCC                | MSYS2      |
  +----------------------+--------------------+------------+
  | GFortran             | GCC                | WSL        |
  +----------------------+--------------------+------------+
  | GFortran             | GCC                | Cygwin     |
  +----------------------+--------------------+------------+
  | Classic Flang        | MSVC               | Source     |
  +----------------------+--------------------+------------+
  | Intel Fortran        | MSVC               | exe        |
  +----------------------+--------------------+------------+
  | Anaconda GFortran    | Anaconda GCC       | exe        |
  +----------------------+--------------------+------------+

`This MSYS2 document`_ covers details of differences between MSYS2 and Cygwin.
Broadly speaking, MSYS2 is geared towards building native Windows, while Cygwin
is closer to providing a `more complete POSIX environment`_. Since MSVC is a
core component of the Windows setup, its installation and the setup for the
Powershell environment are described below.

Powershell and MSVC
====================

MSVC is installed either via the Visual Studio Bundle or the lighter (preferred)
`Build Tools for Visual Studio`_ with the ``Desktop development with C++``
setting.

.. note::
   
  This can take a significant amount of time as it includes a download of around
  2GB and requires a restart.

Though it is possible use the resulting environment from a `standard command
prompt`_, it is more pleasant to use a `Powershell module like VCVars`_ which
exposes a much simpler ``set (vcvars)``. So this would essentially mean testing
the compiler toolchain could look like:

.. code:: bash

   # New Powershell instance
   set (vcvars)
   echo "#include<stdio.h>" > blah.cpp; echo 'int main(){printf("Hi");return 1;}' >> blah.cpp
   cl blah.cpp
   # Hello
   rm blah.cpp

It is also possible to check that the environment has been updated correctly
with ``$ENV:PATH``.

.. _the Microsoft Store: https://www.microsoft.com/en-us/p/python-39/9p7qfqmjrfp7?activetab=pivot:overviewtab
.. _Microsoft Visual Studio Code: https://code.visualstudio.com/Download
.. _more complete POSIX environment: https://www.cygwin.com/
.. _This MSYS2 document: https://www.msys2.org/wiki/How-does-MSYS2-differ-from-Cygwin/
.. _Build Tools for Visual Studio: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019
.. _Powershell 7.x: https://docs.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-windows?view=powershell-7.1
.. _standard command prompt: https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-160#developer_command_file_locations
.. _Powershell module like VCVars: https://github.com/bruxisma/VCVars

.. toctree::
   :maxdepth: 2

   intel
