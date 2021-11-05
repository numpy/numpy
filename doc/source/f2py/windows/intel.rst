.. _f2py-win-intel:

=================================
F2PY and Windows Intel Fortran
=================================

At this time, only the classic Intel compilers (``ifort``) are supported.

.. note::

	The licensing restrictions for beta software `have been relaxed`_ during
	the transition to the LLVM backed ``ifx/icc`` family of compilers.
	However this document does not endorse the usage of Intel in downstream
	projects due to the issues pertaining to `disassembly of components and
	liability`_.
	
	Neither the Python Intel installation nor the `Classic Intel C/C++
	Compiler` are required.

- The `Intel Fortran Compilers`_ come in a combined installer providing both
Classic and Beta versions; these also take around a gigabyte and a half or so

This configuration now works with MSVC as:

.. code::
  :language: bash

   python -m numpy.f2py -c fib1.f -m fib1 --f77exec='C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\bin\intel64\ifort.exe' --f90exec='C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\bin\intel64\ifort.exe' -L'C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\compiler\lib\ia32'
    

.. _have been relaxed: https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-fortran-compiler-release-notes.html
.. _disassembly of components and liability: https://software.intel.com/content/www/us/en/develop/articles/end-user-license-agreement.html
.. _Intel Fortran Compilers: https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#inpage-nav-6-1
.. _Classic Intel C/C++ Compiler: https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#inpage-nav-6-undefined