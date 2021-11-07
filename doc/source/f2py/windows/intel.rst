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

We will consider the classic example of the generation of Fibonnaci numbers,
``fib1.f``, given by:

.. literalinclude:: ./code/fib1.f
   :language: fortran

For ``cmd.exe`` fans, using the Intel oneAPI command prompt is the easiest approach, as
it loads the required environment for both ``ifort`` and ``msvc``. Helper batch
scripts are also provided.

.. code::
   :language: bash

   # cmd.exe
   "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
   python -m numpy.f2py -c fib1.f -m fib1
   python -c "import fib1; import numpy as np; a=np.zeros(8); fib1.fib(a); print(a)"

Powershell usage is a little less pleasant.

.. code::
   :language: bash

   
This configuration now works with MSVC as:

.. code::
  :language: bash

   # Powershell
   python -m numpy.f2py -c fib1.f -m fib1 --f77exec='C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\bin\intel64\ifort.exe' --f90exec='C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\bin\intel64\ifort.exe' -L'C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\compiler\lib\ia32'
   python -c "import fib1; import numpy as np; a=np.zeros(8); fib1.fib(a); print(a)"
   # Alternatively, set environment and reload Powershell
   cmd.exe /k "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" 
   powershell
   python -m numpy.f2py -c fib1.f -m fib1
   python -c "import fib1; import numpy as np; a=np.zeros(8); fib1.fib(a); print(a)"



.. _have been relaxed: https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-fortran-compiler-release-notes.html
.. _disassembly of components and liability: https://software.sintel.com/content/www/us/en/develop/articles/end-user-license-agreement.html
.. _Intel Fortran Compilers: https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#inpage-nav-6-1
.. _Classic Intel C/C++ Compiler: https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#inpage-nav-6-undefined