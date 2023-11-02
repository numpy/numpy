.. _f2py-win-msys2:

===========================
F2PY and Windows with MSYS2
===========================

Follow the standard `installation instructions`_. Then, to grab the requisite Fortran compiler with ``MVSC``:

.. code-block:: bash

   # Assuming a fresh install
   pacman -Syu # Restart the terminal
   pacman -Su  # Update packages
   # Get the toolchains
   pacman -S --needed base-devel gcc-fortran
   pacman -S mingw-w64-x86_64-toolchain


.. _`installation instructions`: https://www.msys2.org/
