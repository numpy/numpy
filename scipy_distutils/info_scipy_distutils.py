"""
scipy_distutils
===============

Modified version of distutils to handle fortran source code, f2py,
and other issues in the scipy build process.

Useful standalone modules
-------------------------

  system_info -- get information about system resources
  cpuinfo -- query CPU information
  from_template -- process Fortran or signature template files
  exec_command -- provides highly portable commands.getstatusoutput

Modules to support Fortran compilers
------------------------------------

  fcompiler -- base class for Fortran compiler abstraction model
  absoftfcompiler -- Absoft Fortran 77/90 compiler
  compaqfcompiler -- Compaq Fortran 77/90 compiler
  gnufcompiler -- GNU Fortran 77 compiler
  hpuxfcompiler -- HPUX Fortran 90 compiler
  ibmfcompiler -- IBM XL Fortran 90/95 compiler
  intelfcompiler -- Intel Fortran 90 compilers
  laheyfcompiler -- Lahey/Fujitsu Fortran 95 compiler
  mipsfcompiler -- MIPSpro Fortran 77/90 compilers
  nagfcompiler -- NAGWare Fortran 95 compiler
  pgfcompiler -- Portland Group 77/90 compilers
  sunfcompiler -- Sun|Forte Developer|WorkShop 90 compilers
  vastfcompile -- Pacific-Sierra Research Fortran 90 compiler

Modules extending distutils
---------------------------

  misc_util -- various useful tools for scipy-style setup.py files.
  lib2def -- generate DEF from MSVC-compiled DLL
  mingw32ccompiler -- MingW32 compiler compatible with an MSVC built Python
  command.config_compiler -- support enhancing compiler flags
  command.build_src -- build swig, f2py, weave, callback sources

"""
standalone = 1
