This directory contains various scripts and code to build binaries installers for
windows.

It can:
        - prepare a bootstrap environment to build binary in a self-contained
          directory
        - build binaries for different architectures using different site.cfg
        - prepare a nsis-based installer which automatically detects the arch
          on the computer where numpy is installed.

Example:
========

python doall.py

Should build the numpy 'super' installer for sse2, sse3 and nosse from scratch.
You have to run it in the win32build directory.

Dependencies:
=============

You need the following to use those scripts:
        - python and mingw tools (gcc, make, g77 at least).
        - the binaries for atlas/blas/lapack for the various archs supported
          (see vendor in numpy repository root for tools to build those).
        - python, nsis and subversion command line tools should be in your
          PATH, e.g. running python, makensis and svn should work in a DOS
          cmd.exe.
        - the CpuCaps nsis plugin (see below on how to build it).

Components:
===========

cpuid
-----

cpuid: contains a mini C lib to detect SSE variants (SSE 1, 2 and 3 for now).
It relies on gcc ASM, but porting it to VS should be trivial (only a few lines
os ASM).

cpucaps:
--------

cpucaps: nsis plugin to add the ability to detect SSE for installers, uses
cpuid. To build it, you have two options:
        - build it manually: build the CpuCaps.dll with sources cpucaps.c and
          cpuid.c in cpuid directory.
        - with scons: if you have scons, just do scons install. It will build
          and put the CpuCaps.dll  in the plugins directory of nsis (if you
          install nsis in the default path).

build.py:
---------

Can build the binaries for each variant of arch in a bootstrap environment

prepare_bootstrap.py
--------------------

Script to prepare a bootstrap environment. A bootstrap environment depends on
the python version (2.5, 2.4, etc...).

It works by building a source distribution, unzipping it in a bootrap
directory, and putting everything (build.py, nsis script, etc...) in it.
