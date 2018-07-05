=================================================================================
NEP 2 — A proposal to build numpy without warning with a big set of warning flags
=================================================================================

:Author: David Cournapeau
:Contact: david@ar.media.kyoto-u.ac.jp
:Date: 2008-09-04
:Status: Deferred

Executive summary
=================

When building numpy and scipy, we are limited to a quite restricted set of
warning compilers, thus missing a large class of potential bugs which could be
detected with stronger warning flags. The goal of this NEP is present the
various methods used to clean the code and implement some policy to make numpy
buildable with a  bigger set of warning flags, while keeping the build warnings
free.

Warning flags
=============

Each compiler detects a different set of potential errors. The baseline will
be gcc -Wall -W -Wextra. Ideally, a complete set would be nice::

  -W -Wall -Wextra -Wstrict-prototypes -Wmissing-prototypes -Waggregate-return
  -Wcast-align -Wcast-qual -Wnested-externs -Wshadow -Wbad-function-cast
  -Wwrite-strings "

Intel compiler, VS with ``/W3 /Wall``, Sun compilers have extra warnings too.

Kind of warnings
================

C Python extension code tends to naturally generate a lot of spurious warnings.
The goal is to have some facilities to tag some typical C-Python code so that
the compilers do not generate warnings in those cases; the tag process has to
be clean, readable, and be robust. In particular, it should not make the code
more obscure or worse, break working code.

unused parameter
----------------

This one appears often: any python-callable C function takes two arguments,
of which the first is not used for functions (only for methods). One way to
solve it is to tag the function argument with a macro NPY_UNUSED. This macro
uses compiler specific code to tag the variable, and mangle it such as it is
not possible to use it accidentally once it is tagged.

The code to apply compiler specific option could be::

  #if defined(__GNUC__)
          #define __COMP_NPY_UNUSED __attribute__ ((__unused__))
  # elif defined(__ICC)
          #define __COMP_NPY_UNUSED __attribute__ ((__unused__))
  #else
          #define __COMP_NPY_UNUSED
  #endif

The variable mangling would be::

  #define NPY_UNUSED(x) (__NPY_UNUSED_TAGGED ## x) __COMP_NPY_UNUSED

When applied to a variable, one would get::

  int foo(int * NPY_UNUSED(dummy))

expanded to

::

   int foo(int * __NPY_UNUSED_TAGGEDdummy __COMP_NPY_UNUSED)

Thus avoiding any accidental use of the variable. The mangling is pure C, and
thuse portable. The per-variable warning disabling is compiler specific.

signed/unsigned comparison
--------------------------

More tricky: not always clear what to do

half-initialized structures
---------------------------

Just put the elements with NULL in it.
