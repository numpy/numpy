.. -*- rest -*-

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 F2PY: Fortran to Python interface generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Author: Pearu Peterson <pearu@cens.ioc.ee>
:License: NumPy License
:Web-site: http://cens.ioc.ee/projects/f2py2e/
:Discussions to: `f2py-users mailing list`_
:Documentation: `User's Guide`__, FAQ__
:Platforms: All
:Date: $Date: 2005/01/30 18:54:53 $

.. _f2py-users mailing list: http://cens.ioc.ee/mailman/listinfo/f2py-users/
__ usersguide/index.html
__ FAQ.html

-------------------------------

.. topic:: NEWS!!!

  January 5, 2006

    WARNING -- these notes are out of date! The package structure for NumPy and
    SciPy has changed considerably.  Much of this information is now incorrect.

  January 30, 2005

    Latest F2PY release (version 2.45.241_1926).
    New features: wrapping unsigned integers, support for ``.pyf.src`` template files,
    callback arguments can now be CObjects, fortran objects, built-in functions.
    Introduced ``intent(aux)`` attribute. Wrapped objects have ``_cpointer``
    attribute holding C pointer to wrapped functions or variables.
    Many bug fixes and improvements, updated documentation.
    `Differences with the previous release (version 2.43.239_1831)`__.

  __ http://cens.ioc.ee/cgi-bin/cvsweb/python/f2py2e/docs/HISTORY.txt.diff?r1=1.163&r2=1.137&f=h

  October 4, 2004
    F2PY bug fix release (version 2.43.239_1831).
    Better support for 64-bit platforms.
    Introduced ``--help-link`` and ``--link-<resource>`` options.
    Bug fixes.
    `Differences with the previous release (version 2.43.239_1806)`__.

  __ http://cens.ioc.ee/cgi-bin/cvsweb/python/f2py2e/docs/HISTORY.txt.diff?r1=1.137&r2=1.131&f=h

  September 25, 2004
    Latest F2PY release (version 2.43.239_1806).
    Support for ``ENTRY`` statement. New attributes:
    ``intent(inplace)``, ``intent(callback)``. Supports Numarray 1.1.
    Introduced ``-*- fix -*-`` header content. Improved ``PARAMETER`` support.
    Documentation updates. `Differences with the previous release
    (version 2.39.235-1693)`__.

  __ http://cens.ioc.ee/cgi-bin/cvsweb/python/f2py2e/docs/HISTORY.txt.diff?r1=1.131&r2=1.98&f=h

  `History of NEWS`__

  __ OLDNEWS.html

-------------------------------

.. Contents::

==============
 Introduction
==============

The purpose of the F2PY --*Fortran to Python interface generator*--
project is to provide connection between Python_ and Fortran
languages. F2PY is a Python extension tool for creating Python C/API
modules from (handwritten or F2PY generated) signature files (or
directly from Fortran sources). The generated extension modules
facilitate:

* Calling Fortran 77/90/95, Fortran 90/95 module, and C functions from
  Python.

* Accessing Fortran 77 ``COMMON`` blocks and Fortran 90/95 module
  data (including allocatable arrays) from Python.

* Calling Python functions from Fortran or C (call-backs).

* Automatically handling the difference in the data storage order of
  multi-dimensional Fortran and Numerical Python (i.e. C) arrays.

In addition, F2PY can build the generated extension modules to shared
libraries with one command. F2PY uses the ``numpy_distutils`` module
from SciPy_ that supports number of major Fortran compilers.

..
  (see `COMPILERS.txt`_ for more information).

F2PY generated extension modules depend on NumPy_ package that
provides fast multi-dimensional array language facility to Python.


---------------
 Main features
---------------

Here follows a more detailed list of F2PY features:

* F2PY scans real Fortran codes to produce the so-called signature
  files (.pyf files). The signature files contain all the information
  (function names, arguments and their types, etc.)  that is needed to
  construct Python bindings to Fortran (or C) functions.

  The syntax of signature files is borrowed from the
  Fortran 90/95 language specification and has some F2PY specific
  extensions. The signature files can be modified to dictate how
  Fortran (or C) programs are called from Python:

    + F2PY solves dependencies between arguments (this is relevant for
      the order of initializing variables in extension modules).

    + Arguments can be specified to be optional or hidden that
      simplifies calling Fortran programs from Python considerably.

    + In principle, one can design any Python signature for a given
      Fortran function, e.g. change the order arguments, introduce
      auxiliary arguments, hide the arguments, process the arguments
      before passing to Fortran, return arguments as output of F2PY
      generated functions, etc.

* F2PY automatically generates __doc__ strings (and optionally LaTeX
  documentation) for extension modules.

* F2PY generated functions accept arbitrary (but sensible) Python
  objects as arguments. The F2PY interface automatically takes care of
  type-casting and handling of non-contiguous arrays.

* The following Fortran constructs are recognized by F2PY:

  + All basic Fortran types::

      integer[ | *1 | *2 | *4 | *8 ], logical[ | *1 | *2 | *4 | *8 ]
      integer*([ -1 | -2 | -4 | -8 ])
      character[ | *(*) | *1 | *2 | *3 | ... ]
      real[ | *4 | *8 | *16 ], double precision
      complex[ | *8 | *16 | *32 ]

    Negative ``integer`` kinds are used to wrap unsigned integers.

  + Multi-dimensional arrays of all basic types with the following
    dimension specifications::

      <dim> | <start>:<end> | * | :

  + Attributes and statements::

      intent([ in | inout | out | hide | in,out | inout,out | c |
               copy | cache | callback | inplace | aux ])
      dimension(<dimspec>)
      common, parameter
      allocatable
      optional, required, external
      depend([<names>])
      check([<C-booleanexpr>])
      note(<LaTeX text>)
      usercode, callstatement, callprotoargument, threadsafe, fortranname
      pymethoddef
      entry

* Because there are only little (and easily handleable) differences
  between calling C and Fortran functions from F2PY generated
  extension modules, then F2PY is also well suited for wrapping C
  libraries to Python.

* Practice has shown that F2PY generated interfaces (to C or Fortran
  functions) are less error prone and even more efficient than
  handwritten extension modules. The F2PY generated interfaces are
  easy to maintain and any future optimization of F2PY generated
  interfaces transparently apply to extension modules by just
  regenerating them with the latest version of F2PY.

* `F2PY Users Guide and Reference Manual`_


===============
 Prerequisites
===============

F2PY requires the following software installed:

* Python_ (versions 1.5.2 or later; 2.1 and up are recommended).
  You must have python-dev package installed.
* NumPy_ (versions 13 or later; 20.x, 21.x, 22.x, 23.x are recommended)
* Numarray_ (version 0.9 and up), optional, partial support.
* Scipy_distutils (version 0.2.2 and up are recommended) from SciPy_
  project. Get it from Scipy CVS or download it below.

Python 1.x users also need distutils_.

Of course, to build extension modules, you'll need also working C
and/or Fortran compilers installed.

==========
 Download
==========

You can download the sources for the latest F2PY and numpy_distutils
releases as:

* `2.x`__/`F2PY-2-latest.tar.gz`__
* `2.x`__/`numpy_distutils-latest.tar.gz`__

Windows users might be interested in Win32 installer for F2PY and
Scipy_distutils (these installers are built using Python 2.3):

* `2.x`__/`F2PY-2-latest.win32.exe`__
* `2.x`__/`numpy_distutils-latest.win32.exe`__

Older releases are also available in the directories
`rel-0.x`__, `rel-1.x`__, `rel-2.x`__, `rel-3.x`__, `rel-4.x`__, `rel-5.x`__,
if you need them.

.. __: 2.x/
.. __: 2.x/F2PY-2-latest.tar.gz
.. __: 2.x/
.. __: 2.x/numpy_distutils-latest.tar.gz
.. __: 2.x/
.. __: 2.x/F2PY-2-latest.win32.exe
.. __: 2.x/
.. __: 2.x/numpy_distutils-latest.win32.exe
.. __: rel-0.x
.. __: rel-1.x
.. __: rel-2.x
.. __: rel-3.x
.. __: rel-4.x
.. __: rel-5.x

Development version of F2PY from CVS is available as `f2py2e.tar.gz`__.

__ http://cens.ioc.ee/cgi-bin/viewcvs.cgi/python/f2py2e/f2py2e.tar.gz?tarball=1

Debian Sid users can simply install ``python-f2py`` package.

==============
 Installation
==============

Unpack the source file, change to directrory ``F2PY-?-???/`` and run
(you may need to become a root)::

  python setup.py install

The F2PY installation installs a Python package ``f2py2e`` to your
Python ``site-packages`` directory and a script ``f2py`` to your
Python executable path.

See also Installation__ section in `F2PY FAQ`_.

.. __: FAQ.html#installation

Similarly, to install ``numpy_distutils``, unpack its tar-ball and run::

  python setup.py install

=======
 Usage
=======

To check if F2PY is installed correctly, run
::

  f2py

without any arguments. This should print out the usage information of
the ``f2py`` program.

Next, try out the following three steps:

1) Create a Fortran file `hello.f`__ that contains::

    C File hello.f
          subroutine foo (a)
          integer a
          print*, "Hello from Fortran!"
          print*, "a=",a
          end

__ hello.f

2) Run

  ::

    f2py -c -m hello hello.f

  This will build an extension module ``hello.so`` (or ``hello.sl``,
  or ``hello.pyd``, etc. depending on your platform) into the current
  directory.

3) Now in Python try::

    >>> import hello
    >>> print hello.__doc__
    >>> print hello.foo.__doc__
    >>> hello.foo(4)
     Hello from Fortran!
     a= 4
    >>>

If the above works, then you can try out more thorough
`F2PY unit tests`__ and read the `F2PY Users Guide and Reference Manual`_.

__ FAQ.html#q-how-to-test-if-f2py-is-working-correctly

===============
 Documentation
===============

The documentation of the F2PY project is collected in ``f2py2e/docs/``
directory. It contains the following documents:

`README.txt`_ (in CVS__)
  The first thing to read about F2PY -- this document.

__ http://cens.ioc.ee/cgi-bin/cvsweb/python/f2py2e/docs/README.txt?rev=HEAD&content-type=text/x-cvsweb-markup

`usersguide/index.txt`_, `usersguide/f2py_usersguide.pdf`_
  F2PY Users Guide and Reference Manual. Contains lots of examples.

`FAQ.txt`_ (in CVS__)
  F2PY Frequently Asked Questions.

__ http://cens.ioc.ee/cgi-bin/cvsweb/python/f2py2e/docs/FAQ.txt?rev=HEAD&content-type=text/x-cvsweb-markup

`TESTING.txt`_ (in CVS__)
  About F2PY testing site. What tests are available and how to run them.

__ http://cens.ioc.ee/cgi-bin/cvsweb/python/f2py2e/docs/TESTING.txt?rev=HEAD&content-type=text/x-cvsweb-markup

`HISTORY.txt`_ (in CVS__)
  A list of latest changes in F2PY. This is the most up-to-date
  document on F2PY.

__ http://cens.ioc.ee/cgi-bin/cvsweb/python/f2py2e/docs/HISTORY.txt?rev=HEAD&content-type=text/x-cvsweb-markup

`THANKS.txt`_
  Acknowledgments.

..
  `COMPILERS.txt`_
  Compiler and platform specific notes.

===============
 Mailing list
===============

A mailing list f2py-users@cens.ioc.ee is open for F2PY releated
discussion/questions/etc.

* `Subscribe..`__
* `Archives..`__

__ http://cens.ioc.ee/mailman/listinfo/f2py-users
__ http://cens.ioc.ee/pipermail/f2py-users


=====
 CVS
=====

F2PY is being developed under CVS_. The CVS version of F2PY can be
obtained as follows:

1) First you need to login (the password is ``guest``)::

    cvs -d :pserver:anonymous@cens.ioc.ee:/home/cvs login

2) and then do the checkout::

    cvs -z6 -d :pserver:anonymous@cens.ioc.ee:/home/cvs checkout f2py2e

3) You can update your local F2PY tree ``f2py2e/`` by executing::

    cvs -z6 update -P -d

You can browse the `F2PY CVS`_ repository.

===============
 Contributions
===============

* `A short introduction to F2PY`__ by Pierre Schnizer.

* `F2PY notes`__ by Fernando Perez.

* `Debian packages of F2PY`__ by José Fonseca. [OBSOLETE, Debian Sid
  ships python-f2py package]

__ http://fubphpc.tu-graz.ac.at/~pierre/f2py_tutorial.tar.gz
__ http://cens.ioc.ee/pipermail/f2py-users/2003-April/000472.html
__ http://jrfonseca.dyndns.org/debian/


===============
 Related sites
===============

* `Numerical Python`_ -- adds a fast array facility to the Python language.
* Pyfort_ -- A Python-Fortran connection tool.
* SciPy_ -- An open source library of scientific tools for Python.
* `Scientific Python`_ -- A collection of Python modules that are
  useful for scientific computing.
* `The Fortran Company`_ -- A place to find products, services, and general
  information related to the Fortran programming language.
* `American National Standard Programming Language FORTRAN ANSI(R) X3.9-1978`__
* `J3`_ -- The US Fortran standards committee.
* SWIG_ -- A software development tool that connects programs written
  in C and C++ with a variety of high-level programming languages.
* `Mathtools.net`_ -- A technical computing portal for all scientific
  and engineering needs.

.. __: http://www.fortran.com/fortran/F77_std/rjcnf.html

.. References
   ==========


.. _F2PY Users Guide and Reference Manual: usersguide/index.html
.. _usersguide/index.txt: usersguide/index.html
.. _usersguide/f2py_usersguide.pdf: usersguide/f2py_usersguide.pdf
.. _README.txt: README.html
.. _COMPILERS.txt: COMPILERS.html
.. _F2PY FAQ:
.. _FAQ.txt: FAQ.html
.. _HISTORY.txt: HISTORY.html
.. _HISTORY.txt from CVS: http://cens.ioc.ee/cgi-bin/cvsweb/python/f2py2e/docs/HISTORY.txt?rev=HEAD&content-type=text/x-cvsweb-markup
.. _THANKS.txt: THANKS.html
.. _TESTING.txt: TESTING.html
.. _F2PY CVS2: http://cens.ioc.ee/cgi-bin/cvsweb/python/f2py2e/
.. _F2PY CVS: http://cens.ioc.ee/cgi-bin/viewcvs.cgi/python/f2py2e/

.. _CVS: http://www.cvshome.org/
.. _Python: http://www.python.org/
.. _SciPy: http://www.numpy.org/
.. _NumPy: http://www.numpy.org/
.. _Numarray: http://www.stsci.edu/resources/software_hardware/numarray
.. _docutils: http://docutils.sourceforge.net/
.. _distutils: http://www.python.org/sigs/distutils-sig/
.. _Numerical Python: http://www.numpy.org/
.. _Pyfort: http://pyfortran.sourceforge.net/
.. _Scientific Python:
   http://starship.python.net/crew/hinsen/scientific.html
.. _The Fortran Company: http://www.fortran.com/fortran/
.. _J3: http://www.j3-fortran.org/
.. _Mathtools.net: http://www.mathtools.net/
.. _SWIG: http://www.swig.org/

..
   Local Variables:
   mode: indented-text
   indent-tabs-mode: nil
   sentence-end-double-space: t
   fill-column: 70
   End:
