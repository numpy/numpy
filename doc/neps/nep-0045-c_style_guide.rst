.. _NEP45:

=================================
NEP 45 — C Style Guide
=================================

:Author: Charles Harris <charlesr.harris@gmail.com>
:Status: Accepted
:Type: Process
:Created: 2012-02-26
:Resolution: https://github.com/numpy/numpy/issues/11911

Abstract
--------

This document gives coding conventions for the C code comprising
the C implementation of NumPy.

Motivation and Scope
--------------------

The NumPy C coding conventions are based on Python
`PEP 7 -- Style Guide for C Code <https://www.python.org/dev/peps/pep-0007>`_
by Guido van Rossum with a few added strictures.

Because the NumPy conventions are very close to those in PEP 7, that PEP is
used as a template with the NumPy additions and variations in the appropriate
spots.

Usage and Impact
----------------

There are many C coding conventions and it must be emphasized that the primary
goal of the NumPy conventions isn't to choose the "best," about which there is
certain to be disagreement, but to achieve uniformity.

Two good reasons to break a particular rule:

1. When applying the rule would make the code less readable, even
   for someone who is used to reading code that follows the rules.

2. To be consistent with surrounding code that also breaks it
   (maybe for historic reasons) -- although this is also an
   opportunity to clean up someone else's mess.


Backward compatibility
----------------------

No impact.


Detailed description
--------------------

C dialect
=========

* Use C99 (that is, the standard defined by ISO/IEC 9899:1999).

* Don't use GCC extensions (for instance, don't write multi-line strings
  without trailing backslashes). Preferably break long strings
  up onto separate lines like so::

          "blah blah"
          "blah blah"

  This will work with MSVC, which otherwise chokes on very long
  strings.

* All function declarations and definitions must use full prototypes (that is,
  specify the types of all arguments).

* No compiler warnings with major compilers (gcc, VC++, a few others).

.. note::
   NumPy still produces compiler warnings that need to be addressed.

Code layout
============

* Use 4-space indents and no tabs at all.

* No line should be longer than 80 characters.  If this and the
  previous rule together don't give you enough room to code, your code is
  too complicated -- consider using subroutines.

* No line should end in whitespace.  If you think you need
  significant trailing whitespace, think again; somebody's editor might
  delete it as a matter of routine.

* Function definition style: function name in column 1, outermost
  curly braces in column 1, blank line after local variable declarations::

        static int
        extra_ivars(PyTypeObject *type, PyTypeObject *base)
        {
            int t_size = PyType_BASICSIZE(type);
            int b_size = PyType_BASICSIZE(base);

            assert(t_size >= b_size); /* type smaller than base! */
            ...
            return 1;
        }

  If the transition to C++ goes through it is possible that this form will
  be relaxed so that short class methods meant to be inlined can have the
  return type on the same line as the function name. However, that is yet to
  be determined.

* Code structure: one space between keywords like ``if``, ``for`` and
  the following left parenthesis; no spaces inside the parenthesis; braces
  around all ``if`` branches, and no statements on the same line as the
  ``if``. They should be formatted as shown::

        if (mro != NULL) {
            one_line_statement;
        }
        else {
            ...
        }


        for (i = 0; i < n; i++) {
            one_line_statement;
        }


        while (isstuff) {
            dostuff;
        }


        do {
            stuff;
        } while (isstuff);


        switch (kind) {
            /* Boolean kind */
            case 'b':
                return 0;
            /* Unsigned int kind */
            case 'u':
                ...
            /* Anything else */
            default:
                return 3;
        }


* The return statement should *not* get redundant parentheses::

        return Py_None; /* correct */
        return(Py_None); /* incorrect */

* Function and macro call style: ``foo(a, b, c)``, no space before
  the open paren, no spaces inside the parens, no spaces before
  commas, one space after each comma.

* Always put spaces around the assignment, Boolean, and comparison
  operators.  In expressions using a lot of operators, add spaces
  around the outermost (lowest priority) operators.

* Breaking long lines: If you can, break after commas in the
  outermost argument list.  Always indent continuation lines
  appropriately: ::

        PyErr_SetString(PyExc_TypeError,
                "Oh dear, you messed up.");

  Here appropriately means at least a double indent (8 spaces). It isn't
  necessary to line everything up with the opening parenthesis of the function
  call.

* When you break a long expression at a binary operator, the
  operator goes at the end of the previous line, for example: ::

        if (type > tp_dictoffset != 0 &&
                base > tp_dictoffset == 0 &&
                type > tp_dictoffset == b_size &&
                (size_t)t_size == b_size + sizeof(PyObject *)) {
            return 0;
        }

  Note that the terms in the multi-line Boolean expression are indented so
  as to make the beginning of the code block clearly visible.

* Put blank lines around functions, structure definitions, and
  major sections inside functions.

* Comments go before the code they describe. Multi-line comments should
  be like so: ::

        /*
         * This would be a long
         * explanatory comment.
         */

  Trailing comments should be used sparingly. Instead of ::

        if (yes) { // Success!

  do ::

        if (yes) {
            // Success!

* All functions and global variables should be declared static
  when they aren't needed outside the current compilation unit.

* Declare external functions and variables in a header file.


Naming conventions
==================

* There has been no consistent prefix for NumPy public functions, but
  they all begin with a prefix of some sort, followed by an underscore, and
  are in camel case: ``PyArray_DescrAlignConverter``, ``NpyIter_GetIterNext``.
  In the future the names should be of the form ``Npy*_PublicFunction``,
  where the star is something appropriate.

* Public Macros should have a ``NPY_`` prefix and then use upper case,
  for example, ``NPY_DOUBLE``.

* Private functions should be lower case with underscores, for example:
  ``array_real_get``. Single leading underscores should not be used, but
  some current function names violate that rule due to historical accident.

.. note::

   Functions whose names begin with a single underscore should be renamed at
   some point.


Function documentation
======================

NumPy doesn't have a C function documentation standard at this time, but
needs one. Most NumPy functions are not documented in the code, and that
should change. One possibility is Doxygen with a plugin so that the same
NumPy style used for Python functions can also be used for documenting
C functions, see the files in ``doc/cdoc/``.


Related Work
------------

Based on Van Rossum and Warsaw, :pep:`7`


Discussion
----------

https://github.com/numpy/numpy/issues/11911
recommended that this proposal, which originated as ``doc/C_STYLE_GUIDE.rst.txt``,
be turned into an NEP.


Copyright
---------

This document has been placed in the public domain.
