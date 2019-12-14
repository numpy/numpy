======================
Advanced F2PY usages
======================

Adding self-written functions to F2PY generated modules
=======================================================

Self-written Python C/API functions can be defined inside
signature files using ``usercode`` and ``pymethoddef`` statements
(they must be used inside the ``python module`` block). For
example, the following signature file ``spam.pyf``

.. include:: spam.pyf
   :literal:

wraps the C library function ``system()``::

  f2py -c spam.pyf

In Python:

.. include:: spam_session.dat
   :literal:

Modifying the dictionary of a F2PY generated module
===================================================

The following example illustrates how to add a user-defined
variables to a F2PY generated extension module. Given the following
signature file

.. include:: var.pyf
  :literal:

compile it as ``f2py -c var.pyf``.

Notice that the second ``usercode`` statement must be defined inside
an ``interface`` block and where the module dictionary is available through
the variable ``d`` (see ``f2py var.pyf``-generated ``varmodule.c`` for
additional details).

In Python:

.. include:: var_session.dat
  :literal:


Dealing with KIND specifiers
============================

Currently, F2PY can handle only ``<type spec>(kind=<kindselector>)``
declarations where ``<kindselector>`` is a numeric integer (e.g. 1, 2,
4,...), but not a function call ``KIND(..)`` or any other
expression. F2PY needs to know what would be the corresponding C type
and a general solution for that would be too complicated to implement.

However, F2PY provides a hook to overcome this difficulty, namely,
users can define their own <Fortran type> to <C type> maps. For
example, if Fortran 90 code contains::

    REAL(kind=KIND(0.0D0)) ...

then create a mapping file containing a Python dictionary::

    {'real': {'KIND(0.0D0)': 'double'}}

for instance.

Use the ``--f2cmap`` command-line option to pass the file name to F2PY.
By default, F2PY assumes file name is ``.f2py_f2cmap`` in the current
working directory.

Or more generally, the f2cmap file must contain a dictionary
with items::

    <Fortran typespec> : {<selector_expr>:<C type>}

that defines mapping between Fortran type::

    <Fortran typespec>([kind=]<selector_expr>)

and the corresponding <C type>. <C type> can be one of the following::

    char
    signed_char
    short
    int
    long_long
    float
    double
    long_double
    complex_float
    complex_double
    complex_long_double
    string

For more information, see F2Py source code ``numpy/f2py/capi_maps.py``.
