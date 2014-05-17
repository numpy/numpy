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

The following example illustrates how to add an user-defined
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
