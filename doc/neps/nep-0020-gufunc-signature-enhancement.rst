===============================================================
NEP 20 — Expansion of Generalized Universal Function Signatures
===============================================================

:Author: Marten van Kerkwijk <mhvk@astro.utoronto.ca>
:Status: Accepted
:Type: Standards Track
:Created: 2018-06-10
:Resolution: https://mail.python.org/pipermail/numpy-discussion/2018-April/077959.html,
             https://mail.python.org/pipermail/numpy-discussion/2018-May/078078.html

.. note:: The proposal to add fixed (i) and flexible (ii) dimensions
          was accepted, while that to add broadcastable (iii) ones was deferred.

Abstract
--------

Generalized universal functions are, as their name indicates, generalization
of universal functions: they operate on non-scalar elements.  Their signature
describes the structure of the elements they operate on, with names linking
dimensions of the operands that should be the same.  Here, it is proposed to
extend the signature to allow the signature to indicate that a dimension (i)
has fixed size; (ii) can be absent; and (iii) can be broadcast.

Detailed description
--------------------

Each part of the proposal is driven by specific needs [1]_.

1. Fixed-size dimensions.  Code working with spatial vectors often explicitly
   is for 2 or 3-dimensional space (e.g., the code from the `Standards Of
   Fundamental Astronomy <http://www.iausofa.org/>`_, which the author hopes
   to wrap using gufuncs for astropy [2]_).  The signature should be able to
   indicate that.  E.g., the signature of a function that converts a polar
   angle to a two-dimensional cartesian unit vector would currently have to be
   ``()->(n)``, with there being no way to indicate that ``n`` has to equal 2.
   Indeed, this signature is particularly annoying since without putting in an
   output argument, the current gufunc wrapper code fails because it cannot
   determine ``n``.  Similarly, the signature for an cross product of two
   3-dimensional vectors has to be ``(n),(n)->(n)``, with again no way to
   indicate that ``n`` has to equal 3.  Hence, the proposal here to allow one
   to give numerical values in addition to variable names.  Thus, angle to
   two-dimensional unit vector would be ``()->(2)``; two angles to
   three-dimensional unit vector ``(),()->(3)``; and that for the cross
   product of two three-dimensional vectors would be ``(3),(3)->(3)``.

2. Possibly missing dimensions.  This part is almost entirely driven by the
   wish to wrap ``matmul`` in a gufunc. ``matmul`` stands for matrix
   multiplication, and if it did only that, it could be covered with the
   signature ``(m,n),(n,p)->(m,p)``. However, it has special cases for when a
   dimension is missing, allowing either argument to be treated as a single
   vector, with the function thus becoming, effectively, vector-matrix,
   matrix-vector, or vector-vector multiplication (but with no
   broadcasting). To support this, it is suggested to allow postfixing a
   dimension name with a question mark to indicate that the dimension does not
   necessarily have to be present.

   With this addition, the signature for ``matmul`` can be expressed as
   ``(m?,n),(n,p?)->(m?,p?)``.  This indicates that if, e.g., the second
   operand has only one dimension, for the purposes of the elementary function
   it will be treated as if that input has core shape ``(n, 1)``, and the
   output has the corresponding core shape of ``(m, 1)``. The actual output
   array, however, has the flexible dimension removed, i.e., it will have
   shape ``(..., m)``.  Similarly, if both arguments have only a single
   dimension, the inputs will be presented as having shapes ``(1, n)`` and
   ``(n, 1)`` to the elementary function, and the output as ``(1, 1)``, while
   the actual output array returned will have shape ``()``. In this way, the
   signature allows one to use a single elementary function for four related
   but different signatures, ``(m,n),(n,p)->(m,p)``, ``(n),(n,p)->(p)``,
   ``(m,n),(n)->(m)`` and ``(n),(n)->()``.

3. Dimensions that can be broadcast. For some applications, broadcasting
   between operands makes sense. For instance, an ``all_equal`` function that
   compares vectors in arrays could have a signature ``(n),(n)->()``, but this
   forces both operands to be arrays, while it would be useful also to check
   that, e.g., all parts of a vector are constant (maybe zero). The proposal
   is to allow the implementer of a gufunc to indicate that a dimension can be
   broadcast by post-fixing the dimension name with ``|1``. Hence, the
   signature for ``all_equal`` would become ``(n|1),(n|1)->()``.  The
   signature seems handy more generally for "chained ufuncs"; e.g., another
   application might be in a putative ufunc implementing ``sumproduct``.

   Another example that arose in the discussion, is of a weighted mean, which
   might look like ``weighted_mean(y, sigma[, axis, ...])``, returning the
   mean and its uncertainty.  With a signature of ``(n),(n)->(),()``, one
   would be forced to always give as many sigmas as there are data points,
   while broadcasting would allow one to give a single sigma for all points
   (which is still useful to calculate the uncertainty on the mean).

Implementation
--------------

The proposed changes have all been implemented [3]_, [4]_, [5]_. These PRs
extend the ufunc structure with two new fields, each of size equal to the
number of distinct dimensions, with ``core_dim_sizes`` holding possibly fixed
sizes, and ``core_dim_flags`` holding flags indicating whether a dimension can
be missing or broadcast.  To ensure we can distinguish between this new
version and previous versions, an unused entry ``reserved1`` is repurposed as
a version number.

In the implementation, care is taken that to the elementary function flagged
dimensions are not treated any differently than non-flagged ones: for
instance, sizes of fixed-size dimensions are still passed on to the elementary
function (but the loop can now count on that size being equal to the fixed one
given in the signature).

An implementation detail to be decided upon is whether it might be handy to
have a summary of all flags. This could possibly be stored in ``core_enabled``
(which currently is a bool), with non-zero continuing to indicate a gufunc,
but specific flags indicating whether or not a gufunc uses fixed, flexible, or
broadcastable dimensions.

With the above, the formal defition of the syntax would become [4]_::

  <Signature>            ::= <Input arguments> "->" <Output arguments>
  <Input arguments>      ::= <Argument list>
  <Output arguments>     ::= <Argument list>
  <Argument list>        ::= nil | <Argument> | <Argument> "," <Argument list>
  <Argument>             ::= "(" <Core dimension list> ")"
  <Core dimension list>  ::= nil | <Core dimension> |
                             <Core dimension> "," <Core dimension list>
  <Core dimension>       ::= <Dimension name> <Dimension modifier>
  <Dimension name>       ::= valid Python variable name | valid integer
  <Dimension modifier>   ::= nil | "|1" | "?"

#. All quotes are for clarity.
#. Unmodified core dimensions that share the same name must have the same size.
   Each dimension name typically corresponds to one level of looping in the
   elementary function's implementation.
#. White spaces are ignored.
#. An integer as a dimension name freezes that dimension to the value.
#. If a name if suffixed with the ``|1`` modifier, it is allowed to broadcast
   against other dimensions with the same name.  All input dimensions
   must share this modifier, while no output dimensions should have it.
#. If the name is suffixed with the ``?`` modifier, the dimension is a core
   dimension only if it exists on all inputs and outputs that share it;
   otherwise it is ignored (and replaced by a dimension of size 1 for the
   elementary function).

Examples of signatures [4]_:

+----------------------------+-----------------------------------+
| Signature                  | Possible use                      |
+----------------------------+-----------------------------------+
| ``(),()->()``              | Addition                          |
+----------------------------+-----------------------------------+
| ``(i)->()``                | Sum over last axis                |
+----------------------------+-----------------------------------+
| ``(i|1),(i|1)->()``        | Test for equality along axis,     |
|                            | allowing comparison with a scalar |
+----------------------------+-----------------------------------+
| ``(i),(i)->()``            | inner vector product              |
+----------------------------+-----------------------------------+
| ``(m,n),(n,p)->(m,p)``     | matrix multiplication             |
+----------------------------+-----------------------------------+
| ``(n),(n,p)->(p)``         | vector-matrix multiplication      |
+----------------------------+-----------------------------------+
| ``(m,n),(n)->(m)``         | matrix-vector multiplication      |
+----------------------------+-----------------------------------+
| ``(m?,n),(n,p?)->(m?,p?)`` | all four of the above at once,    |
|                            | except vectors cannot have loop   |
|                            | dimensions (ie, like ``matmul``)  |
+----------------------------+-----------------------------------+
| ``(3),(3)->(3)``           | cross product for 3-vectors       |
+----------------------------+-----------------------------------+
| ``(i,t),(j,t)->(i,j)``     | inner over the last dimension,    |
|                            | outer over the second to last,    |
|                            | and loop/broadcast over the rest. |
+----------------------------+-----------------------------------+

Backward compatibility
----------------------

One possible worry is the change in ufunc structure.  For most applications,
which call ``PyUFunc_FromDataAndSignature``, this is entirely transparent.
Furthermore, by repurposing ``reserved1`` as a version number, code compiled
against older versions of numpy will continue to work (though one will get a
warning upon import of that code with a newer version of numpy), except if
code explicitly changes the ``reserved1`` entry.

Alternatives
------------

It was suggested instead of extending the signature, to have multiple
dispatch, so that, e.g., ``matmul`` would simply have the multiple signatures
it supports, i.e., instead of ``(m?,n),(n,p?)->(m?,p?)`` one would have
``(m,n),(n,p)->(m,p) | (n),(n,p)->(p) | (m,n),(n)->(m) | (n),(n)->()``.  A
disadvantage of this is that the developer now has to make sure that the
elementary function can deal with these different signatures.  Furthermore,
the expansion quickly becomes cumbersome.  For instance, for the ``all_equal``
signature of ``(n|1),(n|1)->()``, one would have to have five entries:
``(n),(n)->() | (n),(1)->() | (1),(n)->() | (n),()->() | (),(n)->()``.  For
signatures like ``(m|1,n|1,o|1),(m|1,n|1,o|1)->()`` (from the ``cube_equal``
test case in [4]_), it is not even worth writing out the expansion.

For broadcasting, the alternative suffix of ``^`` was suggested (as
broadcasting can be thought of as increasing the size of the array).  This
seems less clear.  Furthermore, it was wondered whether it should not just be
an all-or-nothing flag.  This could be the case, though given the postfix
for flexible dimensions, arguably another postfix is clearer (as is the
implementation).

Discussion
----------

The proposals here were discussed at fair length on the mailing list [6]_,
[7]_.  The main points of contention were whether the use cases were
sufficiently strong. In particular, for frozen dimensions, it was argued that
checks on the right number could be put in loop selection code.  This seems
much less clear for no benefit.

For broadcasting, the lack of examples of elementary functions that might need
it was noted, with it being questioned whether something like ``all_equal``
was best done with a gufunc rather than as a special method on ``np.equal``.
One counter-argument to this would be that there is an actual PR for
``all_equal`` [8]_.  Another that even if one were to use a method, it would
be good to be able to express their signature (just as is possible at least
for ``reduce`` and ``accumulate``).

A final argument was that we were making the gufuncs too complex. This
arguably holds for the dimensions that can be omitted, but that also has the
strongest use case. The frozen dimensions has a very simple implementation and
its meaning is obvious. The ability to broadcast is simple too, once the
flexible dimensions are supported.

References and Footnotes
------------------------

.. [1] Identified needs and suggestions for the implementation are not all by
       the author. In particular, the suggestion for fixed dimensions and
       initial implementation was by Jaime Frio (`gh-5015
       <https://github.com/numpy/numpy/pull/5015>`_), the suggestion of ``?``
       to indicate dimensions can be omitted was by Nathaniel Smith, and the
       initial implementation of that by Matti Picus (`gh-11132
       <https://github.com/numpy/numpy/pull/11132>`_).
.. [2] `wrap ERFA functions in gufuncs
       <https://github.com/astropy/astropy/pull/7502>`_ (`ERFA
       <https://github.com/liberfa/erfa>`_) is the less stringently licensed
       version of `Standards Of Fundamental Astronomy
       <http://www.iausofa.org/>`_
.. [3] `fixed-size and flexible dimensions
       <https://github.com/numpy/numpy/pull/11175>`_
.. [4] `broadcastable dimensions
       <https://github.com/numpy/numpy/pull/11179>`_
.. [5] `use in matmul <https://github.com/numpy/numpy/pull/11133>`_
.. [6] Discusses implementations for ``matmul``:
       https://mail.python.org/pipermail/numpy-discussion/2018-May/077972.html,
       https://mail.python.org/pipermail/numpy-discussion/2018-May/078021.html
.. [7] Broadcasting:
       https://mail.python.org/pipermail/numpy-discussion/2018-May/078078.html
.. [8] `Logical gufuncs <https://github.com/numpy/numpy/pull/8528>`_ (includes
       ``all_equal``)

Copyright
---------

This document has been placed in the public domain.
