PEP: 465
Title: A dedicated infix operator for matrix multiplication
Version: $Revision$
Last-Modified: $Date$
Author: Nathaniel J. Smith <njs@pobox.com>
Status: Draft
Type: Standards Track
Content-Type: text/x-rst
Created: 20-Feb-2014
Python-Version: 3.5
Post-History: 13-Mar-2014

Abstract
========

This PEP proposes a new binary operator to be used for matrix
multiplication, called ``@``.  (Mnemonic: ``@`` is ``*`` for
mATrices.)


Specification
=============

A new binary operator is added to the Python language, together
with the corresponding in-place version:

=======  ========================= ===============================
 Op      Precedence/associativity     Methods
=======  ========================= ===============================
``@``    Same as ``*``             ``__matmul__``, ``__rmatmul__``
``@=``   n/a                       ``__imatmul__``
=======  ========================= ===============================

No implementations of these methods are added to the builtin or
standard library types.  However, a number of projects have reached
consensus on the recommended semantics for these operations; see
`Intended usage details`_ below for details.

For details on how this operator will be implemented in CPython, see
`Implementation details`_.


Motivation
==========

Executive summary
-----------------

In numerical code, there are two important operations which compete
for use of Python's ``*`` operator: elementwise multiplication, and
matrix multiplication.  In the nearly twenty years since the Numeric
library was first proposed, there have been many attempts to resolve
this tension [#hugunin]_; none have been really satisfactory.
Currently, most numerical Python code uses ``*`` for elementwise
multiplication, and function/method syntax for matrix multiplication;
however, this leads to ugly and unreadable code in common
circumstances.  The problem is bad enough that significant amounts of
code continue to use the opposite convention (which has the virtue of
producing ugly and unreadable code in *different* circumstances), and
this API fragmentation across codebases then creates yet more
problems.  There does not seem to be any *good* solution to the
problem of designing a numerical API within current Python syntax --
only a landscape of options that are bad in different ways.  The
minimal change to Python syntax which is sufficient to resolve these
problems is the addition of a single new infix operator for matrix
multiplication.

Matrix multiplication has a singular combination of features which
distinguish it from other binary operations, which together provide a
uniquely compelling case for the addition of a dedicated infix
operator:

* Just as for the existing numerical operators, there exists a vast
  body of prior art supporting the use of infix notation for matrix
  multiplication across all fields of mathematics, science, and
  engineering; ``@`` harmoniously fills a hole in Python's existing
  operator system.

* ``@`` greatly clarifies real-world code.

* ``@`` provides a smoother onramp for less experienced users, who are
  particularly harmed by hard-to-read code and API fragmentation.

* ``@`` benefits a substantial and growing portion of the Python user
  community.

* ``@`` will be used frequently -- in fact, evidence suggests it may
  be used more frequently than ``//`` or the bitwise operators.

* ``@`` allows the Python numerical community to reduce fragmentation,
  and finally standardize on a single consensus duck type for all
  numerical array objects.


Background: What's wrong with the status quo?
---------------------------------------------

When we crunch numbers on a computer, we usually have lots and lots of
numbers to deal with.  Trying to deal with them one at a time is
cumbersome and slow -- especially when using an interpreted language.
Instead, we want the ability to write down simple operations that
apply to large collections of numbers all at once.  The *n-dimensional
array* is the basic object that all popular numeric computing
environments use to make this possible.  Python has several libraries
that provide such arrays, with numpy being at present the most
prominent.

When working with n-dimensional arrays, there are two different ways
we might want to define multiplication.  One is elementwise
multiplication::

  [[1, 2],     [[11, 12],     [[1 * 11, 2 * 12],
   [3, 4]]  x   [13, 14]]  =   [3 * 13, 4 * 14]]

and the other is `matrix multiplication`_:

.. _matrix multiplication: https://en.wikipedia.org/wiki/Matrix_multiplication

::

  [[1, 2],     [[11, 12],     [[1 * 11 + 2 * 13, 1 * 12 + 2 * 14],
   [3, 4]]  x   [13, 14]]  =   [3 * 11 + 4 * 13, 3 * 12 + 4 * 14]]

Elementwise multiplication is useful because it lets us easily and
quickly perform many multiplications on a large collection of values,
without writing a slow and cumbersome ``for`` loop.  And this works as
part of a very general schema: when using the array objects provided
by numpy or other numerical libraries, all Python operators work
elementwise on arrays of all dimensionalities.  The result is that one
can write functions using straightforward code like ``a * b + c / d``,
treating the variables as if they were simple values, but then
immediately use this function to efficiently perform this calculation
on large collections of values, while keeping them organized using
whatever arbitrarily complex array layout works best for the problem
at hand.

Matrix multiplication is more of a special case.  It's only defined on
2d arrays (also known as "matrices"), and multiplication is the only
operation that has an important "matrix" version -- "matrix addition"
is the same as elementwise addition; there is no such thing as "matrix
bitwise-or" or "matrix floordiv"; "matrix division" and "matrix
to-the-power-of" can be defined but are not very useful, etc.
However, matrix multiplication is still used very heavily across all
numerical application areas; mathematically, it's one of the most
fundamental operations there is.

Because Python syntax currently allows for only a single
multiplication operator ``*``, libraries providing array-like objects
must decide: either use ``*`` for elementwise multiplication, or use
``*`` for matrix multiplication.  And, unfortunately, it turns out
that when doing general-purpose number crunching, both operations are
used frequently, and there are major advantages to using infix rather
than function call syntax in both cases.  Thus it is not at all clear
which convention is optimal, or even acceptable; often it varies on a
case-by-case basis.

Nonetheless, network effects mean that it is very important that we
pick *just one* convention.  In numpy, for example, it is technically
possible to switch between the conventions, because numpy provides two
different types with different ``__mul__`` methods.  For
``numpy.ndarray`` objects, ``*`` performs elementwise multiplication,
and matrix multiplication must use a function call (``numpy.dot``).
For ``numpy.matrix`` objects, ``*`` performs matrix multiplication,
and elementwise multiplication requires function syntax.  Writing code
using ``numpy.ndarray`` works fine.  Writing code using
``numpy.matrix`` also works fine.  But trouble begins as soon as we
try to integrate these two pieces of code together.  Code that expects
an ``ndarray`` and gets a ``matrix``, or vice-versa, may crash or
return incorrect results.  Keeping track of which functions expect
which types as inputs, and return which types as outputs, and then
converting back and forth all the time, is incredibly cumbersome and
impossible to get right at any scale.  Functions that defensively try
to handle both types as input and DTRT, find themselves floundering
into a swamp of ``isinstance`` and ``if`` statements.

PEP 238 split ``/`` into two operators: ``/`` and ``//``.  Imagine the
chaos that would have resulted if it had instead split ``int`` into
two types: ``classic_int``, whose ``__div__`` implemented floor
division, and ``new_int``, whose ``__div__`` implemented true
division.  This, in a more limited way, is the situation that Python
number-crunchers currently find themselves in.

In practice, the vast majority of projects have settled on the
convention of using ``*`` for elementwise multiplication, and function
call syntax for matrix multiplication (e.g., using ``numpy.ndarray``
instead of ``numpy.matrix``).  This reduces the problems caused by API
fragmentation, but it doesn't eliminate them.  The strong desire to
use infix notation for matrix multiplication has caused a number of
specialized array libraries to continue to use the opposing convention
(e.g., scipy.sparse, pyoperators, pyviennacl) despite the problems
this causes, and ``numpy.matrix`` itself still gets used in
introductory programming courses, often appears in StackOverflow
answers, and so forth.  Well-written libraries thus must continue to
be prepared to deal with both types of objects, and, of course, are
also stuck using unpleasant funcall syntax for matrix multiplication.
After nearly two decades of trying, the numerical community has still
not found any way to resolve these problems within the constraints of
current Python syntax (see `Rejected alternatives to adding a new
operator`_ below).

This PEP proposes the minimum effective change to Python syntax that
will allow us to drain this swamp.  It splits ``*`` into two
operators, just as was done for ``/``: ``*`` for elementwise
multiplication, and ``@`` for matrix multiplication.  (Why not the
reverse?  Because this way is compatible with the existing consensus,
and because it gives us a consistent rule that all the built-in
numeric operators also apply in an elementwise manner to arrays; the
reverse convention would lead to more special cases.)

So that's why matrix multiplication doesn't and can't just use ``*``.
Now, in the the rest of this section, we'll explain why it nonetheless
meets the high bar for adding a new operator.


Why should matrix multiplication be infix?
------------------------------------------

Right now, most numerical code in Python uses syntax like
``numpy.dot(a, b)`` or ``a.dot(b)`` to perform matrix multiplication.
This obviously works, so why do people make such a fuss about it, even
to the point of creating API fragmentation and compatibility swamps?

Matrix multiplication shares two features with ordinary arithmetic
operations like addition and multiplication on numbers: (a) it is used
very heavily in numerical programs -- often multiple times per line of
code -- and (b) it has an ancient and universally adopted tradition of
being written using infix syntax.  This is because, for typical
formulas, this notation is dramatically more readable than any
function call syntax.  Here's an example to demonstrate:

One of the most useful tools for testing a statistical hypothesis is
the linear hypothesis test for OLS regression models.  It doesn't
really matter what all those words I just said mean; if we find
ourselves having to implement this thing, what we'll do is look up
some textbook or paper on it, and encounter many mathematical formulas
that look like:

.. math::

    S = (H \beta - r)^T (H V H^T)^{-1} (H \beta - r)

Here the various variables are all vectors or matrices (details for
the curious: [#lht]_).

Now we need to write code to perform this calculation. In current
numpy, matrix multiplication can be performed using either the
function or method call syntax. Neither provides a particularly
readable translation of the formula::

    import numpy as np
    from numpy.linalg import inv, solve

    # Using dot function:
    S = np.dot((np.dot(H, beta) - r).T,
               np.dot(inv(np.dot(np.dot(H, V), H.T)), np.dot(H, beta) - r))

    # Using dot method:
    S = (H.dot(beta) - r).T.dot(inv(H.dot(V).dot(H.T))).dot(H.dot(beta) - r)

With the ``@`` operator, the direct translation of the above formula
becomes::

    S = (H @ beta - r).T @ inv(H @ V @ H.T) @ (H @ beta - r)

Notice that there is now a transparent, 1-to-1 mapping between the
symbols in the original formula and the code that implements it.

Of course, an experienced programmer will probably notice that this is
not the best way to compute this expression.  The repeated computation
of :math:`H \beta - r` should perhaps be factored out; and,
expressions of the form ``dot(inv(A), B)`` should almost always be
replaced by the more numerically stable ``solve(A, B)``.  When using
``@``, performing these two refactorings gives us::

    # Version 1 (as above)
    S = (H @ beta - r).T @ inv(H @ V @ H.T) @ (H @ beta - r)

    # Version 2
    trans_coef = H @ beta - r
    S = trans_coef.T @ inv(H @ V @ H.T) @ trans_coef

    # Version 3
    S = trans_coef.T @ solve(H @ V @ H.T, trans_coef)

Notice that when comparing between each pair of steps, it's very easy
to see exactly what was changed.  If we apply the equivalent
transformations to the code using the .dot method, then the changes
are much harder to read out or verify for correctness::

    # Version 1 (as above)
    S = (H.dot(beta) - r).T.dot(inv(H.dot(V).dot(H.T))).dot(H.dot(beta) - r)

    # Version 2
    trans_coef = H.dot(beta) - r
    S = trans_coef.T.dot(inv(H.dot(V).dot(H.T))).dot(trans_coef)

    # Version 3
    S = trans_coef.T.dot(solve(H.dot(V).dot(H.T)), trans_coef)

Readability counts!  The statements using ``@`` are shorter, contain
more whitespace, can be directly and easily compared both to each
other and to the textbook formula, and contain only meaningful
parentheses.  This last point is particularly important for
readability: when using function-call syntax, the required parentheses
on every operation create visual clutter that makes it very difficult
to parse out the overall structure of the formula by eye, even for a
relatively simple formula like this one.  Eyes are terrible at parsing
non-regular languages.  I made and caught many errors while trying to
write out the 'dot' formulas above.  I know they still contain at
least one error, maybe more.  (Exercise: find it.  Or them.)  The
``@`` examples, by contrast, are not only correct, they're obviously
correct at a glance.

If we are even more sophisticated programmers, and writing code that
we expect to be reused, then considerations of speed or numerical
accuracy might lead us to prefer some particular order of evaluation.
Because ``@`` makes it possible to omit irrelevant parentheses, we can
be certain that if we *do* write something like ``(H @ V) @ H.T``,
then our readers will know that the parentheses must have been added
intentionally to accomplish some meaningful purpose.  In the ``dot``
examples, it's impossible to know which nesting decisions are
important, and which are arbitrary.

Infix ``@`` dramatically improves matrix code usability at all stages
of programmer interaction.


Transparent syntax is especially crucial for non-expert programmers
-------------------------------------------------------------------

A large proportion of scientific code is written by people who are
experts in their domain, but are not experts in programming.  And
there are many university courses run each year with titles like "Data
analysis for social scientists" which assume no programming
background, and teach some combination of mathematical techniques,
introduction to programming, and the use of programming to implement
these mathematical techniques, all within a 10-15 week period.  These
courses are more and more often being taught in Python rather than
special-purpose languages like R or Matlab.

For these kinds of users, whose programming knowledge is fragile, the
existence of a transparent mapping between formulas and code often
means the difference between succeeding and failing to write that code
at all.  This is so important that such classes often use the
``numpy.matrix`` type which defines ``*`` to mean matrix
multiplication, even though this type is buggy and heavily
disrecommended by the rest of the numpy community for the
fragmentation that it causes.  This pedagogical use case is, in fact,
the *only* reason ``numpy.matrix`` remains a supported part of numpy.
Adding ``@`` will benefit both beginning and advanced users with
better syntax; and furthermore, it will allow both groups to
standardize on the same notation from the start, providing a smoother
on-ramp to expertise.


But isn't matrix multiplication a pretty niche requirement?
-----------------------------------------------------------

The world is full of continuous data, and computers are increasingly
called upon to work with it in sophisticated ways.  Arrays are the
lingua franca of finance, machine learning, 3d graphics, computer
vision, robotics, operations research, econometrics, meteorology,
computational linguistics, recommendation systems, neuroscience,
astronomy, bioinformatics (including genetics, cancer research, drug
discovery, etc.), physics engines, quantum mechanics, geophysics,
network analysis, and many other application areas.  In most or all of
these areas, Python is rapidly becoming a dominant player, in large
part because of its ability to elegantly mix traditional discrete data
structures (hash tables, strings, etc.) on an equal footing with
modern numerical data types and algorithms.

We all live in our own little sub-communities, so some Python users
may be surprised to realize the sheer extent to which Python is used
for number crunching -- especially since much of this particular
sub-community's activity occurs outside of traditional Python/FOSS
channels.  So, to give some rough idea of just how many numerical
Python programmers are actually out there, here are two numbers: In
2013, there were 7 international conferences organized specifically on
numerical Python [#scipy-conf]_ [#pydata-conf]_.  At PyCon 2014, ~20%
of the tutorials appear to involve the use of matrices
[#pycon-tutorials]_.

To quantify this further, we used Github's "search" function to look
at what modules are actually imported across a wide range of
real-world code (i.e., all the code on Github).  We checked for
imports of several popular stdlib modules, a variety of numerically
oriented modules, and various other extremely high-profile modules
like django and lxml (the latter of which is the #1 most downloaded
package on PyPI).  Starred lines indicate packages which export array-
or matrix-like objects which will adopt ``@`` if this PEP is
approved::

    Count of Python source files on Github matching given search terms
                     (as of 2014-04-10, ~21:00 UTC)
    ================ ==========  ===============  =======  ===========
    module           "import X"  "from X import"    total  total/numpy
    ================ ==========  ===============  =======  ===========
    sys                 2374638            63301  2437939         5.85
    os                  1971515            37571  2009086         4.82
    re                  1294651             8358  1303009         3.12
    numpy ************** 337916 ********** 79065 * 416981 ******* 1.00
    warnings             298195            73150   371345         0.89
    subprocess           281290            63644   344934         0.83
    django                62795           219302   282097         0.68
    math                 200084            81903   281987         0.68
    threading            212302            45423   257725         0.62
    pickle+cPickle       215349            22672   238021         0.57
    matplotlib           119054            27859   146913         0.35
    sqlalchemy            29842            82850   112692         0.27
    pylab *************** 36754 ********** 41063 ** 77817 ******* 0.19
    scipy *************** 40829 ********** 28263 ** 69092 ******* 0.17
    lxml                  19026            38061    57087         0.14
    zlib                  40486             6623    47109         0.11
    multiprocessing       25247            19850    45097         0.11
    requests              30896              560    31456         0.08
    jinja2                 8057            24047    32104         0.08
    twisted               13858             6404    20262         0.05
    gevent                11309             8529    19838         0.05
    pandas ************** 14923 *********** 4005 ** 18928 ******* 0.05
    sympy                  2779             9537    12316         0.03
    theano *************** 3654 *********** 1828 *** 5482 ******* 0.01
    ================ ==========  ===============  =======  ===========

These numbers should be taken with several grains of salt (see
footnote for discussion: [#github-details]_), but, to the extent they
can be trusted, they suggest that ``numpy`` might be the single
most-imported non-stdlib module in the entire Pythonverse; it's even
more-imported than such stdlib stalwarts as ``subprocess``, ``math``,
``pickle``, and ``threading``.  And numpy users represent only a
subset of the broader numerical community that will benefit from the
``@`` operator.  Matrices may once have been a niche data type
restricted to Fortran programs running in university labs and military
clusters, but those days are long gone.  Number crunching is a
mainstream part of modern Python usage.

In addition, there is some precedence for adding an infix operator to
handle a more-specialized arithmetic operation: the floor division
operator ``//``, like the bitwise operators, is very useful under
certain circumstances when performing exact calculations on discrete
values.  But it seems likely that there are many Python programmers
who have never had reason to use ``//`` (or, for that matter, the
bitwise operators).  ``@`` is no more niche than ``//``.


So ``@`` is good for matrix formulas, but how common are those really?
----------------------------------------------------------------------

We've seen that ``@`` makes matrix formulas dramatically easier to
work with for both experts and non-experts, that matrix formulas
appear in many important applications, and that numerical libraries
like numpy are used by a substantial proportion of Python's user base.
But numerical libraries aren't just about matrix formulas, and being
important doesn't necessarily mean taking up a lot of code: if matrix
formulas only occured in one or two places in the average
numerically-oriented project, then it still wouldn't be worth adding a
new operator.  So how common is matrix multiplication, really?

When the going gets tough, the tough get empirical.  To get a rough
estimate of how useful the ``@`` operator will be, the table below
shows the rate at which different Python operators are actually used
in the stdlib, and also in two high-profile numerical packages -- the
scikit-learn machine learning library, and the nipy neuroimaging
library -- normalized by source lines of code (SLOC).  Rows are sorted
by the 'combined' column, which pools all three code bases together.
The combined column is thus strongly weighted towards the stdlib,
which is much larger than both projects put together (stdlib: 411575
SLOC, scikit-learn: 50924 SLOC, nipy: 37078 SLOC). [#sloc-details]_

The ``dot`` row (marked ``******``) counts how common matrix multiply
operations are in each codebase.

::

    ====  ======  ============  ====  ========
      op  stdlib  scikit-learn  nipy  combined
    ====  ======  ============  ====  ========
       =    2969          5536  4932      3376 / 10,000 SLOC
       -     218           444   496       261
       +     224           201   348       231
      ==     177           248   334       196
       *     156           284   465       192
       %     121           114   107       119
      **      59           111   118        68
      !=      40            56    74        44
       /      18           121   183        41
       >      29            70   110        39
      +=      34            61    67        39
       <      32            62    76        38
      >=      19            17    17        18
      <=      18            27    12        18
     dot ***** 0 ********** 99 ** 74 ****** 16
       |      18             1     2        15
       &      14             0     6        12
      <<      10             1     1         8
      //       9             9     1         8
      -=       5            21    14         8
      *=       2            19    22         5
      /=       0            23    16         4
      >>       4             0     0         3
       ^       3             0     0         3
       ~       2             4     5         2
      |=       3             0     0         2
      &=       1             0     0         1
     //=       1             0     0         1
      ^=       1             0     0         0
     **=       0             2     0         0
      %=       0             0     0         0
     <<=       0             0     0         0
     >>=       0             0     0         0
    ====  ======  ============  ====  ========

These two numerical packages alone contain ~780 uses of matrix
multiplication.  Within these packages, matrix multiplication is used
more heavily than most comparison operators (``<`` ``!=`` ``<=``
``>=``).  Even when we dilute these counts by including the stdlib
into our comparisons, matrix multiplication is still used more often
in total than any of the bitwise operators, and 2x as often as ``//``.
This is true even though the stdlib, which contains a fair amount of
integer arithmetic and no matrix operations, makes up more than 80% of
the combined code base.

By coincidence, the numeric libraries make up approximately the same
proportion of the 'combined' codebase as numeric tutorials make up of
PyCon 2014's tutorial schedule, which suggests that the 'combined'
column may not be *wildly* unrepresentative of new Python code in
general.  While it's impossible to know for certain, from this data it
seems entirely possible that across all Python code currently being
written, matrix multiplication is already used more often than ``//``
and the bitwise operations.


But isn't it weird to add an operator with no stdlib uses?
----------------------------------------------------------

It's certainly unusual (though extended slicing existed for some time
builtin types gained support for it, ``Ellipsis`` is still unused
within the stdlib, etc.).  But the important thing is whether a change
will benefit users, not where the software is being downloaded from.
It's clear from the above that ``@`` will be used, and used heavily.
And this PEP provides the critical piece that will allow the Python
numerical community to finally reach consensus on a standard duck type
for all array-like objects, which is a necessary precondition to ever
adding a numerical array type to the stdlib.


Compatibility considerations
============================

Currently, the only legal use of the ``@`` token in Python code is at
statement beginning in decorators.  The new operators are both infix;
the one place they can never occur is at statement beginning.
Therefore, no existing code will be broken by the addition of these
operators, and there is no possible parsing ambiguity between
decorator-@ and the new operators.

Another important kind of compatibility is the mental cost paid by
users to update their understanding of the Python language after this
change, particularly for users who do not work with matrices and thus
do not benefit.  Here again, ``@`` has minimal impact: even
comprehensive tutorials and references will only need to add a
sentence or two to fully document this PEP's changes for a
non-numerical audience.


Intended usage details
======================

This section is informative, rather than normative -- it documents the
consensus of a number of libraries that provide array- or matrix-like
objects on how ``@`` will be implemented.

This section uses the numpy terminology for describing arbitrary
multidimensional arrays of data, because it is a superset of all other
commonly used models.  In this model, the *shape* of any array is
represented by a tuple of integers.  Because matrices are
two-dimensional, they have len(shape) == 2, while 1d vectors have
len(shape) == 1, and scalars have shape == (), i.e., they are "0
dimensional".  Any array contains prod(shape) total entries.  Notice
that `prod(()) == 1`_ (for the same reason that sum(()) == 0); scalars
are just an ordinary kind of array, not a special case.  Notice also
that we distinguish between a single scalar value (shape == (),
analogous to ``1``), a vector containing only a single entry (shape ==
(1,), analogous to ``[1]``), a matrix containing only a single entry
(shape == (1, 1), analogous to ``[[1]]``), etc., so the dimensionality
of any array is always well-defined.  Other libraries with more
restricted representations (e.g., those that support 2d arrays only)
might implement only a subset of the functionality described here.

.. _prod(()) == 1: https://en.wikipedia.org/wiki/Empty_product

Semantics
---------

The recommended semantics for ``@`` for different inputs are:

* 2d inputs are conventional matrices, and so the semantics are
  obvious: we apply conventional matrix multiplication.  If we write
  ``arr(2, 3)`` to represent an arbitrary 2x3 array, then ``arr(2, 3)
  @ arr(3, 4)`` returns an array with shape (2, 4).

* 1d vector inputs are promoted to 2d by prepending or appending a '1'
  to the shape, the operation is performed, and then the added
  dimension is removed from the output.  The 1 is always added on the
  "outside" of the shape: prepended for left arguments, and appended
  for right arguments.  The result is that matrix @ vector and vector
  @ matrix are both legal (assuming compatible shapes), and both
  return 1d vectors; vector @ vector returns a scalar.  This is
  clearer with examples.

  * ``arr(2, 3) @ arr(3, 1)`` is a regular matrix product, and returns
    an array with shape (2, 1), i.e., a column vector.

  * ``arr(2, 3) @ arr(3)`` performs the same computation as the
    previous (i.e., treats the 1d vector as a matrix containing a
    single *column*, shape = (3, 1)), but returns the result with
    shape (2,), i.e., a 1d vector.

  * ``arr(1, 3) @ arr(3, 2)`` is a regular matrix product, and returns
    an array with shape (1, 2), i.e., a row vector.

  * ``arr(3) @ arr(3, 2)`` performs the same computation as the
    previous (i.e., treats the 1d vector as a matrix containing a
    single *row*, shape = (1, 3)), but returns the result with shape
    (2,), i.e., a 1d vector.

  * ``arr(1, 3) @ arr(3, 1)`` is a regular matrix product, and returns
    an array with shape (1, 1), i.e., a single value in matrix form.

  * ``arr(3) @ arr(3)`` performs the same computation as the
    previous, but returns the result with shape (), i.e., a single
    scalar value, not in matrix form.  So this is the standard inner
    product on vectors.

  An infelicity of this definition for 1d vectors is that it makes
  ``@`` non-associative in some cases (``(Mat1 @ vec) @ Mat2`` !=
  ``Mat1 @ (vec @ Mat2)``).  But this seems to be a case where
  practicality beats purity: non-associativity only arises for strange
  expressions that would never be written in practice; if they are
  written anyway then there is a consistent rule for understanding
  what will happen (``Mat1 @ vec @ Mat2`` is parsed as ``(Mat1 @ vec)
  @ Mat2``, just like ``a - b - c``); and, not supporting 1d vectors
  would rule out many important use cases that do arise very commonly
  in practice.  No-one wants to explain to new users why to solve the
  simplest linear system in the obvious way, they have to type
  ``(inv(A) @ b[:, np.newaxis]).flatten()`` instead of ``inv(A) @ b``,
  or perform an ordinary least-squares regression by typing
  ``solve(X.T @ X, X @ y[:, np.newaxis]).flatten()`` instead of
  ``solve(X.T @ X, X @ y)``.  No-one wants to type ``(a[np.newaxis, :]
  @ b[:, np.newaxis])[0, 0]`` instead of ``a @ b`` every time they
  compute an inner product, or ``(a[np.newaxis, :] @ Mat @ b[:,
  np.newaxis])[0, 0]`` for general quadratic forms instead of ``a @
  Mat @ b``.  In addition, sage and sympy (see below) use these
  non-associative semantics with an infix matrix multiplication
  operator (they use ``*``), and they report that they haven't
  experienced any problems caused by it.

* For inputs with more than 2 dimensions, we treat the last two
  dimensions as being the dimensions of the matrices to multiply, and
  'broadcast' across the other dimensions.  This provides a convenient
  way to quickly compute many matrix products in a single operation.
  For example, ``arr(10, 2, 3) @ arr(10, 3, 4)`` performs 10 separate
  matrix multiplies, each of which multiplies a 2x3 and a 3x4 matrix
  to produce a 2x4 matrix, and then returns the 10 resulting matrices
  together in an array with shape (10, 2, 4).  The intuition here is
  that we treat these 3d arrays of numbers as if they were 1d arrays
  *of matrices*, and then apply matrix multiplication in an
  elementwise manner, where now each 'element' is a whole matrix.
  Note that broadcasting is not limited to perfectly aligned arrays;
  in more complicated cases, it allows several simple but powerful
  tricks for controlling how arrays are aligned with each other; see
  [#broadcasting]_ for details.  (In particular, it turns out that
  when broadcasting is taken into account, the standard scalar *
  matrix product is a special case of the elementwise multiplication
  operator ``*``.)

  If one operand is >2d, and another operand is 1d, then the above
  rules apply unchanged, with 1d->2d promotion performed before
  broadcasting.  E.g., ``arr(10, 2, 3) @ arr(3)`` first promotes to
  ``arr(10, 2, 3) @ arr(3, 1)``, then broadcasts the right argument to
  create the aligned operation ``arr(10, 2, 3) @ arr(10, 3, 1)``,
  multiplies to get an array with shape (10, 2, 1), and finally
  removes the added dimension, returning an array with shape (10, 2).
  Similarly, ``arr(2) @ arr(10, 2, 3)`` produces an intermediate array
  with shape (10, 1, 3), and a final array with shape (10, 3).

* 0d (scalar) inputs raise an error.  Scalar * matrix multiplication
  is a mathematically and algorithmically distinct operation from
  matrix @ matrix multiplication, and is already covered by the
  elementwise ``*`` operator.  Allowing scalar @ matrix would thus
  both require an unnecessary special case, and violate TOOWTDI.


Adoption
--------

We group existing Python projects which provide array- or matrix-like
types based on what API they currently use for elementwise and matrix
multiplication.

**Projects which currently use * for elementwise multiplication, and
function/method calls for matrix multiplication:**

The developers of the following projects have expressed an intention
to implement ``@`` on their array-like types using the above
semantics:

* numpy
* pandas
* blaze
* theano

The following projects have been alerted to the existence of the PEP,
but it's not yet known what they plan to do if it's accepted.  We
don't anticipate that they'll have any objections, though, since
everything proposed here is consistent with how they already do
things:

* pycuda
* panda3d

**Projects which currently use * for matrix multiplication, and
function/method calls for elementwise multiplication:**

The following projects have expressed an intention, if this PEP is
accepted, to migrate from their current API to the elementwise-``*``,
matmul-``@`` convention (i.e., this is a list of projects whose API
fragmentation will probably be eliminated if this PEP is accepted):

* numpy (``numpy.matrix``)
* scipy.sparse
* pyoperators
* pyviennacl

The following projects have been alerted to the existence of the PEP,
but it's not known what they plan to do if it's accepted (i.e., this
is a list of projects whose API fragmentation may or may not be
eliminated if this PEP is accepted):

* cvxopt

**Projects which currently use * for matrix multiplication, and which
don't really care about elementwise multiplication of matrices:**

There are several projects which implement matrix types, but from a
very different perspective than the numerical libraries discussed
above.  These projects focus on computational methods for analyzing
matrices in the sense of abstract mathematical objects (i.e., linear
maps over free modules over rings), rather than as big bags full of
numbers that need crunching.  And it turns out that from the abstract
math point of view, there isn't much use for elementwise operations in
the first place; as discussed in the Background section above,
elementwise operations are motivated by the bag-of-numbers approach.
So these projects don't encounter the basic problem that this PEP
exists to address, making it mostly irrelevant to them; while they
appear superficially similar to projects like numpy, they're actually
doing something quite different.  They use ``*`` for matrix
multiplication (and for group actions, and so forth), and if this PEP
is accepted, their expressed intention is to continue doing so, while
perhaps adding ``@`` as an alias.  These projects include:

* sympy
* sage


Implementation details
======================

New functions ``operator.matmul`` and ``operator.__matmul__`` are
added to the standard library, with the usual semantics.

A corresponding function ``PyObject* PyObject_MatrixMultiply(PyObject
*o1, PyObject o2)`` is added to the C API.

A new AST node is added named ``MatMult``, along with a new token
``ATEQUAL`` and new bytecode opcodes ``BINARY_MATRIX_MULTIPLY`` and
``INPLACE_MATRIX_MULTIPLY``.

Two new type slots are added; whether this is to ``PyNumberMethods``
or a new ``PyMatrixMethods`` struct remains to be determined.


Rationale for specification details
===================================

Choice of operator
------------------

Why ``@`` instead of some other spelling?  There isn't any consensus
across other programming languages about how this operator should be
named [#matmul-other-langs]_; here we discuss the various options.

Restricting ourselves only to symbols present on US English keyboards,
the punctuation characters that don't already have a meaning in Python
expression context are: ``@``, backtick, ``$``, ``!``, and ``?``.  Of
these options, ``@`` is clearly the best; ``!`` and ``?`` are already
heavily freighted with inapplicable meanings in the programming
context, backtick has been banned from Python by BDFL pronouncement
(see PEP 3099), and ``$`` is uglier, even more dissimilar to ``*`` and
:math:`\cdot`, and has Perl/PHP baggage.  ``$`` is probably the
second-best option of these, though.

Symbols which are not present on US English keyboards start at a
significant disadvantage (having to spend 5 minutes at the beginning
of every numeric Python tutorial just going over keyboard layouts is
not a hassle anyone really wants).  Plus, even if we somehow overcame
the typing problem, it's not clear there are any that are actually
better than ``@``.  Some options that have been suggested include:

* U+00D7 MULTIPLICATION SIGN: ``A × B``
* U+22C5 DOT OPERATOR: ``A ⋅ B``
* U+2297 CIRCLED TIMES: ``A ⊗ B``
* U+00B0 DEGREE: ``A ° B``

What we need, though, is an operator that means "matrix
multiplication, as opposed to scalar/elementwise multiplication".
There is no conventional symbol with this meaning in either
programming or mathematics, where these operations are usually
distinguished by context.  (And U+2297 CIRCLED TIMES is actually used
conventionally to mean exactly the wrong things: elementwise
multiplication -- the "Hadamard product" -- or outer product, rather
than matrix/inner product like our operator).  ``@`` at least has the
virtue that it *looks* like a funny non-commutative operator; a naive
user who knows maths but not programming couldn't look at ``A * B``
versus ``A × B``, or ``A * B`` versus ``A ⋅ B``, or ``A * B`` versus
``A ° B`` and guess which one is the usual multiplication, and which
one is the special case.

Finally, there is the option of using multi-character tokens.  Some
options:

* Matlab and Julia use a ``.*`` operator.  Aside from being visually
  confusable with ``*``, this would be a terrible choice for us
  because in Matlab and Julia, ``*`` means matrix multiplication and
  ``.*`` means elementwise multiplication, so using ``.*`` for matrix
  multiplication would make us exactly backwards from what Matlab and
  Julia users expect.

* APL apparently used ``+.×``, which by combining a multi-character
  token, confusing attribute-access-like . syntax, and a unicode
  character, ranks somewhere below U+2603 SNOWMAN on our candidate
  list.  If we like the idea of combining addition and multiplication
  operators as being evocative of how matrix multiplication actually
  works, then something like ``+*`` could be used -- though this may
  be too easy to confuse with ``*+``, which is just multiplication
  combined with the unary ``+`` operator.

* PEP 211 suggested ``~*``.  This has the downside that it sort of
  suggests that there is a unary ``*`` operator that is being combined
  with unary ``~``, but it could work.

* R uses ``%*%`` for matrix multiplication.  In R this forms part of a
  general extensible infix system in which all tokens of the form
  ``%foo%`` are user-defined binary operators.  We could steal the
  token without stealing the system.

* Some other plausible candidates that have been suggested: ``><`` (=
  ascii drawing of the multiplication sign ×); the footnote operator
  ``[*]`` or ``|*|`` (but when used in context, the use of vertical
  grouping symbols tends to recreate the nested parentheses visual
  clutter that was noted as one of the major downsides of the function
  syntax we're trying to get away from); ``^*``.

So, it doesn't matter much, but ``@`` seems as good or better than any
of the alternatives:

* It's a friendly character that Pythoneers are already used to typing
  in decorators, but the decorator usage and the math expression
  usage are sufficiently dissimilar that it would be hard to confuse
  them in practice.

* It's widely accessible across keyboard layouts (and thanks to its
  use in email addresses, this is true even of weird keyboards like
  those in phones).

* It's round like ``*`` and :math:`\cdot`.

* The mATrices mnemonic is cute.

* The swirly shape is reminiscent of the simultaneous sweeps over rows
  and columns that define matrix multiplication

* Its asymmetry is evocative of its non-commutative nature.

* Whatever, we have to pick something.


Precedence and associativity
----------------------------

There was a long discussion [#associativity-discussions]_ about
whether ``@`` should be right- or left-associative (or even something
more exotic [#group-associativity]_). Almost all Python operators are
left-associative, so following this convention would be the simplest
approach, but there were two arguments that suggested matrix
multiplication might be worth making right-associative as a special
case:

First, matrix multiplication has a tight conceptual association with
function application/composition, so many mathematically sophisticated
users have an intuition that an expression like :math:`R S x` proceeds
from right-to-left, with first :math:`S` transforming the vector
:math:`x`, and then :math:`R` transforming the result. This isn't
universally agreed (and not all number-crunchers are steeped in the
pure-math conceptual framework that motivates this intuition
[#oil-industry-versus-right-associativity]_), but at the least this
intuition is more common than for other operations like :math:`2 \cdot
3 \cdot 4` which everyone reads as going from left-to-right.

Second, if expressions like ``Mat @ Mat @ vec`` appear often in code,
then programs will run faster (and efficiency-minded programmers will
be able to use fewer parentheses) if this is evaluated as ``Mat @ (Mat
@ vec)`` then if it is evaluated like ``(Mat @ Mat) @ vec``.

However, weighing against these arguments are the following:

Regarding the efficiency argument, empirically, we were unable to find
any evidence that ``Mat @ Mat @ vec`` type expressions actually
dominate in real-life code. Parsing a number of large projects that
use numpy, we found that when forced by numpy's current funcall syntax
to choose an order of operations for nested calls to ``dot``, people
actually use left-associative nesting slightly *more* often than
right-associative nesting [#numpy-associativity-counts]_.  And anyway,
writing parentheses isn't so bad -- if an efficiency-minded programmer
is going to take the trouble to think through the best way to evaluate
some expression, they probably *should* write down the parentheses
regardless of whether they're needed, just to make it obvious to the
next reader that they order of operations matter.

In addition, it turns out that other languages, including those with
much more of a focus on linear algebra, overwhelmingly make their
matmul operators left-associative. Specifically, the ``@`` equivalent
is left-associative in R, Matlab, Julia, IDL, and Gauss. The only
exceptions we found are Mathematica, in which ``a @ b @ c`` would be
parsed non-associatively as ``dot(a, b, c)``, and APL, in which all
operators are right-associative. There do not seem to exist any
languages that make ``@`` right-associative and ``*``
left-associative. And these decisions don't seem to be controversial
-- I've never seen anyone complaining about this particular aspect of
any of these other languages, and the left-associativity of ``*``
doesn't seem to bother users of the existing Python libraries that use
``*`` for matrix multiplication. So, at the least we can conclude from
this that making ``@`` left-associative will certainly not cause any
disasters. Making ``@`` right-associative, OTOH, would be exploring
new and uncertain ground.

And another advantage of left-associativity is that it is much easier
to learn and remember that ``@`` acts like ``*``, than it is to
remember first that ``@`` is unlike other Python operators by being
right-associative, and then on top of this, also have to remember
whether it is more tightly or more loosely binding than
``*``. (Right-associativity forces us to choose a precedence, and
intuitions were about equally split on which precedence made more
sense. So this suggests that no matter which choice we made, no-one
would be able to guess or remember it.)

On net, therefore, the general consensus of the numerical community is
that while matrix multiplication is something of a special case, it's
not special enough to break the rules, and ``@`` should parse like
``*`` does.


(Non)-Definitions for built-in types
------------------------------------

No ``__matmul__`` or ``__matpow__`` are defined for builtin numeric
types (``float``, ``int``, etc.) or for the ``numbers.Number``
hierarchy, because these types represent scalars, and the consensus
semantics for ``@`` are that it should raise an error on scalars.

We do not -- for now -- define a ``__matmul__`` method on the standard
``memoryview`` or ``array.array`` objects, for several reasons.  Of
course this could be added if someone wants it, but these types would
require quite a bit of additional work beyond ``__matmul__`` before
they could be used for numeric work -- e.g., they have no way to do
addition or scalar multiplication either! -- and adding such
functionality is beyond the scope of this PEP.  In addition, providing
a quality implementation of matrix multiplication is highly
non-trivial.  Naive nested loop implementations are very slow and
shipping such an implementation in CPython would just create a trap
for users.  But the alternative -- providing a modern, competitive
matrix multiply -- would require that CPython link to a BLAS library,
which brings a set of new complications.  In particular, several
popular BLAS libraries (including the one that ships by default on
OS X) currently break the use of ``multiprocessing`` [#blas-fork]_.
Together, these considerations mean that the cost/benefit of adding
``__matmul__`` to these types just isn't there, so for now we'll
continue to delegate these problems to numpy and friends, and defer a
more systematic solution to a future proposal.

There are also non-numeric Python builtins which define ``__mul__``
(``str``, ``list``, ...).  We do not define ``__matmul__`` for these
types either, because why would we even do that.


Non-definition of matrix power
------------------------------

Earlier versions of this PEP also proposed a matrix power operator,
``@@``, analogous to ``**``.  But on further consideration, it was
decided that the utility of this was sufficiently unclear that it
would be better to leave it out for now, and only revisit the issue if
-- once we have more experience with ``@`` -- it turns out that ``@@``
is truly missed. [#atat-discussion]_


Rejected alternatives to adding a new operator
==============================================

Over the past few decades, the Python numeric community has explored a
variety of ways to resolve the tension between matrix and elementwise
multiplication operations.  PEP 211 and PEP 225, both proposed in 2000
and last seriously discussed in 2008 [#threads-2008]_, were early
attempts to add new operators to solve this problem, but suffered from
serious flaws; in particular, at that time the Python numerical
community had not yet reached consensus on the proper API for array
objects, or on what operators might be needed or useful (e.g., PEP 225
proposes 6 new operators with unspecified semantics).  Experience
since then has now led to consensus that the best solution, for both
numeric Python and core Python, is to add a single infix operator for
matrix multiply (together with the other new operators this implies
like ``@=``).

We review some of the rejected alternatives here.

**Use a second type that defines __mul__ as matrix multiplication:**
As discussed above (`Background: What's wrong with the status quo?`_),
this has been tried this for many years via the ``numpy.matrix`` type
(and its predecessors in Numeric and numarray).  The result is a
strong consensus among both numpy developers and developers of
downstream packages that ``numpy.matrix`` should essentially never be
used, because of the problems caused by having conflicting duck types
for arrays.  (Of course one could then argue we should *only* define
``__mul__`` to be matrix multiplication, but then we'd have the same
problem with elementwise multiplication.)  There have been several
pushes to remove ``numpy.matrix`` entirely; the only counter-arguments
have come from educators who find that its problems are outweighed by
the need to provide a simple and clear mapping between mathematical
notation and code for novices (see `Transparent syntax is especially
crucial for non-expert programmers`_).  But, of course, starting out
newbies with a dispreferred syntax and then expecting them to
transition later causes its own problems.  The two-type solution is
worse than the disease.

**Add lots of new operators, or add a new generic syntax for defining
infix operators:** In addition to being generally un-Pythonic and
repeatedly rejected by BDFL fiat, this would be using a sledgehammer
to smash a fly.  The scientific python community has consensus that
adding one operator for matrix multiplication is enough to fix the one
otherwise unfixable pain point. (In retrospect, we all think PEP 225
was a bad idea too -- or at least far more complex than it needed to
be.)

**Add a new @ (or whatever) operator that has some other meaning in
general Python, and then overload it in numeric code:** This was the
approach taken by PEP 211, which proposed defining ``@`` to be the
equivalent of ``itertools.product``.  The problem with this is that
when taken on its own terms, it's pretty clear that
``itertools.product`` doesn't actually need a dedicated operator.  It
hasn't even been deemed worth of a builtin.  (During discussions of
this PEP, a similar suggestion was made to define ``@`` as a general
purpose function composition operator, and this suffers from the same
problem; ``functools.compose`` isn't even useful enough to exist.)
Matrix multiplication has a uniquely strong rationale for inclusion as
an infix operator.  There almost certainly don't exist any other
binary operations that will ever justify adding any other infix
operators to Python.

**Add a .dot method to array types so as to allow "pseudo-infix"
A.dot(B) syntax:** This has been in numpy for some years, and in many
cases it's better than dot(A, B).  But it's still much less readable
than real infix notation, and in particular still suffers from an
extreme overabundance of parentheses.  See `Why should matrix
multiplication be infix?`_ above.

**Use a 'with' block to toggle the meaning of * within a single code
block**: E.g., numpy could define a special context object so that
we'd have::

    c = a * b   # element-wise multiplication
    with numpy.mul_as_dot:
        c = a * b  # matrix multiplication

However, this has two serious problems: first, it requires that every
array-like type's ``__mul__`` method know how to check some global
state (``numpy.mul_is_currently_dot`` or whatever).  This is fine if
``a`` and ``b`` are numpy objects, but the world contains many
non-numpy array-like objects.  So this either requires non-local
coupling -- every numpy competitor library has to import numpy and
then check ``numpy.mul_is_currently_dot`` on every operation -- or
else it breaks duck-typing, with the above code doing radically
different things depending on whether ``a`` and ``b`` are numpy
objects or some other sort of object.  Second, and worse, ``with``
blocks are dynamically scoped, not lexically scoped; i.e., any
function that gets called inside the ``with`` block will suddenly find
itself executing inside the mul_as_dot world, and crash and burn
horribly -- if you're lucky.  So this is a construct that could only
be used safely in rather limited cases (no function calls), and which
would make it very easy to shoot yourself in the foot without warning.

**Use a language preprocessor that adds extra numerically-oriented
operators and perhaps other syntax:** (As per recent BDFL suggestion:
[#preprocessor]_) This suggestion seems based on the idea that
numerical code needs a wide variety of syntax additions.  In fact,
given ``@``, most numerical users don't need any other operators or
syntax; it solves the one really painful problem that cannot be solved
by other means, and that causes painful reverberations through the
larger ecosystem.  Defining a new language (presumably with its own
parser which would have to be kept in sync with Python's, etc.), just
to support a single binary operator, is neither practical nor
desireable.  In the numerical context, Python's competition is
special-purpose numerical languages (Matlab, R, IDL, etc.).  Compared
to these, Python's killer feature is exactly that one can mix
specialized numerical code with code for XML parsing, web page
generation, database access, network programming, GUI libraries, and
so forth, and we also gain major benefits from the huge variety of
tutorials, reference material, introductory classes, etc., which use
Python.  Fragmenting "numerical Python" from "real Python" would be a
major source of confusion.  A major motivation for this PEP is to
*reduce* fragmentation.  Having to set up a preprocessor would be an
especially prohibitive complication for unsophisticated users.  And we
use Python because we like Python!  We don't want
almost-but-not-quite-Python.

**Use overloading hacks to define a "new infix operator" like *dot*,
as in a well-known Python recipe:** (See: [#infix-hack]_) Beautiful is
better than ugly.  This is... not beautiful.  And not Pythonic.  And
especially unfriendly to beginners, who are just trying to wrap their
heads around the idea that there's a coherent underlying system behind
these magic incantations that they're learning, when along comes an
evil hack like this that violates that system, creates bizarre error
messages when accidentally misused, and whose underlying mechanisms
can't be understood without deep knowledge of how object oriented
systems work.

**Use a special "facade" type to support syntax like arr.M * arr:**
This is very similar to the previous proposal, in that the ``.M``
attribute would basically return the same object as ``arr *dot` would,
and thus suffers the same objections about 'magicalness'.  This
approach also has some non-obvious complexities: for example, while
``arr.M * arr`` must return an array, ``arr.M * arr.M`` and ``arr *
arr.M`` must return facade objects, or else ``arr.M * arr.M * arr``
and ``arr * arr.M * arr`` will not work.  But this means that facade
objects must be able to recognize both other array objects and other
facade objects (which creates additional complexity for writing
interoperating array types from different libraries who must now
recognize both each other's array types and their facade types).  It
also creates pitfalls for users who may easily type ``arr * arr.M`` or
``arr.M * arr.M`` and expect to get back an array object; instead,
they will get a mysterious object that throws errors when they attempt
to use it.  Basically with this approach users must be careful to
think of ``.M*`` as an indivisible unit that acts as an infix operator
-- and as infix-operator-like token strings go, at least ``*dot*``
is prettier looking (look at its cute little ears!).


Discussions of this PEP
=======================

Collected here for reference:

* Github pull request containing much of the original discussion and
  drafting: https://github.com/numpy/numpy/pull/4351

* sympy mailing list discussions of an early draft:

  * https://groups.google.com/forum/#!topic/sympy/22w9ONLa7qo
  * https://groups.google.com/forum/#!topic/sympy/4tGlBGTggZY

* sage-devel mailing list discussions of an early draft:
  https://groups.google.com/forum/#!topic/sage-devel/YxEktGu8DeM

* 13-Mar-2014 python-ideas thread:
  https://mail.python.org/pipermail/python-ideas/2014-March/027053.html

* numpy-discussion thread on whether to keep ``@@``:
  http://mail.scipy.org/pipermail/numpy-discussion/2014-March/069448.html

* numpy-discussion threads on precedence/associativity of ``@``:
  * http://mail.scipy.org/pipermail/numpy-discussion/2014-March/069444.html
  * http://mail.scipy.org/pipermail/numpy-discussion/2014-March/069605.html


References
==========

.. [#preprocessor] From a comment by GvR on a G+ post by GvR; the
   comment itself does not seem to be directly linkable: https://plus.google.com/115212051037621986145/posts/hZVVtJ9bK3u
.. [#infix-hack] http://code.activestate.com/recipes/384122-infix-operators/
   http://www.sagemath.org/doc/reference/misc/sage/misc/decorators.html#sage.misc.decorators.infix_operator
.. [#scipy-conf] http://conference.scipy.org/past.html
.. [#pydata-conf] http://pydata.org/events/
.. [#lht] In this formula, :math:`\beta` is a vector or matrix of
   regression coefficients, :math:`V` is the estimated
   variance/covariance matrix for these coefficients, and we want to
   test the null hypothesis that :math:`H\beta = r`; a large :math:`S`
   then indicates that this hypothesis is unlikely to be true. For
   example, in an analysis of human height, the vector :math:`\beta`
   might contain one value which was the the average height of the
   measured men, and another value which was the average height of the
   measured women, and then setting :math:`H = [1, -1], r = 0` would
   let us test whether men and women are the same height on
   average. Compare to eq. 2.139 in
   http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/xegbohtmlnode17.html

   Example code is adapted from https://github.com/rerpy/rerpy/blob/0d274f85e14c3b1625acb22aed1efa85d122ecb7/rerpy/incremental_ls.py#L202

.. [#pycon-tutorials] Out of the 36 tutorials scheduled for PyCon 2014
   (https://us.pycon.org/2014/schedule/tutorials/), we guess that the
   8 below will almost certainly deal with matrices:

   * Dynamics and control with Python

   * Exploring machine learning with Scikit-learn

   * How to formulate a (science) problem and analyze it using Python
     code

   * Diving deeper into Machine Learning with Scikit-learn

   * Data Wrangling for Kaggle Data Science Competitions – An etude

   * Hands-on with Pydata: how to build a minimal recommendation
     engine.

   * Python for Social Scientists

   * Bayesian statistics made simple

   In addition, the following tutorials could easily involve matrices:

   * Introduction to game programming

   * mrjob: Snakes on a Hadoop *("We'll introduce some data science
     concepts, such as user-user similarity, and show how to calculate
     these metrics...")*

   * Mining Social Web APIs with IPython Notebook

   * Beyond Defaults: Creating Polished Visualizations Using Matplotlib

   This gives an estimated range of 8 to 12 / 36 = 22% to 33% of
   tutorials dealing with matrices; saying ~20% then gives us some
   wiggle room in case our estimates are high.

.. [#sloc-details] SLOCs were defined as physical lines which contain
   at least one token that is not a COMMENT, NEWLINE, ENCODING,
   INDENT, or DEDENT.  Counts were made by using ``tokenize`` module
   from Python 3.2.3 to examine the tokens in all files ending ``.py``
   underneath some directory.  Only tokens which occur at least once
   in the source trees are included in the table.  The counting script
   is available `in the PEP repository
   <http://hg.python.org/peps/file/tip/pep-0465/scan-ops.py>`_.

   Matrix multiply counts were estimated by counting how often certain
   tokens which are used as matrix multiply function names occurred in
   each package.  This creates a small number of false positives for
   scikit-learn, because we also count instances of the wrappers
   around ``dot`` that this package uses, and so there are a few dozen
   tokens which actually occur in ``import`` or ``def`` statements.

   All counts were made using the latest development version of each
   project as of 21 Feb 2014.

   'stdlib' is the contents of the Lib/ directory in commit
   d6aa3fa646e2 to the cpython hg repository, and treats the following
   tokens as indicating matrix multiply: n/a.

   'scikit-learn' is the contents of the sklearn/ directory in commit
   69b71623273ccfc1181ea83d8fb9e05ae96f57c7 to the scikit-learn
   repository (https://github.com/scikit-learn/scikit-learn), and
   treats the following tokens as indicating matrix multiply: ``dot``,
   ``fast_dot``, ``safe_sparse_dot``.

   'nipy' is the contents of the nipy/ directory in commit
   5419911e99546401b5a13bd8ccc3ad97f0d31037 to the nipy repository
   (https://github.com/nipy/nipy/), and treats the following tokens as
   indicating matrix multiply: ``dot``.

.. [#blas-fork] BLAS libraries have a habit of secretly spawning
   threads, even when used from single-threaded programs.  And threads
   play very poorly with ``fork()``; the usual symptom is that
   attempting to perform linear algebra in a child process causes an
   immediate deadlock.

.. [#threads-2008] http://fperez.org/py4science/numpy-pep225/numpy-pep225.html

.. [#broadcasting] http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

.. [#matmul-other-langs] http://mail.scipy.org/pipermail/scipy-user/2014-February/035499.html

.. [#github-details] Counts were produced by manually entering the
   string ``"import foo"`` or ``"from foo import"`` (with quotes) into
   the Github code search page, e.g.:
   https://github.com/search?q=%22import+numpy%22&ref=simplesearch&type=Code
   on 2014-04-10 at ~21:00 UTC.  The reported values are the numbers
   given in the "Languages" box on the lower-left corner, next to
   "Python".  This also causes some undercounting (e.g., leaving out
   Cython code, and possibly one should also count HTML docs and so
   forth), but these effects are negligible (e.g., only ~1% of numpy
   usage appears to occur in Cython code, and probably even less for
   the other modules listed).  The use of this box is crucial,
   however, because these counts appear to be stable, while the
   "overall" counts listed at the top of the page ("We've found ___
   code results") are highly variable even for a single search --
   simply reloading the page can cause this number to vary by a factor
   of 2 (!!).  (They do seem to settle down if one reloads the page
   repeatedly, but nonetheless this is spooky enough that it seemed
   better to avoid these numbers.)

   These numbers should of course be taken with multiple grains of
   salt; it's not clear how representative Github is of Python code in
   general, and limitations of the search tool make it impossible to
   get precise counts.  AFAIK this is the best data set currently
   available, but it'd be nice if it were better.  In particular:

   * Lines like ``import sys, os`` will only be counted in the ``sys``
     row.

   * A file containing both ``import X`` and ``from X import`` will be
     counted twice

   * Imports of the form ``from X.foo import ...`` are missed.  We
     could catch these by instead searching for "from X", but this is
     a common phrase in English prose, so we'd end up with false
     positives from comments, strings, etc.  For many of the modules
     considered this shouldn't matter too much -- for example, the
     stdlib modules have flat namespaces -- but it might especially
     lead to undercounting of django, scipy, and twisted.

   Also, it's possible there exist other non-stdlib modules we didn't
   think to test that are even more-imported than numpy -- though we
   tried quite a few of the obvious suspects.  If you find one, let us
   know!  The modules tested here were chosen based on a combination
   of intuition and the top-100 list at pypi-ranking.info.

   Fortunately, it doesn't really matter if it turns out that numpy
   is, say, merely the *third* most-imported non-stdlib module, since
   the point is just that numeric programming is a common and
   mainstream activity.

   Finally, we should point out the obvious: whether a package is
   import**ed** is rather different from whether it's import**ant**.
   No-one's claiming numpy is "the most important package" or anything
   like that.  Certainly more packages depend on distutils, e.g., then
   depend on numpy -- and far fewer source files import distutils than
   import numpy.  But this is fine for our present purposes.  Most
   source files don't import distutils because most source files don't
   care how they're distributed, so long as they are; these source
   files thus don't care about details of how distutils' API works.
   This PEP is in some sense about changing how numpy's and related
   packages' APIs work, so the relevant metric is to look at source
   files that are choosing to directly interact with that API, which
   is sort of like what we get by looking at import statements.

.. [#hugunin] The first such proposal occurs in Jim Hugunin's very
   first email to the matrix SIG in 1995, which lays out the first
   draft of what became Numeric. He suggests using ``*`` for
   elementwise multiplication, and ``%`` for matrix multiplication:
   https://mail.python.org/pipermail/matrix-sig/1995-August/000002.html

.. [#atat-discussion] http://mail.scipy.org/pipermail/numpy-discussion/2014-March/069502.html

.. [#associativity-discussions]
   http://mail.scipy.org/pipermail/numpy-discussion/2014-March/069444.html
   http://mail.scipy.org/pipermail/numpy-discussion/2014-March/069605.html

.. [#oil-industry-versus-right-associativity]
   http://mail.scipy.org/pipermail/numpy-discussion/2014-March/069610.html

.. [#numpy-associativity-counts]
   http://mail.scipy.org/pipermail/numpy-discussion/2014-March/069578.html

.. [#group-associativity]
   http://mail.scipy.org/pipermail/numpy-discussion/2014-March/069530.html


Copyright
=========

This document has been placed in the public domain.
