PEP: XXXX
Title: Dedicated infix operators for matrix multiplication and matrix power
Version: $Revision$
Last-Modified: $Date$
Author: Nathaniel J. Smith <njs@pobox.com>
Status: Draft
Type: Standards Track
Python-Version: 3.5
Content-Type: text/x-rst
Created: 20-Feb-2014
Post-History:

[NOTE NOTE NOTE]
================

[This document is currently a draft.  It's being posted because we
want **your** feedback.  Yes, **you**.  *Even* if you're 'just a
user', or 'just' the author of some little obscure project that's only
been downloaded 13 times (and 9 of those were you playing around with
pip, and the other 4 are your office-mates).  *Even* if you're not
'really' a programmer, but only a
scientist/roboticist/statistician/financial modeller/whatever, and
don't want to bother the 'real' programmers while they do... whatever
it is they do (and why do they keep talking about ducks?).  *Even* if
your lab has been feuding with the numpy developers for the last 3
generations over some ufunc-related mishap that corrupted your
advisor's advisor's favorite data [#feud]_.  *Even* if you think we're
a bunch of idiots.  Actually, especially if you think we're a bunch of
idiots.  We want this document to reflect the consensus of -- and
serve the needs of -- the *whole* Python numerical/mathematical
ecosystem.  We've probably missed important things due to our limited
perspective.  Nothing here is finalized.  So please do send feedback.
Some appropriate ways to reach us:

* This Github PR: https://github.com/numpy/numpy/pull/4351 (this is
  also where you can view the most up-to-date draft)

* Email to: njs@pobox.com

* python-dev, once this is posted there...

Of course, we can't guarantee that your brilliant suggestion will
actually be incorporated, because it probably contradicts three other
people's brilliant suggestions, and somehow we have to agree on
something to actually propose.  Life is full of compromises.  But
we'll do our best.

Note that we especially would like feedback on the proposed
`Semantics`_, and in particular if they look good to you and you
maintain a library that implements some matrix-like type, then we'd
love to add your project to the list in the `Adoption`_ section, as
further evidence of what a big happy family we all are.

Now, without further ado:]


Abstract
========

This PEP proposes two new binary operators dedicated to matrix
multiplication and matrix power, spelled ``@`` and ``@@``
respectively.  (Mnemonic: ``@`` is ``*`` for mATrices.)


Specification
=============

Two new binary operators are added to the Python language, together
with corresponding in-place versions:

=======  ========================= ===============================
 Op      Precedence/associativity     Methods
=======  ========================= ===============================
``@``    Same as ``*``             ``__matmul__``, ``__rmatmul__``
``@@``   Same as ``**``            ``__matpow__``, ``__rmatpow__``
``@=``   n/a                       ``__imatmul__``
``@@=``  n/a                       ``__imatpow__``
=======  ========================= ===============================

No implementations of these methods are added to the builtin or
standard library types.  However, a number of projects have agreed on
consensus semantics for these operations; see `Intended usage
details`_ below.


Motivation
==========

Executive summary
-----------------

In numerical code, there are two important operations which compete
for use of the ``*`` operator: elementwise multiplication, and matrix
multiplication.  Most Python code uses ``*`` for the former, leaving
no operator for matrix multiplication; this leads to hard-to-read code
and API fragmentation.  Matrix multiplication has a combination of
features which together provide a uniquely compelling case for the
addition of a dedicated infix operator:

* ``@`` brings Python into alignment with universal notational
  practice across all fields of mathematics, science, and engineering.

* ``@`` greatly clarifies real-world code.

* ``@`` provides a smoother onramp for less experienced users, who are
  particularly harmed by the current API fragmentation.

* ``@`` benefits a substantial and growing fraction of the Python user
  community.

* ``@`` will be used frequently -- likely more frequently than ``//``
  or the bitwise operators.

* ``@`` helps the Python numerical community reduce fragmentation, by
  finally standardizing on a single duck type for all array-like
  objects.

And, given the existence of ``@``, it makes more sense than not to
have ``@@``, ``@=``, and ``@@=``, so they are added as well.


Background: What's wrong with the status quo?
---------------------------------------------

When it comes to crunching numbers on a computer, we usually have lots
and lots of numbers to deal with, and we want to be able to write down
simple operations that apply to large collections of numbers all at
once.  The *n-dimensional array* is the basic object that all popular
numeric computing environments use to make this possible.  Python has
a number of libraries that provide such arrays, with numpy being the
most prominent.

When working with arrays, there are two different ways we might want
to define multiplication.  One is elementwise multiplication, e.g.::

  [[1, 2],     [[11, 12],     [[1 * 11, 2 * 12],
   [3, 4]]  x   [13, 14]]  =   [3 * 13, 4 * 14]]

and the other is `matrix multiplication`_:

.. _matrix multiplication: https://en.wikipedia.org/wiki/Matrix_multiplication

::

  [[1, 2],     [[11, 12],     [[1 * 11 + 2 * 13, 1 * 12 + 2 * 14],
   [3, 4]]  x   [13, 14]]  =   [3 * 11 + 4 * 13, 3 * 12 + 4 * 14]]

Elementwise multiplication is useful because it fits the common
pattern for numerical code: it lets us easily and quickly perform a
basic operation (ordinary multiplication) on a large number of aligned
values without writing a slow and cumbersome ``for`` loop.  And this
works as part of a very general schema: when using the array objects
provided by numpy or other numerical libraries, all Python operators
work elementwise on arrays of all dimensionalities.  The result is
that simple formulas like ``a * b + c / d`` can be written and tested
using single numbers for the variables, but then used to efficiently
perform this calculation on large collections of numbers all at once.

Matrix multiplication, by comparison, is slightly more
special-purpose.  It's only defined on 2d arrays (also known as
"matrices"), and multiplication is the only operation that has a
meaningful "matrix" version -- "matrix addition" is the same as
elementwise addition; there is no such thing "matrix bitwise-or" or
"matrix floordiv"; "matrix division" can be defined but is not very
useful, etc.  However, matrix multiplication is still used very
heavily across all numerical application areas; mathematically, it's
one of the most fundamental operations there is.

Because Python currently contains only a single multiplication
operator ``*``, libraries providing array-like objects must decide:
either use ``*`` for elementwise multiplication, or use ``*`` for
matrix multiplication.  For some libraries -- those which have an
explicit focus on a specialized application area where only one of
these operations is used -- this may be an easy choice.  But it turns
out that when doing general-purpose number crunching, both operations
are used frequently, and there are major advantages to using infix
rather than function call syntax in both cases.  It is not at all
clear which convention is optimal; often it varies on a case-by-case
basis.

Nonetheless, network effects mean that it is very important that we
pick *just one* convention.  In numpy, for example, it is technically
possible to switch between the conventions, because numpy provides two
different types: for ``numpy.ndarray`` objects, ``*`` performs
elementwise multiplication, and matrix multiplication must use a
function call (``numpy.dot``).  For ``numpy.matrix`` objects, ``*``
performs matrix multiplication, and elementwise multiplication
requires function syntax.  Writing code using ``numpy.ndarray`` works
fine.  Writing code using ``numpy.matrix`` also works fine.  But
trouble begins as soon as we try to put these two pieces of code
together.  Code that expects an ``ndarray`` and gets a ``matrix``, or
vice-versa, will not work.  Keeping track of which functions expect
which types and converting back and forth all the time is impossible
to get right.  Functions that defensively try to handle both types as
input find themselves floundering into a swamp of ``isinstance`` and
``if`` statements.

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
libraries to continue to use the opposite convention (e.g.,
scipy.sparse, pyoperators, pyviennacl), and ``numpy.matrix`` itself
still gets used in introductory programming courses, often appears in
StackOverflow answers, and so forth.  Well-written libraries thus must
continue to be prepared to deal with both types of objects, and, of
course, are also stuck using unpleasant funcall syntax for matrix
multiplication.  These problems cannot be resolved within the
constraints of current Python syntax (see `Rejected alternatives to
adding a new operator`_ below).

This PEP proposes the minimum effective change to Python syntax that
will allow us to drain this swamp.  We split ``*`` into two operators,
just as was done for ``/``: ``*`` for elementwise multiplication, and
``@`` for matrix multiplication.  (Why not the reverse?  Because this
way is compatible with the existing consensus, and because it gives us
a consistent rule that all the built-in numeric operators also apply
in an elementwise manner to arrays; the reverse convention would lead
to more special cases.)

So that's why matrix multiplication can't just use ``*``.  Now, in the
the rest of this section, we'll explain why it nonetheless meets the
high bar for adding a new operator.


Why should matrix multiplication be infix?
------------------------------------------

Right now, most numerical code in Python uses syntax like
``numpy.dot(a, b)`` or ``a.dot(b)`` to perform matrix multiplication.
This obviously works, so what's the problem?

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
function ``numpy.dot``, or the ``.dot`` method on arrays. Neither
provides a particularly readable translation of the formula::

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

For yet more sophisticated programmers writing code that will be
reused, considerations of speed or numerical accuracy might lead us to
prefer some particular order of operations.  In the ``@`` examples we
could be certain that if we see something like ``(H @ V) @ H.T`` then
the parentheses must have been added intentionally to accomplish some
meaningful purpose; in the ``dot`` examples, it's impossible to know
which nesting decisions are important, and which are arbitrary.

``@`` dramatically improves matrix code usability on many axes.


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
multiplication, even though this type is buggy and heavily deprecated
by the rest of the numpy community for the fragmentation that it
causes.  This pedagogical use case is the *only* reason
``numpy.matrix`` has not been deprecated.  Adding ``@`` will benefit
both beginning and advanced users with better syntax; and furthermore,
it will allow both groups to standardize on the same notation from the
start, providing a smoother on-ramp to expertise.


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
numerical Python [#scipy-conf]_ [#pydata-conf]_.  At PyCon 2014, ~20% of
the tutorials will involve the use of matrices [#pycon-tutorials]_.

To quantify this further, we used Github's "search" function to look
at what modules are actually imported across a wide range of
real-world code (i.e., all the code on Github).  We checked for
imports of several popular stdlib modules, a variety of numeric
modules, and other extremely high-profile modules like django and lxml
(the latter of which is the #1 most downloaded package on PyPI)::

       Python source files on Github containing the given strings
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
    pylab                 36754            41063    77817         0.19
    scipy                 40829            28263    69092         0.17
    lxml                  19026            38061    57087         0.14
    zlib                  40486             6623    47109         0.11
    multiprocessing       25247            19850    45097         0.11
    requests              30896              560    31456         0.08
    jinja2                 8057            24047    32104         0.08
    twisted               13858             6404    20262         0.05
    gevent                11309             8529    19838         0.05
    pandas                14923             4005    18928         0.05
    sympy                  2779             9537    12316         0.03
    theano                 3654             1828     5482         0.01
    ================ ==========  ===============  =======  ===========

These numbers should be taken with several grains of salt (see
footnote for discussion: [#github-details]_), but, to the extent that
we can trust this data, ``numpy`` appears to be the most-imported
non-stdlib module in the entire Pythonverse; it's even more-imported
than such stdlib stalwarts as ``subprocess``, ``math``, ``pickle``,
and ``threading``.  And numpy users represent only a subset of the
broader numerical community that will benefit from the ``@`` operator.

In addition, there is some precedence for adding an infix operator to
handle a somewhat specialized arithmetic operation: the floor division
operator ``//``, like the bitwise operators, is very useful under
certain circumstances when performing exact calculations on discrete
values.  But it seems likely that there are many Python programmers
who have never had reason to use ``//`` (or, for that matter, the
bitwise operators).  ``@`` is no more niche than ``//``.

Matrices may once have been a niche data type restricted to Fortran
programs running in university labs and on military hardware, but
those days are long gone.


So ``@`` is good for matrix formulas, but how common are those really?
----------------------------------------------------------------------

We've seen that ``@`` makes matrix formulas dramatically easier to
work with for both experts and non-experts, that matrix formulas are
important in general, and that numerical libraries like numpy are used
by a substantial proportion of Python's user base.  But numerical
libraries aren't just about linear algebra, and being important
doesn't necessarily mean taking up a lot of code: if matrix formulas
only occured in one or two places in the average numerically-oriented
project, then it still wouldn't be worth adding a new operator.

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
``>=``).  When we include the stdlib into our comparisons, matrix
multiplication is still used more often in total than any of the
bitwise operators, and 2x as often as ``//``.  This is true even
though the stdlib, which contains a fair amount of integer arithmetic
and no matrix operations, makes up more than 80% of the combined code
base.

By coincidence, the numeric libraries make up approximately the same
proportion of the 'combined' codebase as numeric tutorials make up of
PyCon 2014's tutorial schedule, which suggests that the 'combined'
column may not be *wildly* unrepresentative of new Python code in
general.  While it's impossible to know for certain, from this data
it's plausible that across all Python code currently being written,
matrix multiplication is used more often than ``//`` and the bitwise
operations.


But isn't it weird to add an operator with no stdlib uses?
----------------------------------------------------------

It's certainly unusual (though ``Ellipsis`` was also added without any
stdlib uses), but the important thing is whether a change will benefit
users, not where the software is being downloaded from.  It's clear
from the above that ``@`` will be used, and used heavily.  And -- who
knows? -- perhaps someday the stdlib will contain an array type of
some sort.  This PEP only moves us closer to that possibility, by
helping the Python numerical community finally standardize on a single
duck type for all array-like objects.


Matrix power and in-place operators
-----------------------------------

The primary motivation for this PEP is ``@``; no-one cares terribly
much about the other proposed operators.  The matrix power operator
``@@`` is useful and well-defined, but not really necessary.  It is
included here for consistency: if we have an ``@`` that is analogous
to ``*``, then it would be weird and surprising to *not* have an
``@@`` that is analogous to ``**``.  Similarly, the in-place operators
``@=`` and ``@@=`` have limited utility -- it's more common to write
``a = (b @ a)`` than it is to write ``a = (a @ b)``, and it is not
generally possible to implement in-place matrix multiplication any
more efficiently than by making a full copy of the matrix -- but they
are included for completeness and symmetry.


Compatibility considerations
============================

Currently, the only legal use of the ``@`` token in Python code is at
statement beginning in decorators, and the token strings ``@@``,
``@=``, and ``@@=`` are entirely illegal.  The new operators are all
binary infix; therefore they cannot occur at statement beginning.
This means that no existing code will be broken by the addition of
these operators, and there is no possible parsing ambiguity between
decorator-@ and the new operators.

Another important kind of compatibility is the mental cost paid by
users to update their understanding of the Python language after this
change, particularly for users who do not work with matrices and thus
do not benefit.  Here again, ``@`` has minimal impact: even
comprehensive tutorials and references will only need to add a
sentence or two to fully document this PEP's changes.


Intended usage details
======================

This section is informative, rather than normative -- it documents the
consensus of a number of libraries that provide array- or matrix-like
objects on how the ``@`` and ``@@`` operators will be implemented.

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

The recommended semantics for ``@`` are:

* 0d (scalar) inputs raise an error.  Scalar * matrix multiplication
  is a mathematically and algorithmically distinct operation from
  matrix @ matrix multiplication; scalar * matrix multiplication
  should go through ``*`` instead of ``@``.

* 1d vector inputs are promoted to 2d by prepending or appending a '1'
  to the shape on the 'away' side, the operation is performed, and
  then the added dimension is removed from the output.  The result is
  that matrix @ vector and vector @ matrix are both legal (assuming
  compatible shapes), and both return 1d vectors; vector @ vector
  returns a scalar.  This is clearer with examples.  If ``arr(2, 3)``
  represents a 2x3 array, and ``arr(3)`` represents a 1d vector with 3
  elements, then:

  * ``arr(2, 3) @ arr(3, 1)`` is a regular matrix product, and returns
    an array with shape (2, 1), i.e., a column vector.

  * ``arr(2, 3) @ arr(3)`` performs the same computation as the
    previous (i.e., treats the 1d vector as a matrix containing a
    single **column**), but returns the result with shape (2,), i.e.,
    a 1d vector.

  * ``arr(1, 3) @ arr(3, 2)`` is a regular matrix product, and returns
    an array with shape (1, 2), i.e., a row vector.

  * ``arr(3) @ arr(3, 2)`` performs the same computation as the
    previous (i.e., treats the 1d vector as a matrix containing a
    single **row**), but returns the result with shape (2,), i.e., a
    1d vector.

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
  in practice.  No-one wants to explain to newbies why to solve the
  simplest linear system in the obvious way, they have to type
  ``(inv(A) @ b[:, np.newaxis]).flatten()``, or do OLS by typing
  ``solve(X.T @ X, X @ y[:, np.newaxis]).flatten()``; no-one wants to
  type ``(a[np.newaxis, :] @ a[:, np.newaxis])[0, 0]`` every time they
  compute an inner product, or ``(a[np.newaxis, :] @ Mat @ a[:,
  np.newaxis])[0, 0]`` for general quadratic forms.

* 2d inputs are conventional matrices, and treated in the obvious
  way -- ``arr(3, 4) @ arr(4, 5)`` returns an array with shape (3,
  5).

* For higher dimensional inputs, we treat the last two dimensions as
  being the dimensions of the matrices to multiply, and 'broadcast'
  across the other dimensions.  This provides a convenient way to
  quickly compute many matrix products in a single operation.  For
  example, ``arr(10, 2, 3) @ arr(10, 3, 4)`` performs 10 separate
  matrix multiplies, each of which multiplies a 2x3 and a 3x4 matrix
  to produce a 2x4 matrix, and then returns the 10 resulting matrices
  together in an array with shape (10, 2, 4).  Note that in more
  complicated cases, broadcasting allows several simple but powerful
  tricks for controlling how arrays are aligned with each other; see
  [#broadcasting]_ for details.  (In particular, it turns out that
  elementwise multiplication with broadcasting includes the standard
  scalar * matrix product as a special case, further motivating the
  use of ``*`` for this case.)

  If one operand is >2d, and another operand is 1d, then the above
  rules apply unchanged, with 1d->2d promotion performed before
  broadcasting.  E.g., ``arr(10, 2, 3) @ arr(3)`` first promotes to
  ``arr(10, 2, 3) @ arr(3, 1)``, then broadcasts to ``arr(10, 2, 3) @
  arr(10, 3, 1)``, multiplies to get an array with shape (10, 2, 1),
  and finally removes the added dimension, returning an array with
  shape (10, 2).  Similarly, ``arr(2) @ arr(10, 2, 3)`` produces an
  intermediate array with shape (10, 1, 3), and a final array with
  shape (10, 3).

The recommended semantics for ``@@`` are::

    def __matpow__(self, n):
        if not isinstance(n, numbers.Integral):
            raise TypeError("@@ not implemented for fractional powers")
        if n == 0:
            return identity_matrix_with_shape(self.shape)
        elif n < 0:
            return inverse(self) @ (self @@ (n + 1))
        else:
            return self @ (self @@ (n - 1))

(Of course we expect that much more efficient implementations will be
used in practice.)  Notice that if given an appropriate definition of
``identity_matrix_with_shape``, then this definition will
automatically handle >2d arrays appropriately.  Notice also that with
this definition, ``vector @@ 2`` gives the squared Euclidean length of
the vector, a commonly used value.  Also, while it is rarely useful to
explicitly compute inverses or other negative powers in standard
immediate-mode dense matrix code, these computations are natural when
doing symbolic or deferred-mode computations (as in e.g. sympy,
theano, numba, numexpr); therefore, negative powers are fully
supported.  Fractional powers, though, are somewhat more dicey in
general, so we leave it to individual projects to decide whether they
want to try to define some reasonable semantics for fractional inputs.


Adoption
--------

The following projects have expressed an intention to implement ``@``
and ``@@`` on their matrix-like types in a manner consistent with the
above definitions: numpy (+), scipy.sparse (+), pandas, blaze,
pyoperators (+?), pyviennacl (+).

In addition: (+) indicates projects which (a) currently have the
convention of using ``*`` for matrix multiplication in at least some
cases *and* (b) if this PEP is accepted, have expressed a goal of
migrating from this to the majority convention of elementwise-``*``,
matmul-``@``. I.e., each (+) indicates a reduction in cross-project
API fragmentation.

[And (+?) means that I think they probably count as (+), but need to
double check with the relevant devs.  More to check: Theano (emailed),
pycuda (emailed), panda3d (emailed devs directly), cvxopt (mostly
dead, but emailed), OpenCV (emailed, though I'm not sure if I sent it
to the right place), pysparse (appears to be totally dead).  Are there
any other libraries that define matrix types?  Is it worth trying to
talk to the PyQt people about QTransform?  PyOpenGL seems to assume
that if you want to do anything interesting with matrices you'll use
numpy.]


Partial- or Non-adoption
------------------------

The sympy and sage projects don't include elementwise multiplication
at all, and have no plans to add it.  This is consistent with their
approach of focusing on matrices as abstract mathematical objects
(i.e., linear maps over free modules over rings) rather than as big
bags full of numbers that need crunching.  They thus don't encounter
the problems this PEP addresses to solve, making it mostly irrelevant
to them; they define ``*`` to be matrix multiplication, and if this
PEP is accepted, plan to define ``@`` as an alias for ``*``.  So
technically this would be adoption of the semantics in this PEP, just
without full API convergence.


Rationale for specification details
===================================

Choice of operator
------------------

Why ``@`` instead of some other punctuation symbol? It doesn't matter
much, and there isn't any consensus across other programming languages
about how this operator should be named [#matmul-other-langs]_, but
``@`` has a few advantages:

* ``@`` is a friendly character that Pythoneers are already used to
  typing in decorators, and its use in email addresses means it is
  more likely to be easily accessible across keyboard layouts than
  some other characters (e.g. ``$`` or multibyte characters).

* The mATrices mnemonic is cute.

* It's round like ``*`` and :math:`\cdot`.

* The swirly shape is reminiscent of the simultaneous sweeps over rows
  and columns that define matrix multiplication.


(Non)-Definitions for built-ins
-------------------------------

No ``__matmul__`` or ``__matpow__`` are defined for builtin numeric
types (``float``, ``int``, etc.), because these are scalars, and the
consensus semantics for ``@`` are that it should raise an error on
scalars.

We do not (for now) define a ``__matmul__`` operator on the standard
``memoryview`` or ``array.array`` objects, for several reasons.  There
is currently no way to create multidimensional memoryview objects
using only the stdlib, and memoryview objects do not contain type
information needed to interpret their contents numerically (e.g., as
float32 versus int32).  Array objects are typed, but cannot represent
multidimensional data.  And finally, providing a quality
implementation of matrix multiplication is highly non-trivial.  The
naive nested loop implementation is very slow and providing it in the
Python core would just create a trap for users.  But the alternative
-- providing a modern, competitive matrix multiply -- would require
that Python link to a BLAS library, which brings a set of new
complications.  In particular, several popular BLAS libraries
(including the one that ships by default on OS X) currently break the
use of ``multiprocessing`` [#blas-fork]_.  Thus the Python core will
continue to delegate dealing with these problems to numpy and friends,
at least for now.

There are also non-numeric Python builtins which define ``__mul__``
(``str``, ``list``, ...).  We do not define ``__matmul__`` for these
types either, because why would we even do that.


Rejected alternatives to adding a new operator
==============================================

Over the past 15+ years, the Python numeric community has explored a
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

**Use a type that defines __mul__ as matrix multiplication:** As
discussed above (`Background: What's wrong with the status quo?`_),
this has been tried this for many years via the ``numpy.matrix`` type
(and its predecessors in Numeric and numarray).  The result is a
strong consensus among experienced numerical programmers that
``numpy.matrix`` should essentially never be used, because of the
problems caused by having conflicting duck types for arrays.  (Of
course one could then argue we should *only* define ``__mul__`` to be
matrix multiplication, but then we'd have the same problem with
elementwise multiplication.)  There have been several pushes to remove
``numpy.matrix`` entirely; the only argument against this has come
from educators who find that its problems are outweighed by the need
to provide a simple and clear mapping between mathematical notation
and code for novices (see `Transparent syntax is especially crucial
for non-expert programmers`_).  But, of course, starting out newbies
with a dispreferred syntax and then expecting them to transition later
causes its own problems.  This solution is worse than the disease.

**Add a new @ (or whatever) operator that has some other meaning
in general Python, and then overload it in numeric code:** This was
the approach proposed by PEP 211, which suggested defining ``@`` to be
the equivalent of ``itertools.product``.  The problem with this is
that when taken on its own terms, adding an infix operator for
``itertools.product`` is just silly.  (Similar arguments can be made
against the suggestion that arose during discussions this PEP, that
``@`` be defined as a general operator for function composition.)
Matrix multiplication has a uniquely strong rationale for inclusion as
an infix operator.  There almost certainly don't exist any other
binary operations that will ever justify adding another infix
operator.

**Add a .dot method to array types so as to allow "pseudo-infix"
A.dot(B) syntax:** This has been in numpy for some years, and in many
cases it's better than dot(A, B).  But it's still much less readable
than real infix notation, and in particular still suffers from an
extreme overabundance of parentheses.  See `Why should matrix
multiplication be infix?`_ above.

**Add lots of new operators / add a new generic syntax for defining
infix operators:** In addition to being generally un-Pythonic and
repeatedly rejected by BDFL fiat, this would be using a sledgehammer
to smash a fly.  There is a consensus in the scientific python
community that matrix multiplication really is the only missing infix
operator that matters enough to bother about. (In retrospect, we all
think PEP 225 was a bad idea too.)

**Use a 'with' block to toggle the meaning of * within a single code
block**: E.g., numpy could define a special context object so that
we'd have::

    c = a * b   # element-wise multiplication
    with numpy.mul_as_dot:
        c = a * b  # matrix multiplication

However, this has two serious problems: first, it requires that every
matrix-like object ``__mul__`` method know how to check some global
state (``numpy.mul_is_currently_dot`` or whatever).  This is fine if
``a`` and ``b`` are numpy objects, but the world contains many
non-numpy matrix-like objects.  So this either requires non-local
coupling -- every numpy competitor library has to import numpy and
then check ``numpy.mul_is_currently_dot`` on every operation -- or
else it breaks duck-typing, with the above code doing radically
different things depending on whether ``a`` and ``b`` are numpy
objects or some other sort of object.  Second, and worse, ``with``
blocks are dynamically scoped, not lexically scoped; i.e., any
function that gets called inside the ``with`` block will suddenly find
itself executing inside the mul_as_dot world, and crash and burn
horribly (if you're lucky).  So this is a construct that could only be
used safely in rather limited cases (no function calls), and which
would make it very easy to shoot yourself in the foot without warning.

**Use a language preprocessor that adds extra operators and perhaps
other syntax (as per recent BDFL suggestion):** (See: [#preprocessor]_) Aside
from matrix multiplication, there are no other operators or syntax
that anyone in the number-crunching community cares enough about to
bother adding.  But defining a new language (presumably with its own
parser which would have to be kept in sync with Python's, etc.), just
to support a single binary operator, is neither practical nor
desireable.  In the numerical context, Python's competition is
special-purpose numerical languages (Matlab, R, IDL, etc.).  Compared
to these, Python's killer feature is exactly that one can mix
specialized numerical code with code for XML parsing, web page
generation, database access, network programming, GUI libraries, etc.,
and we also gain major benefits from the huge variety of tutorials,
reference material, introductory classes, etc., which use Python.
Fragmenting "numerical Python" from "real Python" would be a major
source of confusion -- an a major motivation for this PEP is to
*reduce* fragmentation.  Having to set up a preprocessor would be an
especially prohibitive complication for unsophisticated users.  And we
use Python because we like Python!  We don't want
almost-but-not-quite-Python.

**Use overloading hacks to define a "new infix operator" like *dot*, as
in a well-known Python recipe:** (See: [#infix-hack]_) Beautiful is
better than ugly. This solution is so ugly that most developers will
simply refuse to consider it for use in serious, reusable code.  This
isn't just speculation -- a variant of this recipe is actually
distributed as a supported part of a major Python mathematics system
[#sage-infix]_, so it's widely available, yet still receives minimal
use.  OTOH, the fact that people even consider such a 'solution', and
are supporting it in shipping code, could be taken as further
evidence for the need for a proper infix operator for matrix product.

References
==========

.. [#preprocessor] From a comment by GvR on a G+ post by GvR; the
   comment itself does not seem to be directly linkable: https://plus.google.com/115212051037621986145/posts/hZVVtJ9bK3u
.. [#infix-hack] http://code.activestate.com/recipes/384122-infix-operators/
.. [#sage-infix] http://www.sagemath.org/doc/reference/misc/sage/misc/decorators.html#sage.misc.decorators.infix_operator
.. [#scipy-conf] http://conference.scipy.org/past.html
.. [#pydata-conf] http://pydata.org/events/
.. [#lht] In this formula, :math:`\beta` is a vector or matrix of
   regression coefficients, :math:`V` is the estimated
   variance/covariance matrix for these coefficients, and we want to
   test the null hypothesis that :math:`H\beta = r`; a large :math:`S`
   then indicates that this hypothesis is unlikely to be true. For
   example, in an analysis of human height, the vector :math:`\beta`
   might contain the average heights of men and women respectively,
   and then setting :math:`H = [1, -1], r = 0` would let us test
   whether men and women are the same height on average. Compare to
   eq. 2.139 in
   http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/xegbohtmlnode17.html

   Example code is adapted from https://github.com/rerpy/rerpy/blob/0d274f85e14c3b1625acb22aed1efa85d122ecb7/rerpy/incremental_ls.py#L202

.. [#pycon-tutorials] Out of the 36 tutorials scheduled for PyCon
   2014, we guess that the 8 below will almost certainly deal with
   matrices:

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

   See: https://us.pycon.org/2014/schedule/tutorials/

.. [#sloc-details] SLOCs were defined as physical lines which contain
   at least one token that is not a COMMENT, NEWLINE, ENCODING,
   INDENT, or DEDENT.  Counts were made by using ``tokenize`` module
   from Python 3.2.3 to examine the tokens in all files ending ``.py``
   underneath some directory.  Only tokens which occur at least once
   in the source trees are included in the table.  The counting script
   will be available as an auxiliary file once this PEP is submitted;
   until then, it can be found here:
   https://gist.github.com/njsmith/9157645

   Matrix multiply counts were estimated by counting how often certain
   tokens which are used as matrix multiply function names occurred in
   each package.  In principle this could create false positives, but
   as far as I know the counts are exact; it's unlikely that anyone is
   using ``dot`` as a variable name when it's also the name of one of
   the most widely-used numpy functions.

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

.. [#feud] Also, if this is true, then please file a bug: https://github.com/numpy/numpy/issues

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

   These numbers should of course be taken with a grain of salt; it's not
   clear how representative Github is of Python code in general, and
   limitations of the search tool make it impossible to get precise
   counts (in particular, a line like ``import sys, os`` will only be
   counted in the ``sys`` row; OTOH, files containing both ``import X``
   and ``from X import`` will be double-counted).  But AFAIK this is the
   best data set currently available.

   Also, it's possible there some other non-stdlib module I didn't
   think to test that is even more-imported than numpy -- though I
   tried quite a few of the obvious suspects.  If you find one, let me
   know!

   Modules tested were chosen based on a combination of intuition and
   the top-100 list at pypi-ranking.info.


Copyright
=========

This document has been placed in the public domain.
