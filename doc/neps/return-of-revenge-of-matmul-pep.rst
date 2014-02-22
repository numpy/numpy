PEP: XXXX
Title: Dedicated infix operators for matrix multiplication and matrix power
Version: $Revision$
Last-Modified: $Date$
Author: Nathaniel J. Smith <njs@pobox.com>
Status: Draft
Type: Standards Track
python-Version: 3.5
Content-Type: text/x-rst
Created: 20-Feb-2014
Post-History:

Abstract
========

This PEP proposes two new binary operators dedicated to matrix
multiplication and matrix power, spelled ``@`` and ``@@``
respectively.  (Mnemonic: ``@`` is ``*`` for mATrices.)


Specification
=============

Two new binary operators are added, together with corresponding
in-place versions:

=======  ========================= ===============================
 Op      Precedence/associativity     Methods
=======  ========================= ===============================
``@``    Same as ``*``             ``__matmul__``, ``__rmatmul__``
``@@``   Same as ``**``            ``__matpow__``, ``__rmatpow__``
``@=``   n/a                       ``__imatmul__``
``@@=``  n/a                       ``__imatpow__``
=======  ========================= ===============================

The intention is that these will be overridden by numpy (and other
libraries that define array-like objects, e.g. pandas, Theano,
scipy.sparse, blaze, OpenCV, ...) to perform matrix multiplication, in
contrast with ``*``'s elementwise multiplication.

For scalar/scalar operations, matrix, scalar, and elementwise
multiplication all coincide, so we also add the following methods to
``numbers.Complex`` and all built-in numeric types::

    def __matmul__(self, other):
        if isinstance(other, numbers.Number):
            return self * other
        else:
            return NotImplemented

    def __matpow__(self, other):
        return self ** other

    # The reverse version isn't really needed given the above, but
    # doesn't hurt either, and improves forwards compatibility with
    # any 3rd-party numbers.Number types that merely .register as
    # subclasses of Complex without actually inheriting.
    def __rmatmul__(self, other):
        if isinstance(other, numbers.Number):
            return other * self
        else:
            return NotImplemented

    # Likewise.
    def __rmatpow__(self, other):
        if isinstance(other, numbers.Number):
            return other ** self
        else:
            return NotImplemented

And for builtin types, statements like ``a @= b`` will perform
``a = (a @ b)`` via the usual mechanism.


Motivation
==========

The main motivation for this PEP is the addition of a binary operator
``@`` for matrix multiplication.  No-one cares terribly much about the
matrix power operator ``@@`` -- it's useful and well-defined, but not
really necessary.  It is included here for aesthetic reasons: if we
have an ``@`` that is like ``*``, then it would be weird and
surprising to *not* have an ``@@`` that is like ``**``.  Similarly,
the in-place operators ``@=`` and ``@@=`` are of marginal utility --
it is not generally possible to implement in-place matrix
multiplication any more efficiently than by doing ``a = (a @ b)`` --
but are included for completeness and symmetry. So let's focus on the
motivation for ``@``; everything else follows from that.


Why should matrix multiplication be infix?
------------------------------------------

When moving from scalars -- like ordinary Python floats -- to more
general n-dimensional arrays and matrices, there are two standard ways
to generalize the usual multiplication operation.  One is elementwise
multiplication::

  [2, 3] * [4, 5] = [2 * 4, 3 * 5] = [8, 15]

and the other is the `matrix product`_.  For various reasons, the
numerical Python ecosystem has universally settled on the convention
that ``*`` refers to elementwise multiplication.  However, this leaves
us with no convenient notation for matrix multiplication.

.. _matrix product: https://en.wikipedia.org/wiki/Matrix_multiplication

Matrix multiplication is similar to ordinary arithmetic operations
like addition and scalar multiplication in two ways: (a) it is used
very heavily in numerical programs -- often multiple times per line of
code -- and (b) it has an ancient and universally adopted tradition of
being written using infix syntax with varying precedence.  This is
because, for typical formulas, this notation is dramatically more
readable than any function syntax.  For example, one of the most
useful tools for testing a statistical hypothesis is the linear
hypothesis test for OLS regression models. If we want to implement
this, we will look up some textbook or paper on it, and encounter many
mathematical formulas that look like:

.. math::

    S = (H \beta - r)^T (H V H^T)^{-1} (H \beta - r)

Here the various variables are all vectors or matrices (details for
the curious: [#lht]).

Our job is to write code to perform this calculation. In
current numpy, matrix multiplication can be performed using either the
function numpy.dot, or the .dot method on arrays. Neither provides a
particularly readable translation of the formula::

    import numpy as np
    from numpy.linalg import inv, solve

    # Using dot function:
    S = np.dot((np.dot(H, beta) - r).T,
               np.dot(inv(np.dot(np.dot(H, V), H.T)), np.dot(H, beta) - r))

    # Using dot method:
    S = (H.dot(beta) - r).T.dot(inv(H.dot(V).dot(H.T))).dot(H.dot(beta) - r)

With the ``@`` operator, the direct translation of the above formula is::

    S = (H @ beta - r).T @ inv(H @ V @ H.T) @ (H @ beta - r)

Notice that there is now a transparent, 1-to-1 mapping between symbols
in the original formula and the code.

Of course, a more sophisticated programmer will probably notice that
this is not the best way to compute this expression. The repeated
computation of :math:`H \beta - r` should perhaps be factored out;
and, expressions of the form ``dot(inv(A), B)`` should almost always
be replaced by the more numerically stable ``solve(A, B)``.  When
using ``@``, performing these refactorings give us::

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
relatively simple formula like this one.  I made and caught many
errors while trying to write out the 'dot' formulas above, and I'm
still not certain I got all the parentheses right.  (Exercise: check
my parentheses.)  But the ``@`` examples are obviously correct.


Importance for teaching
-----------------------

A large proportion of scientific code is written by people who are
experts in their domain, but not experts in programming.  And there
are many university courses with titles like "Data analysis for social
scientists" which assume no programming background, and teach some
combination of mathematical techniques, introduction to programming,
and the use of programming to implement these mathematical techniques,
all within a 10-15 week period.  These courses are more and more often
being taught in Python rather than special-purpose languages like R or
Matlab.

For these kinds of users, whose programming knowledge is fragile, the
existence of a transparent mapping between formulas and code often
means the difference between succeeding and failing to write that code
at all.  This is so important that such classes often use the
``numpy.matrix`` type which defines ``*`` to mean matrix
multiplication, even though this type is buggy and heavily deprecated
by the rest of the numpy community.  Adding ``@`` will benefit both
beginning and advanced users; and furthermore, it will allow both
groups to standardize on the same notation, providing a smoother
on-ramp to expertise.


But isn't matrix multiplication a pretty niche requirement?
-----------------------------------------------------------

The world is full of continuous data, and computers are increasingly
called upon to work with it in sophisticated ways.  Matrices are the
lingua franca of finance, machine learning, 3d graphics, computer
vision, robotics, operations research, econometrics, meteorology,
computational linguistics, recommendation systems, neuroscience,
bioinformatics (including genetics, cancer research, drug discovery,
etc.), physics simulation, quantum mechanics, network analysis, and
many other application areas.

In most or all of these areas, Python is rapidly becoming a dominant
player, in large part because of its ability to elegantly mix
traditional discrete data structures (hash tables, strings, etc.) on
an equal footing with modern numerical data types and algorithms.  In
2013, there were 7 international conferences specifically on numerical
Python [#scipy-conf][#pydata-conf], and ~20% of the PyCon 2014
tutorials will involve the use of matrices [#pycon-tutorials].
Matrices may once have been a niche data type restricted to university
labs using Fortran, but those days are long gone.

In addition, there is some precedence for adding an infix operator to
handle a somewhat specialized arithmetic operation: "floor division"
(``//``), like the bitwise operators, is very useful under certain
circumstances when performing exact calculations on discrete values,
but it seems likely that there are many Python programmers who have
never used ``//``.  ``@`` is no more niche than ``//``.


So ``@`` is good for matrix formulas, but how common are those really?
----------------------------------------------------------------------

We've seen that ``@`` makes matrix formulas dramatically easier to
work with, and that matrix formulas are extremely important in
general.  But being important doesn't necessarily mean taking up a lot
of code: if such formulas only occur in one or two places in the
average numerically-oriented project, then it still might not be worth
adding a new operator.

When the going gets tough, the tough get empirical.  To get a rough
estimate of how useful the ``@`` operator will be, this table shows
the rate at which different Python operators are used in the stdlib,
and also in two high-profile numerical projects -- the sklearn machine
learning library, and the nipy neuroimaging library.  Units are
(rounded) usages per 10,000 source lines of code (SLOC).  Rows are
sorted by the 'combined' column, which gives the usage per 10,000 SLOC
when the three code bases are pooled together.  The combined column is
thus strongly weighted towards the stdlib, which is much larger than
both projects put together (stdlib: 411575 SLOC, sklearn: 50924 SLOC,
nipy: 37078 SLOC). [#sloc-details]

The ``dot`` row counts matrix multiply operations, estimated by
assuming there to be zero matrix multiplies in the stdlib, and in
sklearn/nipy assuming -- reasonably -- that all instances of the token
``dot`` are calls to ``np.dot``.

======= ======= ======= ======= ========
     Op  stdlib sklearn    nipy combined
======= ======= ======= ======= ========
  ``(``    6979    6861    7644     7016
  ``)``    6979    6861    7644     7016
  ``=``    2969    5536    4932     3376
  ``-``     218     444     496      261
  ``+``     224     201     348      231
 ``==``     177     248     334      196
  ``*``     156     284     465      192
  ``%``     121     114     107      119
  ``}``     106      56      63       98
  ``{``     106      56      63       98
 ``**``      59     111     118       68
 ``!=``      40      56      74       44
  ``/``      18     121     183       41
  ``>``      29      70     110       39
 ``+=``      34      61      67       39
  ``<``      32      62      76       38
 ``>=``      19      17      17       18
 ``<=``      18      27      12       18
  ``|``      18       1       2       15
``dot``       0      80      74       14
  ``&``      14       0       6       12
 ``<<``      10       1       1        8
 ``//``       9       9       1        8
 ``-=``       5      21      14        8
 ``*=``       2      19      22        5
 ``/=``       0      23      16        4
 ``>>``       4       0       0        3
  ``^``       3       0       0        3
  ``~``       2       4       5        2
 ``|=``       3       0       0        2
 ``&=``       1       0       0        1
``//=``       1       0       0        1
 ``^=``       1       0       0        0
``**=``       0       2       0        0
 ``%=``       0       0       0        0
``<<=``       0       0       0        0
``>>=``       0       0       0        0
======= ======= ======= ======= ========

We see that sklearn and nipy together contain nearly 700 uses of
matrix multiplication.  Within these two libraries, matrix
multiplication is used more heavily than most comparison operators
(``<`` ``>`` ``!=`` ``<=`` ``>=``), and more heavily even than ``{``
and ``}``.  In total across all three of the codebases examined here,
matrix multiplication is used more often than almost all the bitwise
operators (only ``|`` just barely edges it out), and ~2x as often as
``//``.  This is true even though the stdlib, which contains a fair
amount of integer arithmetic and no matrix operations, is ~4x larger
than the numeric libraries put together.  While it's impossible to
know for certain, from this data it seems plausible that on net across
the whole Python ecosystem, matrix multiplication is currently used
more often than ``//`` or other integer operations.


But isn't it weird to add an operator with no stdlib uses?
----------------------------------------------------------

It's certainly unusual, but the important thing is whether a change
will benefit users, not where the software is being downloaded from.
It's clear from the above that ``@`` will be used, and used heavily.
And -- who knows? -- perhaps someday the stdlib will contain a matrix
type of some sort.  This PEP only moves us closer to that possibility,
by helping the Python numerical community finally standardize on a
single duck type for all matrix-like objects.


Summary
-------

Matrix multiplication is uniquely deserving of a new, dedicated infix
operator.  The addition of ``@`` will:

* bring Python into alignment with universal notational practice
  across all fields of mathematics, science, and engineering,

* greatly clarify a large quantity of real-world code,

* provide a smoother onramp for new users,

* benefit a large and growing user community,

* and help this community finally standardize on a single duck type
  for all matrix-like objects.


Compatibility considerations
============================

Currently, the only legal use of the ``@`` token in Python code is at
statement beginning in decorators.  Therefore no code will be broken
by the addition of these operators.

Another important kind of compatibility is the mental cost paid by
users to update their understanding of the Python language after this
change, particularly for users who do not work with matrices and thus
do not benefit.  Here again, ``@`` has minimal impact: even
comprehensive tutorials and references will only need to add a
sentence or two to fully document this PEP's changes.


Rationale
=========

Alternative ways to go about adding a matrix multiplication operator
--------------------------------------------------------------------

Choice of operator
''''''''''''''''''

Why ``@`` instead of some other punctuation symbol? It doesn't matter
much, but ``@`` has a few advantages:

* ``@`` is a friendly character that Pythoneers are already used to
  typing in decorators, and its use in email addresses means it is
  more likely to be easily accessible across keyboard layouts than
  some other characters (e.g. $).
* The mATrices mnemonic is cute.
* The swirly shape is reminiscent of the simultaneous sweeps over rows
  and columns that define matrix multiplication.


Built-ins
'''''''''

Why are the new special methods defined the way they are for Python
builtins? The three goals are:

* Define a meaningful ``@`` and ``@@`` for builtin and user-defined
  numeric types, to maximize duck compatibility between Python scalars
  and 1x1 matrices, single-element vectors, and zero-dimensional
  arrays.
* Do this in as forward-compatible a way as possible.
* Ensure that ``scalar @ matrix`` does *not* delegate to ``scalar *
  matrix``; ``scalar * matrix`` is well-defined, but ``scalar @
  matrix`` should raise an error.

Therefore, we implement these methods so that numbers.Number objects
will in general delegate ``@`` to ``*``, but only when dealing with
other numbers.Number objects. In other cases NotImplemented is
returned.

An alternative approach would be for these methods on builtin types to
always return NotImplemented.  It probably doesn't make much
difference which we choose, since we still won't have full duck
compatibility between Python builtins and numpy scalars (e.g.,
builtins will still miss the very common ``.T`` transpose operator).
But the approach taken here seems marginally more semantically
consistent.

We do not (for now) define a ``__matmul__`` operator on the standard
``memoryview`` or ``array.array`` objects, for several reasons.  There
is currently no way to create multidimensional memoryview objects
using only the stdlib, and memoryview objects do not contain type
information needed to interpret their contents numerically (e.g., as
float32 versus int32).  Array objects are typed, but cannot represent
multidimensional data.  And finally, providing a quality
implementation of matrix multiplication is highly non-trivial.  The
naive nested loop implementation is very slow and would become an
attractive nuisance; but, providing a competitive matrix multiply
would require that Python link to a BLAS library, which brings a set
of new complications -- among them that several popular BLAS libraries
(including the one that ships by default on OS X) currently break the
use of ``multiprocessing`` [#blas-fork].  Thus we'll continue to
delegate dealing with these problems to numpy and friends, at least
for now.

While there are non-numeric Python builtins that define ``__mul__``
(``str``, ``list``, ...), we do not define ``__matmul__`` for these
types either, because that makes no sense and TOOWTDI.


Alternatives to adding a new operator at all
--------------------------------------------

Over the past 15+ years, the Python numeric community has explored a
variety of ways to handle the tension between matrix and elementwise
multiplication operations.  PEP 211 and PEP 225, both proposed in
2000, were early attempts to add new operators to solve this problem,
but suffered from serious flaws; in particular, at that time the
Python numerical community had not yet reached consensus on the proper
API for array objects, or on what operators might be needed or useful
(e.g., PEP 225 proposes 6 new operators with underspecified
semantics).  Experience since then has eventually led to consensus
among the numerical community that the best solution is to add a
single infix operator for matrix multiply (together with any other new
operators this implies like ``@=``).

We review some of these alternatives here.

Use a type that defines ``__mul__`` as matrix multiplication:
    Numpy has had such a type for many years: ``np.matrix``.  And
    based on this experience, a strong consensus has developed that it
    should essentially never be used.  The problem is that the
    presence of two different duck-types for numeric data -- one where
    ``*`` means matrix multiply, and one where ``*`` means elementwise
    multiplication -- makes it impossible to write generic functions
    that can operate on arbitrary data.  In practice, the entire
    Python numeric ecosystem has standardized on using ``*`` for
    elementwise multiplication, and deprecated the use of
    ``np.matrix``.  Most 3rd-party libraries which receive a
    ``matrix`` as input will either error out, return incorrect
    results, or simply convert the input into a standard ``ndarray``,
    and return ``ndarray``s as well.  The only reason ``np.matrix``
    survives is because of strong arguments from some educators who
    find that its problems are outweighed by the need to provide a
    simple and clear mapping between mathematical notation and code
    for novices.

Add a new ``@`` (or whatever) operator that has some other meaning in
general Python, and then overload it in numeric code:
    This was the approach proposed by PEP 211, which suggested
    defining ``@`` to be the equivalent of ``itertools.product``. The
    problem with this is that when taken on its own terms, adding an
    infix operator for ``itertools.product`` is just silly.  Matrix
    multiplication has a uniquely strong rationale for inclusion as an
    infix operator.  There almost certainly don't exist any other
    binary operations that will ever justify adding another infix
    operator.

Add a ``.dot`` method to array types so as to allow "pseudo-infix"
A.dot(B) syntax:
    This has been in numpy for some years, and in many cases it's
    better than dot(A, B).  But it's still much less readable than
    real infix notation, and in particular still suffers from an
    extreme overabundance of parentheses.  See `Motivation`_ above.

Add lots of new operators / add a new generic syntax for defining
infix operators:
    In addition to this being generally un-Pythonic and repeatedly
    rejected by BDFL fiat, this would be using a sledgehammer to smash
    a fly.  There is a strong consensus in the scientific python
    community that matrix multiplication really is the only missing
    infix operator that matters enough to bother about. (In
    retrospect, we all think PEP 225 was a bad idea too.)

Use a language preprocessor that adds extra operators and perhaps
other syntax (as per recent BDFL suggestion [#preprocessor]):
    Aside from matrix multiplication, there are no other operators or
    syntax that anyone cares enough about to bother adding.  But
    defining a new language (presumably with its own parser which
    would have to be kept in sync with Python's, etc.), just to
    support a single binary operator, is neither practical nor
    desireable.  In the scientific context, Python's competition is
    special-purpose numerical languages (Matlab, R, IDL, etc.).
    Compared to these, Python's killer feature is exactly that one can
    mix specialized numerical code with general-purpose code for XML
    parsing, web page generation, database access, network
    programming, GUI libraries, etc., and we also gain major benefits
    from the huge variety of tutorials, reference material,
    introductory classes, etc., which use Python.  Fragmenting
    "numerical Python" from "real Python" would be a major source of
    confusion.  Having to set up a preprocessor would be an especially
    prohibitive complication for unsophisticated users.  And we use
    Python because we like Python!  We don't want
    almost-but-not-quite-Python.

Use overloading hacks to define a "new infix operator" like ``*dot*``,
as in a well-known Python recipe [#infix-hack]:
    Beautiful is better than ugly. This solution is so ugly that most
    developers will simply refuse to consider it for use in serious,
    reusable code.  This isn't just speculation -- a variant of this
    recipe is actually distributed as a supported part of a major
    Python mathematics system [#sage-infix], so it's widely available,
    yet still receives minimal use.  OTOH, the fact that people even
    consider such a 'solution', and are supporting it in shipping
    code, could be taken as further evidence for the need for a proper
    infix operator for matrix product.


References
==========

.. [#preprocessor] GvR comment attached to G+ post, apparently not directly linkable: https://plus.google.com/115212051037621986145/posts/hZVVtJ9bK3u
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

   In addition, the following tutorials could easily deal with
   matrices:

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

.. [#sloc-details] SLOCs are defined as physical lines which contain
   at least one token that is not a COMMENT, NEWLINE, ENCODING,
   INDENT, or DEDENT.  Counts were made by using ``tokenize`` module
   from Python 3.2.3 to examine the tokens in all files ending ``.py``
   underneath some directory.  Only tokens which occur at least once
   in the source trees are included in the table. Several distracting
   rows were trimmed by hand (e.g. ``.``, ``:``, ``...``).  The
   counting script will be available as an auxiliary file once this
   PEP is submitted; until then, it can be found here:
   https://gist.github.com/njsmith/9157645

   All counts were made using the latest development version of each
   project as of 21 Feb 2014.

   'stdlib' is the contents of the Lib/ directory in commit
   d6aa3fa646e2 to the cpython hg repository.

   'sklearn' is the contents of the sklearn/ directory in commit
   69b71623273ccfc1181ea83d8fb9e05ae96f57c7 to the scikits-learn
   repository: https://github.com/scikit-learn/scikit-learn

   'nipy' is the contents of the nipy/ directory in commit
   5419911e99546401b5a13bd8ccc3ad97f0d31037 to the nipy repository:
   https://github.com/nipy/nipy/

.. [#blas-fork]: BLAS libraries have a habit of secretly spawning
   threads, even when used from single-threaded programs.  And threads
   play very poorly with ``fork()``; the usual symptom is that
   attempting to perform linear algebra in a child process causes an
   immediate deadlock.
