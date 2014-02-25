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

Matrix multiplication is uniquely deserving of a new, dedicated infix
operator:

* Adding an infix matrix multiplication operator brings Python into
  alignment with universal notational practice across all fields of
  mathematics, science, and engineering.

* ``@`` greatly clarifies real-world code.

* ``@`` provides a smoother onramp for less experienced users.

* ``@`` benefits a large and growing user community.

* ``@`` will be used frequently -- quite possibly more frequently than
  ``//`` or the bitwise operators.

* ``@`` helps this community finally standardize on a single duck type
  for all matrix-like objects.

And, given the existence of ``@``, it makes more sense than not to
have ``@@``, ``@=``, and ``@@=``, so they are added as well.


Why should matrix multiplication be infix?
------------------------------------------

When moving from scalars -- like ordinary Python floats -- to more
general n-dimensional arrays and matrices, there are two standard ways
to generalize the usual multiplication operation.  One is elementwise
multiplication::

  [2, 3] * [4, 5] = [2 * 4, 3 * 5] = [8, 15]

and the other is the `matrix product`_.  For various reasons, the
numerical Python ecosystem has settled on the convention that ``*``
refers to elementwise multiplication.  However, this leaves us with no
convenient notation for matrix multiplication.

.. _matrix product: https://en.wikipedia.org/wiki/Matrix_multiplication

Matrix multiplication is similar to ordinary arithmetic operations
like addition and multiplication on scalars in two ways: (a) it is
used very heavily in numerical programs -- often multiple times per
line of code -- and (b) it has an ancient and universally adopted
tradition of being written using infix syntax with varying precedence.
This is because, for typical formulas, this notation is dramatically
more readable than any function syntax.

Here's a concrete example.  One of the most useful tools for testing a
statistical hypothesis is the linear hypothesis test for OLS
regression models.  If we want to implement this, we will look up some
textbook or paper on it, and encounter many mathematical formulas that
look like:

.. math::

    S = (H \beta - r)^T (H V H^T)^{-1} (H \beta - r)

Here the various variables are all vectors or matrices (details for
the curious: [#lht]).

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

Notice that there is now a transparent, 1-to-1 mapping between symbols
in the original formula and the code.

Of course, a more sophisticated programmer will probably notice that
this is not the best way to compute this expression.  The repeated
computation of :math:`H \beta - r` should perhaps be factored out;
and, expressions of the form ``dot(inv(A), B)`` should almost always
be replaced by the more numerically stable ``solve(A, B)``.  When
using ``@``, performing these refactorings gives us::

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
errors while trying to write out the 'dot' formulas above.  They still
contain at least one error.  (Exercise: find it, or them.)  In
comparison, the ``@`` examples are not only correct, they're obviously
correct at a glance.


Simple syntax is especially critical for non-expert programmers
---------------------------------------------------------------

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
causes.  Adding ``@`` will benefit both beginning and advanced users;
and furthermore, it will allow both groups to standardize on the same
notation from the start, providing a smoother on-ramp to expertise.


But isn't matrix multiplication a pretty niche requirement?
-----------------------------------------------------------

The world is full of continuous data, and computers are increasingly
called upon to work with it in sophisticated ways.  Matrices are the
lingua franca of finance, machine learning, 3d graphics, computer
vision, robotics, operations research, econometrics, meteorology,
computational linguistics, recommendation systems, neuroscience,
bioinformatics (including genetics, cancer research, drug discovery,
etc.), physics engines, quantum mechanics, network analysis, and many
other application areas.

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
and also in two high-profile numerical packages -- the scikit-learn
machine learning library, and the nipy neuroimaging library --
normalized by source lines of code (SLOC).  Rows are sorted by the
'combined' column, which pools all three code bases together.  The
combined column is thus strongly weighted towards the stdlib, which is
much larger than both projects put together (stdlib: 411575 SLOC,
scikit-learn: 50924 SLOC, nipy: 37078 SLOC). [#sloc-details]

The **dot** row (marked ``******``) counts how common matrix multiply
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

These numerical packages together contain ~780 uses of matrix
multiplication.  Within these packages, matrix multiplication is used
more heavily than most comparison operators (``<`` ``!=`` ``<=``
``>=``).  When we include the stdlib into our comparisons, matrix
multiplication is still used more often in total than any of the
bitwise operators, and 2x as often as ``//``.  This is true even
though the stdlib, which contains a fair amount of integer arithmetic
and no matrix operations, makes up more than 80% of the combined code
base.  (In an interesting coincidence, the numeric libraries make up
approximately the same proportion of the 'combined' codebase as
numeric tutorials make up of PyCon 2014's tutorial schedule.)

While it's impossible to know for certain, from this data it seems
plausible that on net across all Python code currently being written,
matrix multiplication is used more often than ``//`` or other integer
operations.


But isn't it weird to add an operator with no stdlib uses?
----------------------------------------------------------

It's certainly unusual (though ``...`` was also added without any
stdlib uses), but the important thing is whether a change will benefit
users, not where the software is being downloaded from.  It's clear
from the above that ``@`` will be used, and used heavily.  And -- who
knows? -- perhaps someday the stdlib will contain a matrix type of
some sort.  This PEP only moves us closer to that possibility, by
helping the Python numerical community finally standardize on a single
duck type for all matrix-like objects.


Matrix power and in-place operators
-----------------------------------

The primary motivation for this PEP is ``@``; no-one cares terribly
much about the other proposed operators.  The matrix power operator
``@@`` is useful and well-defined, but not really necessary.  It is
included here for consistency: if we have an ``@`` that is analogous
to ``*``, then it would be weird and surprising to *not* have an
``@@`` that is analogous to ``**``.  Similarly, the in-place operators
``@=`` and ``@@=`` are of marginal utility -- it is not generally
possible to implement in-place matrix multiplication any more
efficiently than by doing ``a = (a @ b)`` -- but are included for
completeness and symmetry.


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


Intended usage details
======================

This section is informative, rather than normative -- it documents the
consensus of a number of libraries that provide array- or matrix-like
objects on how the ``@`` and ``@@`` operators will be implemented.
Not all matrix-like data types will provide all of the different
dimensionalities described here; in particular, many will implement
only the 2d or 1d+2d subsets.  But ideally whatever functionality is
available will be consistent with this.

This section uses the numpy terminology for describing arbitrary
multidimensional arrays of data.  In this model, the shape of any
array is represented by a tuple of integers.  Matrices have len(shape)
== 2, 1d vectors have len(shape) == 1, and scalars have shape == (),
i.e., they are "0 dimensional".  Any array contains prod(shape) total
entries.  Notice that prod(()) == 1 (for the same reason that sum(())
== 0); scalars are just an ordinary kind of array, not anything
special.  Notice also that we distinguish between a single scalar
value (shape == (), analogous to `1`), a vector containing only a
single entry (shape == (1,), analogous to `[1]`), a matrix containing
only a single entry (shape == (1, 1), analogous to `[[1]]`), etc., so
the dimensionality of any array is always well-defined.

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

* 2d inputs are conventional matrices, and treated in the obvious
  way.

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
  [#broadcasting] for details.

  If one operand is >2d, and another operand is 1d, then the above
  rules apply unchanged, with 1d->2d promotion performed before
  broadcasting.  E.g., ``arr(10, 2, 3) @ arr(3)`` first promotes to
  ``arr(10, 2, 3) @ arr(3, 1)``, then broadcasts and multiplies to get
  an array with shape (10, 2, 1), and finally removes the added
  dimension, returning an array with shape (10, 2).  Similarly,
  ``arr(2) @ arr(10, 2, 3)`` produces an intermediate array with shape
  (10, 1, 3), and a final array with shape (10, 3).

The recommended semantics for ``@@`` are::

    def __matpow__(self, n):
        if not isinstance(n, numbers.Integral):
            raise TypeError("n must be integer")
        if n == 0:
            return identity_matrix_with_shape(self.shape)
        elif n < 0:
            return inverse(self) @ (self @@ (n + 1))
        else:
            return self @ (self @@ (n - 1))

(Of course we expect that much more efficient implementations will be
used in practice.)

The following projects have expressed an intention to implement ``@``
and ``@@`` on their matrix-like types in a manner consistent with the
above definitions:

* numpy

* scipy.sparse

* pandas

* blaze

* XX (try: Theano, OpenCV, cvxopt, pycuda, sage, sympy, pysparse,
  pyoperators, any others?  QTransform in PyQt? PyOpenGL doesn't seem
  to provide a matrix type. panda3d?)


Rationale
=========

Alternative ways to go about adding a matrix multiplication operator
--------------------------------------------------------------------

Choice of operator
''''''''''''''''''

Why ``@`` instead of some other punctuation symbol? It doesn't matter
much, and there isn't any consensus across other programming languages
about how this operator should be named [#matmul-other-langs], but
``@`` has a few advantages:

* ``@`` is a friendly character that Pythoneers are already used to
  typing in decorators, and its use in email addresses means it is
  more likely to be easily accessible across keyboard layouts than
  some other characters (e.g. $).

* The mATrices mnemonic is cute.

* It's round like ``*`` and :math:`\cdot`.

* The swirly shape is reminiscent of the simultaneous sweeps over rows
  and columns that define matrix multiplication.


Definitions for built-ins
'''''''''''''''''''''''''

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
use of ``multiprocessing`` [#blas-fork].  Thus we'll continue to
delegate dealing with these problems to numpy and friends, at least
for now.

There are also non-numeric Python builtins which define ``__mul__``
(``str``, ``list``, ...).  We do not define ``__matmul__`` for these
types either, because why would we even do that.


Alternatives to adding a new operator at all
--------------------------------------------

Over the past 15+ years, the Python numeric community has explored a
variety of ways to handle the tension between matrix and elementwise
multiplication operations.  PEP 211 and PEP 225, both proposed in 2000
and last seriously discussed in 2008 [#threads-2008], were early
attempts to add new operators to solve this problem, but suffered from
serious flaws; in particular, at that time the Python numerical
community had not yet reached consensus on the proper API for array
objects, or on what operators might be needed or useful (e.g., PEP 225
proposes 6 new operators with underspecified semantics).  Experience
since then has eventually led to consensus among the numerical
community that the best solution is to add a single infix operator for
matrix multiply (together with any other new operators this implies
like ``@=``).

We review some of the rejected alternatives here.

**Use a type that defines ``__mul__`` as matrix multiplication:**
Numpy has had such a type for many years: ``np.matrix`` (as opposed to
the standard array type, ``np.ndarray``).  And based on this
experience, a strong consensus has developed that ``np.matrix`` should
essentially never be used.  The problem is that the presence of two
different duck-types for numeric data -- one where ``*`` means matrix
multiply, and one where ``*`` means elementwise multiplication -- make
it impossible to write generic functions that can operate on arbitrary
data.  In practice, the vast majority of the Python numeric ecosystem
has standardized on using ``*`` for elementwise multiplication, and
deprecated the use of ``np.matrix``.  Most 3rd-party libraries that
receive a ``matrix`` as input will either error out, return incorrect
results, or simply convert the input into a standard ``ndarray``, and
return ``ndarray`` objects as well.  The only reason ``np.matrix``
survives is because of strong arguments from some educators who find
that its problems are outweighed by the need to provide a simple and
clear mapping between mathematical notation and code for novices; and
this, as described above, causes its own problems.

**Add a new ``@`` (or whatever) operator that has some other meaning
in general Python, and then overload it in numeric code:** This was
the approach proposed by PEP 211, which suggested defining ``@`` to be
the equivalent of ``itertools.product``.  The problem with this is
that when taken on its own terms, adding an infix operator for
``itertools.product`` is just silly.  Matrix multiplication has a
uniquely strong rationale for inclusion as an infix operator.  There
almost certainly don't exist any other binary operations that will
ever justify adding another infix operator.

**Add a ``.dot`` method to array types so as to allow "pseudo-infix"
A.dot(B) syntax:** This has been in numpy for some years, and in many
cases it's better than dot(A, B).  But it's still much less readable
than real infix notation, and in particular still suffers from an
extreme overabundance of parentheses.  See `Motivation`_ above.

**Add lots of new operators / add a new generic syntax for defining
infix operators:** In addition to this being generally un-Pythonic and
repeatedly rejected by BDFL fiat, this would be using a sledgehammer
to smash a fly.  There is a consensus in the scientific python
community that matrix multiplication really is the only missing infix
operator that matters enough to bother about. (In retrospect, we all
think PEP 225 was a bad idea too.)

**Use a language preprocessor that adds extra operators and perhaps
other syntax (as per recent BDFL suggestion [#preprocessor]):** Aside
from matrix multiplication, there are no other operators or syntax
that anyone cares enough about to bother adding.  But defining a new
language (presumably with its own parser which would have to be kept
in sync with Python's, etc.), just to support a single binary
operator, is neither practical nor desireable.  In the scientific
context, Python's competition is special-purpose numerical languages
(Matlab, R, IDL, etc.).  Compared to these, Python's killer feature is
exactly that one can mix specialized numerical code with
general-purpose code for XML parsing, web page generation, database
access, network programming, GUI libraries, etc., and we also gain
major benefits from the huge variety of tutorials, reference material,
introductory classes, etc., which use Python.  Fragmenting "numerical
Python" from "real Python" would be a major source of confusion.
Having to set up a preprocessor would be an especially prohibitive
complication for unsophisticated users.  And we use Python because we
like Python!  We don't want almost-but-not-quite-Python.

**Use overloading hacks to define a "new infix operator" like
``*dot*``, as in a well-known Python recipe [#infix-hack]:** Beautiful
is better than ugly. This solution is so ugly that most developers
will simply refuse to consider it for use in serious, reusable code.
This isn't just speculation -- a variant of this recipe is actually
distributed as a supported part of a major Python mathematics system
[#sage-infix], so it's widely available, yet still receives minimal
use.  OTOH, the fact that people even consider such a 'solution', and
are supporting it in shipping code, could be taken as further evidence
for the need for a proper infix operator for matrix product.


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

.. [#blas-fork]: BLAS libraries have a habit of secretly spawning
   threads, even when used from single-threaded programs.  And threads
   play very poorly with ``fork()``; the usual symptom is that
   attempting to perform linear algebra in a child process causes an
   immediate deadlock.

.. [#threads-2008]: http://fperez.org/py4science/numpy-pep225/numpy-pep225.html

.. [#broadcasting]: http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

.. [#matmul-other-langs]: http://mail.scipy.org/pipermail/scipy-user/2014-February/035499.html
