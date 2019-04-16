=======================================
NEP 19 — Random Number Generator Policy
=======================================

:Author: Robert Kern <robert.kern@gmail.com>
:Status: Accepted
:Type: Standards Track
:Created: 2018-05-24
:Resolution: https://mail.python.org/pipermail/numpy-discussion/2018-June/078126.html

Abstract
--------

For the past decade, NumPy has had a strict backwards compatibility policy for
the number stream of all of its random number distributions.  Unlike other
numerical components in ``numpy``, which are usually allowed to return
different when results when they are modified if they remain correct, we have
obligated the random number distributions to always produce the exact same
numbers in every version.  The objective of our stream-compatibility guarantee
was to provide exact reproducibility for simulations across numpy versions in
order to promote reproducible research.  However, this policy has made it very
difficult to enhance any of the distributions with faster or more accurate
algorithms.  After a decade of experience and improvements in the surrounding
ecosystem of scientific software, we believe that there are now better ways to
achieve these objectives.  We propose relaxing our strict stream-compatibility
policy to remove the obstacles that are in the way of accepting contributions
to our random number generation capabilities.


The Status Quo
--------------

Our current policy, in full:

    A fixed seed and a fixed series of calls to ``RandomState`` methods using the
    same parameters will always produce the same results up to roundoff error
    except when the values were incorrect.  Incorrect values will be fixed and
    the NumPy version in which the fix was made will be noted in the relevant
    docstring.  Extension of existing parameter ranges and the addition of new
    parameters is allowed as long the previous behavior remains unchanged.

This policy was first instated in Nov 2008 (in essence; the full set of weasel
words grew over time) in response to a user wanting to be sure that the
simulations that formed the basis of their scientific publication could be
reproduced years later, exactly, with whatever version of ``numpy`` that was
current at the time.  We were keen to support reproducible research, and it was
still early in the life of ``numpy.random``.  We had not seen much cause to
change the distribution methods all that much.

We also had not thought very thoroughly about the limits of what we really
could promise (and by “we” in this section, we really mean Robert Kern, let’s
be honest).  Despite all of the weasel words, our policy overpromises
compatibility.  The same version of ``numpy`` built on different platforms, or
just in a different way could cause changes in the stream, with varying degrees
of rarity.  The biggest is that the ``.multivariate_normal()`` method relies on
``numpy.linalg`` functions.  Even on the same platform, if one links ``numpy``
with a different LAPACK, ``.multivariate_normal()`` may well return completely
different results.  More rarely, building on a different OS or CPU can cause
differences in the stream.  We use C ``long`` integers internally for integer
distribution (it seemed like a good idea at the time), and those can vary in
size depending on the platform.  Distribution methods can overflow their
internal C ``longs`` at different breakpoints depending on the platform and
cause all of the random variate draws that follow to be different.

And even if all of that is controlled, our policy still does not provide exact
guarantees across versions.  We still do apply bug fixes when correctness is at
stake.  And even if we didn’t do that, any nontrivial program does more than
just draw random numbers.  They do computations on those numbers, transform
those with numerical algorithms from the rest of ``numpy``, which is not
subject to so strict a policy.  Trying to maintain stream-compatibility for our
random number distributions does not help reproducible research for these
reasons.

The standard practice now for bit-for-bit reproducible research is to pin all
of the versions of code of your software stack, possibly down to the OS itself.
The landscape for accomplishing this is much easier today than it was in 2008.
We now have ``pip``.  We now have virtual machines.  Those who need to
reproduce simulations exactly now can (and ought to) do so by using the exact
same version of ``numpy``.  We do not need to maintain stream-compatibility
across ``numpy`` versions to help them.

Our stream-compatibility guarantee has hindered our ability to make
improvements to ``numpy.random``.  Several first-time contributors have
submitted PRs to improve the distributions, usually by implementing a faster,
or more accurate algorithm than the one that is currently there.
Unfortunately, most of them would have required breaking the stream to do so.
Blocked by our policy, and our inability to work around that policy, many of
those contributors simply walked away.


Implementation
--------------

Work on a proposed new PRNG subsystem is already underway in the randomgen_
project.  The specifics of the new design are out of scope for this NEP and up
for much discussion, but we will discuss general policies that will guide the
evolution of whatever code is adopted.  We will also outline just a few of the
requirements that such a new system must have to support the policy proposed in
this NEP.

First, we will maintain API source compatibility just as we do with the rest of
``numpy``.  If we *must* make a breaking change, we will only do so with an
appropriate deprecation period and warnings.

Second, breaking stream-compatibility in order to introduce new features or
improve performance will be *allowed* with *caution*.  Such changes will be
considered features, and as such will be no faster than the standard release
cadence of features (i.e. on ``X.Y`` releases, never ``X.Y.Z``).  Slowness will
not be considered a bug for this purpose.  Correctness bug fixes that break
stream-compatibility can happen on bugfix releases, per usual, but developers
should consider if they can wait until the next feature release.  We encourage
developers to strongly weight user’s pain from the break in
stream-compatibility against the improvements.  One example of a worthwhile
improvement would be to change algorithms for a significant increase in
performance, for example, moving from the `Box-Muller transform
<https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform>`_ method of
Gaussian variate generation to the faster `Ziggurat algorithm
<https://en.wikipedia.org/wiki/Ziggurat_algorithm>`_.  An example of a
discouraged improvement would be tweaking the Ziggurat tables just a little bit
for a small performance improvement.

Any new design for the RNG subsystem will provide a choice of different core
uniform PRNG algorithms.  A promising design choice is to make these core
uniform PRNGs their own lightweight objects with a minimal set of methods
(randomgen_ calls them “basic RNGs”).  The broader set of non-uniform
distributions will be its own class that holds a reference to one of these core
uniform PRNG objects and simply delegates to the core uniform PRNG object when
it needs uniform random numbers.  To borrow an example from randomgen_, the
class ``MT19937`` is a basic RNG that implements the classic Mersenne Twister
algorithm.  The class ``RandomGenerator`` wraps around the basic RNG to provide
all of the non-uniform distribution methods::

    # This is not the only way to instantiate this object.
    # This is just handy for demonstrating the delegation.
    >>> brng = MT19937(seed)
    >>> rg = RandomGenerator(brng)
    >>> x = rg.standard_normal(10)

We will be more strict about a select subset of methods on these basic RNG
objects.  They MUST guarantee stream-compatibility for a specified set
of methods which are chosen to make it easier to compose them to build other
distributions and which are needed to abstract over the implementation details
of the variety of core PRNG algorithms.  Namely,

    * ``.bytes()``
    * ``.random_uintegers()``
    * ``.random_sample()``

The distributions class (``RandomGenerator``) SHOULD have all of the same
distribution methods as ``RandomState`` with close-enough function signatures
such that almost all code that currently works with ``RandomState`` instances
will work with ``RandomGenerator`` instances (ignoring the precise stream
values).  Some variance will be allowed for integer distributions: in order to
avoid some of the cross-platform problems described above, these SHOULD be
rewritten to work with ``uint64`` numbers on all platforms.

.. _randomgen: https://github.com/bashtage/randomgen


Supporting Unit Tests
:::::::::::::::::::::

Because we did make a strong stream-compatibility guarantee early in numpy’s
life, reliance on stream-compatibility has grown beyond reproducible
simulations.  One use case that remains for stream-compatibility across numpy
versions is to use pseudorandom streams to generate test data in unit tests.
With care, many of the cross-platform instabilities can be avoided in the
context of small unit tests.

The new PRNG subsystem MUST provide a second, legacy distributions class that
uses the same implementations of the distribution methods as the current
version of ``numpy.random.RandomState``.  The methods of this class will have
strict stream-compatibility guarantees, even stricter than the current policy.
It is intended that this class will no longer be modified, except to keep it
working when numpy internals change.  All new development should go into the
primary distributions class.  Bug fixes that change the stream SHALL NOT be
made to ``RandomState``; instead, buggy distributions should be made to warn
when they are buggy.  The purpose of ``RandomState`` will be documented as
providing certain fixed functionality for backwards compatibility and stable
numbers for the limited purpose of unit testing, and not making whole programs
reproducible across numpy versions.

This legacy distributions class MUST be accessible under the name
``numpy.random.RandomState`` for backwards compatibility.  All current ways of
instantiating ``numpy.random.RandomState`` with a given state should
instantiate the Mersenne Twister basic RNG with the same state.  The legacy
distributions class MUST be capable of accepting other basic RNGs.  The purpose
here is to ensure that one can write a program with a consistent basic RNG
state with a mixture of libraries that may or may not have upgraded from
``RandomState``.  Instances of the legacy distributions class MUST respond
``True`` to ``isinstance(rg, numpy.random.RandomState)`` because there is
current utility code that relies on that check.  Similarly, old pickles of
``numpy.random.RandomState`` instances MUST unpickle correctly.


``numpy.random.*``
::::::::::::::::::

The preferred best practice for getting reproducible pseudorandom numbers is to
instantiate a generator object with a seed and pass it around.  The implicit
global ``RandomState`` behind the ``numpy.random.*`` convenience functions can
cause problems, especially when threads or other forms of concurrency are
involved.  Global state is always problematic.  We categorically recommend
avoiding using the convenience functions when reproducibility is involved.

That said, people do use them and use ``numpy.random.seed()`` to control the
state underneath them.  It can be hard to categorize and count API usages
consistently and usefully, but a very common usage is in unit tests where many
of the problems of global state are less likely.

This NEP does not propose removing these functions or changing them to use the
less-stable ``RandomGenerator`` distribution implementations.  Future NEPs
might.

Specifically, the initial release of the new PRNG subsystem SHALL leave these
convenience functions as aliases to the methods on a global ``RandomState``
that is initialized with a Mersenne Twister basic RNG object.  A call to
``numpy.random.seed()`` will be forwarded to that basic RNG object.  In
addition, the global ``RandomState`` instance MUST be accessible in this
initial release by the name ``numpy.random.mtrand._rand``: Robert Kern long ago
promised ``scikit-learn`` that this name would be stable.  Whoops.

In order to allow certain workarounds, it MUST be possible to replace the basic
RNG underneath the global ``RandomState`` with any other basic RNG object (we
leave the precise API details up to the new subsystem).  Calling
``numpy.random.seed()`` thereafter SHOULD just pass the given seed to the
current basic RNG object and not attempt to reset the basic RNG to the Mersenne
Twister.  The set of ``numpy.random.*`` convenience functions SHALL remain the
same as they currently are.  They SHALL be aliases to the ``RandomState``
methods and not the new less-stable distributions class (``RandomGenerator``,
in the examples above). Users who want to get the fastest, best distributions
can follow best practices and instantiate generator objects explicitly.

This NEP does not propose that these requirements remain in perpetuity.  After
we have experience with the new PRNG subsystem, we can and should revisit these
issues in future NEPs.


Alternatives
------------

Versioning
::::::::::

For a long time, we considered that the way to allow algorithmic improvements
while maintaining the stream was to apply some form of versioning.  That is,
every time we make a stream change in one of the distributions, we increment
some version number somewhere.  ``numpy.random`` would keep all past versions
of the code, and there would be a way to get the old versions.

We will not be doing this.  If one needs to get the exact bit-for-bit results
from a given version of ``numpy``, whether one uses random numbers or not, one
should use the exact version of ``numpy``.

Proposals of how to do RNG versioning varied widely, and we will not
exhaustively list them here.  We spent years going back and forth on these
designs and were not able to find one that sufficed.  Let that time lost, and
more importantly, the contributors that we lost while we dithered, serve as
evidence against the notion.

Concretely, adding in versioning makes maintenance of ``numpy.random``
difficult.  Necessarily, we would be keeping lots of versions of the same code
around.  Adding a new algorithm safely would still be quite hard.

But most importantly, versioning is fundamentally difficult to *use* correctly.
We want to make it easy and straightforward to get the latest, fastest, best
versions of the distribution algorithms; otherwise, what's the point?  The way
to make that easy is to make the latest the default.  But the default will
necessarily change from release to release, so the user’s code would need to be
altered anyway to specify the specific version that one wants to replicate.

Adding in versioning to maintain stream-compatibility would still only provide
the same level of stream-compatibility that we currently do, with all of the
limitations described earlier.  Given that the standard practice for such needs
is to pin the release of ``numpy`` as a whole, versioning ``RandomState`` alone
is superfluous.


``StableRandom``
::::::::::::::::

A previous version of this NEP proposed to leave ``RandomState`` completely
alone for a deprecation period and build the new subsystem alongside with new
names.  To satisfy the unit testing use case, it proposed introducing a small
distributions class nominally called ``StableRandom``. It would have provided
a small subset of distribution methods that were considered most useful in unit
testing, but not the full set such that it would be too likely to be used
outside of the testing context.

During discussion about this proposal, it became apparent that there was no
satisfactory subset.  At least some projects used a fairly broad selection of
the ``RandomState`` methods in unit tests.

Downstream project owners would have been forced to modify their code to
accomodate the new PRNG subsystem.  Some modifications might be simply
mechanical, but the bulk of the work would have been tedious churn for no
positive improvement to the downstream project, just avoiding being broken.

Furthermore, under this old proposal, we would have had a quite lengthy
deprecation period where ``RandomState`` existed alongside the new system of
basic RNGs and distribution classes. Leaving the implementation of
``RandomState`` fixed meant that it could not use the new basic RNG state
objects.  Developing programs that use a mixture of libraries that have and
have not upgraded would require managing two sets of PRNG states.  This would
notionally have been time-limited, but we intended the deprecation to be very
long.

The current proposal solves all of these problems.  All current usages of
``RandomState`` will continue to work in perpetuity, though some may be
discouraged through documentation.  Unit tests can continue to use the full
complement of ``RandomState`` methods.  Mixed ``RandomState/RandomGenerator``
code can safely share the common basic RNG state.  Unmodified ``RandomState``
code can make use of the new features of alternative basic RNGs like settable
streams.


Discussion
----------

- `NEP discussion <https://mail.python.org/pipermail/numpy-discussion/2018-June/078126.html>`_
- `Earlier discussion <https://mail.python.org/pipermail/numpy-discussion/2018-January/077608.html>`_


Copyright
---------

This document has been placed in the public domain.
