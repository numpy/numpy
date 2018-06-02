==============================
Random Number Generator Policy
==============================

:Author: Robert Kern <robert.kern@gmail.com>
:Status: Draft
:Type: Standards Track
:Created: 2018-05-24


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

We propose first freezing ``RandomState`` as it is and developing a new RNG
subsystem alongside it.  This allows anyone who has been relying on our old
stream-compatibility guarantee to have plenty of time to migrate.
``RandomState`` will be considered deprecated, but with a long deprecation
cycle, at least a few years.  Deprecation warnings will start silent but become
increasingly noisy over time.  Bugs in the current state of the code will *not*
be fixed if fixing them would impact the stream.  However, if changes in the
rest of ``numpy`` would break something in the ``RandomState`` code, we will
fix ``RandomState`` to continue working (for example, some change in the
C API).  No new features will be added to ``RandomState``.  Users should
migrate to the new subsystem as they are able to.

Work on a proposed `new PRNG subsystem
<https://github.com/bashtage/randomgen>`_ is already underway.  The specifics
of the new design are out of scope for this NEP and up for much discussion, but
we will discuss general policies that will guide the evolution of whatever code
is adopted.

First, we will maintain API source compatibility just as we do with the rest of
``numpy``.  If we *must* make a breaking change, we will only do so with an
appropriate deprecation period and warnings.

Second, breaking stream-compatibility in order to introduce new features or
improve performance will be *allowed* with *caution*.  Such changes will be
considered features, and as such will be no faster than the standard release
cadence of features (i.e. on ``X.Y`` releases, never ``X.Y.Z``).  Slowness is
not a bug.  Correctness bug fixes that break stream-compatibility can happen on
bugfix releases, per usual, but developers should consider if they can wait
until the next feature release.  We encourage developers to strongly weight
user’s pain from the break in stream-compatibility against the improvements.
One example of a worthwhile improvement would be to change algorithms for
a significant increase in performance, for example, moving from the `Box-Muller
transform <https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform>`_ method
of Gaussian variate generation to the faster `Ziggurat algorithm
<https://en.wikipedia.org/wiki/Ziggurat_algorithm>`_.  An example of an
unworthy improvement would be tweaking the Ziggurat tables just a little bit.

Any new design for the RNG subsystem will provide a choice of different core
uniform PRNG algorithms.  We will be more strict about a select subset of
methods on these core PRNG objects.  They MUST guarantee stream-compatibility
for a minimal, specified set of methods which are chosen to make it easier to
compose them to build other distributions.  Namely,

    * ``.bytes()``
    * ``.random_uintegers()``
    * ``.random_sample()``

Furthermore, the new design should also provide one generator class (we shall
call it ``StableRandom`` for discussion purposes) that provides a slightly
broader subset of distribution methods for which stream-compatibility is
*guaranteed*.  The point of ``StableRandom`` is to provide something that can
be used in unit tests so projects that currently have tests which rely on the
precise stream can be migrated off of ``RandomState``.  For the best
transition, ``StableRandom`` should use as its core uniform PRNG the current
MT19937 algorithm.  As best as possible, the API for the distribution methods
that are provided on ``StableRandom`` should match their counterparts on
``RandomState``.  They should provide the same stream that the current version
of ``RandomState`` does.  Because their intended use is for unit tests, we do
not need the performance improvements from the new algorithms that will be
introduced by the new subsystem.

The list of ``StableRandom`` methods should be chosen to support unit tests:

    * ``.randint()``
    * ``.uniform()``
    * ``.normal()``
    * ``.standard_normal()``
    * ``.choice()``
    * ``.shuffle()``
    * ``.permutation()``


Not Versioning
--------------

For a long time, we considered that the way to allow algorithmic improvements
while maintaining the stream was to apply some form of versioning.  That is,
every time we make a stream change in one of the distributions, we increment
some version number somewhere.  ``numpy.random`` would keep all past versions
of the code, and there would be a way to get the old versions.  Proposals of
how to do this exactly varied widely, but we will not exhaustively list them
here.  We spent years going back and forth on these designs and were not able
to find one that sufficed.  Let that time lost, and more importantly, the
contributors that we lost while we dithered, serve as evidence against the
notion.

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


Discussion
----------

- https://mail.python.org/pipermail/numpy-discussion/2018-January/077608.html
- https://github.com/numpy/numpy/pull/10124#issuecomment-350876221


Copyright
---------

This document has been placed in the public domain.
