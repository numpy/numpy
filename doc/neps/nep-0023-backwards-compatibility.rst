.. _NEP23:

=======================================================
NEP 23 — Backwards compatibility and deprecation policy
=======================================================

:Author: Ralf Gommers <ralf.gommers@gmail.com>
:Status: Draft
:Type: Process
:Created: 2018-07-14
:Resolution: <url> (required for Accepted | Rejected | Withdrawn)

Abstract
--------

In this NEP we describe NumPy's approach to backwards compatibility,
its deprecation and removal policy, and the trade-offs and decision
processes for individual cases where breaking backwards compatibility
is considered.


Detailed description
--------------------

NumPy has a very large user base.  Those users rely on NumPy being stable
and the code they write that uses NumPy functionality to keep working.
NumPy is also actively maintained and improved -- and sometimes improvements
require, or are made much easier, by breaking backwards compatibility.
Finally, there are trade-offs in stability for existing users vs. avoiding
errors or having a better user experience for new users.  These competing
needs often give rise to heated debates and delays in accepting or rejecting
contributions.  This NEP tries to address that by providing a policy as well
as examples and rationales for when it is or isn't a good idea to break
backwards compatibility.

General principles:

- Aim not to break users' code unnecessarily.
- Aim never to change code in ways that can result in users silently getting
  incorrect results from their previously working code.
- Backwards incompatible changes can be made, provided the benefits outweigh
  the costs.
- When assessing the costs, keep in mind that most users do not read the mailing
  list, do not look at deprecation warnings, and sometimes wait more than one or
  two years before upgrading from their old version.  And that NumPy has
  millions of users, so "no one will do or use this" is very likely incorrect.
- Benefits include improved functionality, usability and performance,
  as well as lower maintenance cost and improved future extensibility.
- Bug fixes are exempt from the backwards compatibility policy.  However in case
  of serious impact on users (e.g. a downstream library doesn't build anymore),
  even bug fixes may have to be delayed for one or more releases.
- The Python API and the C API will be treated in the same way.


Examples
^^^^^^^^

We now discuss a number of concrete examples to illustrate typical issues
and trade-offs.

**Changing the behavior of a function**

``np.histogram`` is probably the most infamous example.
First, a new keyword ``new=False`` was introduced, this was then switched
over to None one release later, and finally it was removed again.
Also, it has a ``normed`` keyword that had behavior that could be considered
either suboptimal or broken (depending on ones opinion on the statistics).
A new keyword ``density`` was introduced to replace it; ``normed`` started giving
``DeprecationWarning`` only in v.1.15.0.  Evolution of ``histogram``::

    def histogram(a, bins=10, range=None, normed=False):  # v1.0.0

    def histogram(a, bins=10, range=None, normed=False, weights=None, new=False):  #v1.1.0

    def histogram(a, bins=10, range=None, normed=False, weights=None, new=None):  #v1.2.0

    def histogram(a, bins=10, range=None, normed=False, weights=None):  #v1.5.0

    def histogram(a, bins=10, range=None, normed=False, weights=None, density=None):  #v1.6.0

    def histogram(a, bins=10, range=None, normed=None, weights=None, density=None):  #v1.15.0
        # v1.15.0 was the first release where `normed` started emitting
        # DeprecationWarnings

The ``new`` keyword was planned from the start to be temporary.  Such a plan
forces users to change their code more than once, which is almost never the
right thing to do.  Instead, a better approach here would have been to
deprecate ``histogram`` and introduce a new function ``hist`` in its place.


**Disallowing indexing with floats**

Indexing an array with floats is asking for something ambiguous, and can be a
sign of a bug in user code.  After some discussion, it was deemed a good idea
to deprecate indexing with floats.  This was first tried for the v1.8.0
release, however in pre-release testing it became clear that this would break
many libraries that depend on NumPy.  Therefore it was reverted before release,
to give those libraries time to fix their code first.  It was finally
introduced for v1.11.0 and turned into a hard error for v1.12.0.

This change was disruptive, however it did catch real bugs in, e.g., SciPy and
scikit-learn.  Overall the change was worth the cost, and introducing it in
master first to allow testing, then removing it again before a release, is a
useful strategy.

Similar deprecations that also look like good examples of
cleanups/improvements:

- removing deprecated boolean indexing (in 2016, see `gh-8312 <https://github.com/numpy/numpy/pull/8312>`__)
- deprecating truth testing on empty arrays (in 2017, see `gh-9718 <https://github.com/numpy/numpy/pull/9718>`__)


**Removing the financial functions**

The financial functions (e.g. ``np.pmt``) had short non-descriptive names, were
present in the main NumPy namespace, and didn't really fit well within NumPy's
scope.  They were added in 2008 after
`a discussion <https://mail.python.org/pipermail/numpy-discussion/2008-April/032353.html>`_
on the mailing list where opinion was divided (but a majority in favor).
The financial functions didn't cause a lot of overhead, however there were
still multiple issues and PRs a year for them which cost maintainer time to
deal with.  And they cluttered up the ``numpy`` namespace.  Discussion on
removing them happened in 2013 (gh-2880, rejected) and then again in 2019
(:ref:`NEP32`, accepted without significant complaints).

Given that they were clearly outside of NumPy's scope, moving them to a
separate ``numpy-financial`` package and removing them from NumPy after a
deprecation period made sense.


Policy
------

1. Code changes that have the potential to silently change the results of a users'
   code must never be made (except in the case of clear bugs).
2. Code changes that break users' code (i.e. the user will see a clear exception)
   can be made, *provided the benefit is worth the cost* and suitable deprecation
   warnings have been raised first.
3. Deprecation warnings are in all cases warnings that functionality will be removed.
   If there is no intent to remove functionality, then deprecation in documentation
   only or other types of warnings shall be used.
4. Deprecations for stylistic reasons (e.g. consistency between functions) are
   strongly discouraged.

Deprecations:

- shall include the version numbers of both when the functionality was deprecated
  and when it will be removed (either two releases after the warning is
  introduced, or in the next major version).
- shall include information on alternatives to the deprecated functionality, or a
  reason for the deprecation if no clear alternative is available.
- shall use ``VisibleDeprecationWarning`` rather than ``DeprecationWarning``
  for cases of relevance to end users (as opposed to cases only relevant to
  libraries building on top of NumPy).
- shall be listed in the release notes of the release where the deprecation happened.

Removal of deprecated functionality:

- shall be done after 2 releases (assuming a 6-monthly release cycle; if that changes,
  there shall be at least 1 year between deprecation and removal), unless the
  impact of the removal is such that a major version number increase is
  warranted.
- shall be listed in the release notes of the release where the removal happened.
- can be done in any minor (but not bugfix) release.

In concrete cases where this policy needs to be applied, decisions are made according
to the `NumPy governance model
<https://docs.scipy.org/doc/numpy/dev/governance/index.html>`_.

Functionality with more strict policies:

- ``numpy.random`` has its own backwards compatibility policy,
  see `NEP 19 <http://www.numpy.org/neps/nep-0019-rng-policy.html>`_.
- The file format for ``.npy`` and ``.npz`` files must not be changed in a backwards
  incompatible way.


Alternatives
------------

**Being more aggressive with deprecations.**

The goal of being more aggressive is to allow NumPy to move forward faster.
This would avoid others inventing their own solutions (often in multiple
places), as well as be a benefit to users without a legacy code base.  We
reject this alternative because of the place NumPy has in the scientific Python
ecosystem - being fairly conservative is required in order to not increase the
extra maintenance for downstream libraries and end users to an unacceptable
level.


Discussion
----------

- `Mailing list discussion on the first version of this NEP in 2018 <https://mail.python.org/pipermail/numpy-discussion/2018-July/078432.html>`__


References and Footnotes
------------------------

- `Issue requesting semantic versioning <https://github.com/numpy/numpy/issues/10156>`__


Copyright
---------

This document has been placed in the public domain.
