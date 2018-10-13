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
  many hundreds of thousands or even a couple of million users, so "no one will
  do or use this" is very likely incorrect.
- Benefits include improved functionality, usability and performance (in order
  of importance), as well as lower maintenance cost and improved future
  extensibility.
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

**Returning a view rather than a copy**

The ``ndarray.diag`` method used to return a copy.  A view would be better for
both performance and design consistency.  This change was warned about
(``FutureWarning``) in v.8.0, and in v1.9.0 ``diag`` was changed to return
a *read-only* view.  The planned change to a writeable view in v1.10.0 was
postponed due to backwards compatibility concerns, and is still an open issue
(gh-7661).

What should have happened instead: nothing.  This change resulted in a lot of
discussions and wasted effort, did not achieve its final goal, and was not that
important in the first place.  Finishing the change to a *writeable* view in
the future is not desired, because it will result in users silently getting
different results if they upgraded multiple versions or simply missed the
warnings.

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

Similar recent deprecations also look like good examples of
cleanups/improvements:

- removing deprecated boolean indexing (gh-8312)
- deprecating truth testing on empty arrays (gh-9718)
- deprecating ``np.sum(generator)`` (gh-10670, one issue with this one is that
  its warning message is wrong - this should error in the future).

**Removing the financial functions**

The financial functions (e.g. ``np.pmt``) are badly named, are present in the
main NumPy namespace, and don't really fit well within NumPy's scope.
They were added in 2008 after
`a discussion <https://mail.python.org/pipermail/numpy-discussion/2008-April/032353.html>`_
on the mailing list where opinion was divided (but a majority in favor).
At the moment these functions don't cause a lot of overhead, however there are
multiple issues and PRs a year for them which cost maintainer time to deal
with.  And they clutter up the ``numpy`` namespace.  Discussion in 2013 happened
on removing them again (gh-2880).

This case is borderline, but given that they're clearly out of scope,
deprecation and removal out of at least the main ``numpy`` namespace can be
proposed.  Alternatively, document clearly that new features for financial
functions are unwanted, to keep the maintenance costs to a minimum.

**Examples of features not added because of backwards compatibility**

TODO: do we have good examples here? Possibly subclassing related?


Removing complete submodules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This year there have been suggestions to consider removing some or all of
``numpy.distutils``, ``numpy.f2py``, ``numpy.linalg``, and ``numpy.random``.
The motivation was that all these cost maintenance effort, and that they slow
down work on the core of Numpy (ndarrays, dtypes and ufuncs).

The impact on downstream libraries and users would be very large, and
maintenance of these modules would still have to happen.  Therefore this is
simply not a good idea; removing these submodules should not happen even for
a new major version of NumPy.


Subclassing of ndarray
^^^^^^^^^^^^^^^^^^^^^^

Subclassing of ``ndarray`` is a pain point.  ``ndarray`` was not (or at least
not well) designed to be subclassed.  Despite that, a lot of subclasses have
been created even within the NumPy code base itself, and some of those (e.g.
``MaskedArray``, ``astropy.units.Quantity``) are quite popular.  The main
problems with subclasses are:

- They make it hard to change ``ndarray`` in ways that would otherwise be
  backwards compatible.
- Some of them change the behavior of ndarray methods, making it difficult to
  write code that accepts array duck-types.

Subclassing ``ndarray`` has been officially discouraged for a long time.  Of
the most important subclasses, ``np.matrix`` will be deprecated (see gh-10142)
and ``MaskedArray`` will be kept in NumPy (`NEP 17
<http://www.numpy.org/neps/nep-0017-split-out-maskedarray.html>`_).
``MaskedArray`` will ideally be rewritten in a way such that it uses only
public NumPy APIs.  For subclasses outside of NumPy, more work is needed to
provide alternatives (e.g. mixins, see gh-9016 and gh-10446) or better support
for custom dtypes (see gh-2899).  Until that is done, subclasses need to be
taken into account when making change to the NumPy code base.  A future change
in NumPy to not support subclassing will certainly need a major version
increase.


Policy
------

1. Code changes that have the potential to silently change the results of a users'
   code must never be made (except in the case of clear bugs).
2. Code changes that break users' code (i.e. the user will see a clear exception)
   can be made, *provided the benefit is worth the cost* and suitable deprecation
   warnings have been raised first.
3. Deprecation warnings are in all cases warnings that functionality will be removed.
   If there is no intent to remove functionlity, then deprecation in documentation
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

Versioning:

- removal of deprecated code can be done in any minor (but not bugfix) release.
- for heavily used functionality (e.g. removal of ``np.matrix``, of a whole submodule,
  or significant changes to behavior for subclasses) the major version number shall
  be increased.

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

**Semantic versioning.**

This would change the versioning scheme for code removals; those could then
only be done when the major version number is increased.  Rationale for
rejection: semantic versioning is relatively common in software engineering,
however it is not at all common in the Python world.  Also, it would mean that
NumPy's version number simply starts to increase faster, which would be more
confusing than helpful. gh-10156 contains more discussion on this alternative.


Discussion
----------

TODO

This section may just be a bullet list including links to any discussions
regarding the NEP:

- This includes links to mailing list threads or relevant GitHub issues.


References and Footnotes
------------------------

.. [1] TODO


Copyright
---------

This document has been placed in the public domain. [1]_
