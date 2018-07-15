=======================================================
NEP 23 - Backwards compatibility and deprecation policy
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
NumPy is actively maintained and improved - sometimes improvements require,
or are made much easier, by breaking backwards compatibility.
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

We now discuss a number of concrete examples. TODO

*Changing the behavior of a kewyord*
`np.histogram` had a `normed` keyword that had behavior that could be considered
either suboptimal or broken (depending on ones opinion on the statistics).
First, a new keyword `new_behavior=False` was introduced, this was then switched
over to True two releases later, and finally it was removed again.
Later a new keyword `density` was introduced.  TODO: finish this

What should have happened: TODO

*Returning a view rather than a copy*
The `ndarray.diag` method used to return a copy.  A view would be better for both
performance and design consistency.  This change was introduced, with noisy
warnings in case users were setting elements of the copy.

What should have happened: nothing.  This change did result in users silently
getting different results if they upgraded multiple versions or simply missed
the warnings.  The change was not worth the cost.

*Disallowing indexing with floats*
happened, caught a number of bugs, was disruptive, but worth the cost.

TODO: check other recent deprecations
- `np.sum(generator)` (gh-10670)
- truth testing on empty arrays (gh-9718)
- removing deprecated boolean indexing (gh-8312)

**Examples of features not added because of backwards compatibility**

*Removing the financial functions*
The financial functions (e.g. `np.pmt`) are badly named, are present in the
main NumPy namespace, and clearly aren't in scope for NumPy.  They were added
early on because they were present in Numeric (TODO: check).  At the moment,
they don't cause a lot of overhead, but there are multiple issues and PRs a
year for them which cost maintainer time to deal with.  And they clutter up
the `numpy` namespace.  Discussion in 2013 was mostly in favor of removal,
but there were some dissenting voices (see gh-TODO).  This case is borderline,
but given that they're clearly out of scope, deprecation and removal can be
proposed.


Subclassing of ndarray
^^^^^^^^^^^^^^^^^^^^^^

Subclassing of `ndarray` is a pain point.  `ndarray` was not (or at least not well)
designed to be subclassed.  Despite that, a lot of subclasses have been created
even within the NumPy code base itself, and some of those (e.g. `MaskedArray`, TODO: 
AstroPy ref) are quite popular.  The main problems with subclasses are:

- They make it hard to change `ndarray` in ways that would otherwise be backwards
  compatible.
- Some of them change the behavior of ndarray methods, making it difficult to write
  code that accepts array duck-types.

Subclassing `ndarray` has been officially discouraged for a long time.  Of the most
important subclasses, `np.matrix` will be deprecated (TODO: ref) and `MaskedArray`
will be kept in NumPy (TODO: ref).  `MaskedArray` will ideally be rewritten in a way
such that it uses only public NumPy APIs.  For subclasses outside of NumPy, more
work is needed to provide alternatives (e.g. Mixins (REF) or better support for
custom dtypes (REF)).  Until that is done, subclasses need to be taken into account
when making change to the NumPy code base.  A future change in NumPy to not support
subclassing will certainly need a major version increase.


Policy
------

1. Code changes that have the potential to silently change the results of a users'
   code must never be made (except in the case of clear bugs).
2. Code changes that break users' code (i.e. the user will see a clear exception)
   can be made, *provided the benefit is worth the cost and suitable deprecation
   warnings have been raised first.
3. Deprecation warnings are in all cases warnings that functionality will be removed.
   If there is no intent to remove functionlity, then deprecation in documentation
   only or other types of warnings shall be used.
4. Deprecations for stylistic reasons (e.g. consistency between functions) is
   strongly discouraged.

Deprecations:
- shall include the version numbers of both when the functionality was deprecated
  and when it will be removed.
- shall include information on alternatives to the deprecated functionality, or a
  reason for the deprecation if no clear alternative is available.
- shall use `VisibleDeprecationWarning` rather than `DeprecationWarning` for cases
  of relevance to end users (as opposed to cases only relevant to libraries building
  on top of NumPy).
- shall be listed in the release notes of the release where the deprecation happened.

Removal of deprecated functionality:
- shall be done after 2 releases (assuming a 6-monthly release cycle; if that changes,
  there shall be at least 1 year between deprecation and removal).
- shall be listed in the release notes of the release where the removal happened.

Versioning:
- removal of deprecated code can be done in any minor (but not bugfix) release.
- for heavily used functionality (e.g. removal of `np.matrix`, of a whole submodule,
  or significant changes to behavior for subclasses) the major version number shall
  be increased.

In concrete cases where this policy needs to be applied, decisions are made according
to <link to governance doc>.

Functionality with more strict policies:
- `numpy.random` has its own backwards compatibility policy,
  see `nep-0019-rng-policy`_.
- The file format for `.npy` and `.npz` files must not be changed in a backwards
  incompatible way.


Alternatives
------------

*Being more agressive with deprecations.*  The goal of being more agressive is
to allow NumPy to move forward faster.  This would avoid others inventing their
own solutions (often in multiple places), as well as be a benefit to users
without a legacy code base.  We reject this alternative because of the place
NumPy has in the scientific Python ecosystem - being fairly conservative is
required in order to not increase the extra maintenance for downstream
libraries and end users to an unacceptable level.

*Semantic versioning.*  This would change the versioning scheme for code removals;
those could then only be done when the major version number is increased.
Rationale for rejection: semantic versioning is relatively common in software
engineering, however it is not at all common in Python.  Also, it would mean
that NumPy's version number simply starts to increase faster, which would
be more confusing than helpful. TODO: link issue.

Discussion
----------

This section may just be a bullet list including links to any discussions
regarding the NEP:

- This includes links to mailing list threads or relevant GitHub issues.


References and Footnotes
------------------------

.. [1] TODO


Copyright
---------

This document has been placed in the public domain. [1]_
