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


Motivation and Scope
--------------------

NumPy has a very large user base.  Those users rely on NumPy being stable
and the code they write that uses NumPy functionality to keep working.
NumPy is also actively maintained and improved -- and sometimes improvements
require or are made easier by breaking backwards compatibility.
Finally, there are trade-offs in stability for existing users vs. avoiding
errors or having a better user experience for new users.  These competing
needs often give rise to long debates and to delays in accepting or rejecting
contributions.  This NEP tries to address that by providing a policy as well
as examples and rationales for when it is or isn't a good idea to break
backwards compatibility.

In scope for this NEP are:

- Principles of NumPy's approach to backwards compatibility.
- How to deprecate functionality, and when to remove already deprecated
  functionality.
- Decision making process for deprecations and removals.

Out of scope are:

- Making concrete decisions about deprecations of particular functionality.
- NumPy's versioning scheme.


General principles
------------------

When considering proposed changes that are backwards incompatible, the
main principles the NumPy developers use when making a decision are:

1. Changes need to benefit more than they harm users.
2. NumPy is widely used so breaking changes should by default be assumed to be
   fairly harmful.
3. Decisions should be based on how they affect users and downstream packages.
   This should be based on usage data where possible. It does not matter whether
   this use contradicts the documentation or best practices.
4. The possibility of an incorrect result is much worse than an error or even crash.

When assessing the costs of proposed changes, keep in mind that most users do
not read the mailing list, do not notice deprecation warnings, and sometimes
wait more than one or two years before upgrading from their old version. And
that NumPy has millions of users, so "no one will do or use this" is very
likely incorrect.

Benefits include improved functionality, usability and performance, as well as
lower maintenance cost and improved future extensibility.

Fixes for clear bugs are exempt from this backwards compatibility policy.
However in case of serious impact on users (e.g. a downstream library doesn't
build anymore or would start giving incorrect results), even bug fixes may have
to be delayed for one or more releases.


Strategies related to deprecations
----------------------------------

Getting hard data on the impact of a deprecation of often difficult. Strategies
that can be used to assess such impact include:

- Use a code search engine ([1]_) or static ([2]_) or dynamic ([3]_) code
  analysis tools to determine where and how the functionality is used.
- Testing prominent downstream libraries against a development build of NumPy
  containing the proposed change to get real-world data on its impact.
- Making a change in master and reverting it, if needed, before a release. We
  do encourage other packages to test against NumPy's master branch, so this
  often turns up issues quickly.

If the impact is unclear or significant, it is often good to consider
alternatives to deprecations. For example, discouraging use in documentation
only, or moving the documentation for the functionality to a less prominent
place or even removing it completely. Commenting on open issues related to it
that they are low-prio or labeling them as "wontfix" will also be a signal to
users, and reduce the maintenance effort needing to be spent.


Implementing deprecations and removals
--------------------------------------

Deprecation warnings are necessary in all cases where functionality
will eventually be removed.  If there is no intent to remove functionality,
then it should not be deprecated either. A "please don't use this" in the
documentation or other type of warning should be used instead.

Deprecations:

- shall include the version number of the release in which the functionality
  was deprecated.
- shall include information on alternatives to the deprecated functionality, or a
  reason for the deprecation if no clear alternative is available.
- shall use ``VisibleDeprecationWarning`` rather than ``DeprecationWarning``
  for cases of relevance to end users. For cases only relevant to
  downstream libraries, a regular ``DeprecationWarning`` is fine.
  *Rationale: regular deprecation warnings are invisible by default; library
  authors should be aware how deprecations work and test for them, but we can't
  expect this from all users.*
- shall be listed in the release notes of the release where the deprecation is
  first present.
- shall not be introduced in micro (or bug fix) releases.
- shall set a ``stacklevel``, so the warning appears to come from the correct
  place.
- shall be mentioned in the documentation for the functionality. A
  ``.. deprecated::`` directive can be used for this.

Examples of good deprecation warnings (also note standard form of the comments
above the warning, helps when grepping):

.. code-block:: python

    # NumPy 1.15.0, 2018-09-02
    warnings.warn('np.asscalar(a) is deprecated since NumPy 1.16.0, use '
                  'a.item() instead', DeprecationWarning, stacklevel=3)

    # NumPy 1.15.0, 2018-02-10
    warnings.warn("Importing from numpy.testing.utils is deprecated "
                  "since 1.15.0, import from numpy.testing instead.",
                  DeprecationWarning, stacklevel=2)

    # NumPy 1.14.0, 2017-07-14
    warnings.warn(
        "Reading unicode strings without specifying the encoding "
        "argument is deprecated since NumPy 1.14.0. Set the encoding, "
        "use None for the system default.",
        np.VisibleDeprecationWarning, stacklevel=2)

.. code-block:: C

        /* DEPRECATED 2020-05-13, NumPy 1.20 */
        if (PyErr_WarnFormat(PyExc_DeprecationWarning, 1,
                matrix_deprecation_msg, ufunc->name, "first") < 0) {
            return NULL;
        }

Removal of deprecated functionality:

- shall be done after at least 2 releases (assuming the current 6-monthly
  release cycle; if that changes, there shall be at least 1 year between
  deprecation and removal).
- shall be listed in the release notes of the release where the removal happened.
- can be done in any minor (but not bugfix) release.

For backwards incompatible changes that aren't "deprecate and remove" but for
which code will start behaving differently, a ``FutureWarning`` should be
used. Release notes, mentioning version number and using ``stacklevel`` should
be done in the same way as for deprecation warnings. A ``.. versionchanged::``
directive can be used in the documentation to indicate when the behavior
changed:

.. code-block:: python

    def argsort(self, axis=np._NoValue, ...):
        """
        Parameters
        ----------
        axis : int, optional
            Axis along which to sort. If None, the default, the flattened array
            is used.

            ..  versionchanged:: 1.13.0
                Previously, the default was documented to be -1, but that was
                in error. At some future date, the default will change to -1, as
                originally intended.
                Until then, the axis should be given explicitly when
                ``arr.ndim > 1``, to avoid a FutureWarning.
        """
        ...
        warnings.warn(
            "In the future the default for argsort will be axis=-1, not the "
            "current None, to match its documentation and np.argsort. "
            "Explicitly pass -1 or None to silence this warning.",
            MaskedArrayFutureWarning, stacklevel=3)


Decision making
~~~~~~~~~~~~~~~

In concrete cases where this policy needs to be applied, decisions are made according
to the `NumPy governance model
<https://docs.scipy.org/doc/numpy/dev/governance/index.html>`_.

All deprecations must be proposed on the mailing list, in order to give everyone
with an interest in NumPy development a chance to comment. Removal of
deprecated functionality does not need discussion on the mailing list.


Functionality with more strict deprecation policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``numpy.random`` has its own backwards compatibility policy,
  see `NEP 19 <http://www.numpy.org/neps/nep-0019-rng-policy.html>`_.
- The file format for ``.npy`` and ``.npz`` files must not be changed in a backwards
  incompatible way.


Example cases
-------------

We now discuss a few concrete examples from NumPy's history to illustrate
typical issues and trade-offs.

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

- `PEP 387 - Backwards Compatibility Policy <https://www.python.org/dev/peps/pep-0387/>`__

.. [1] https://searchcode.com/

.. [2] https://github.com/Quansight-Labs/python-api-inspect

.. [3] https://github.com/data-apis/python-record-api

Copyright
---------

This document has been placed in the public domain.
