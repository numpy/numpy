==================================================
NEP 32 — Remove the financial functions from NumPy
==================================================

:Author: Warren Weckesser <warren.weckesser@gmail.com>
:Status: Accepted
:Type: Standards Track
:Created: 2019-08-30
:Resolution: https://mail.python.org/pipermail/numpy-discussion/2019-September/080074.html


Abstract
--------

We propose deprecating and ultimately removing the financial functions [1]_
from NumPy.  The functions will be moved to an independent repository,
and provided to the community as a separate package with the name
``numpy_financial``.


Motivation and scope
--------------------

The NumPy financial functions [1]_ are the 10 functions ``fv``, ``ipmt``,
``irr``, ``mirr``, ``nper``, ``npv``, ``pmt``, ``ppmt``, ``pv`` and ``rate``.
The functions provide elementary financial calculations such as future value,
net present value, etc. These functions were added to NumPy in 2008 [2]_.

In May, 2009, a request by Joe Harrington to add a function called ``xirr`` to
the financial functions triggered a long thread about these functions [3]_.
One important point that came up in that thread is that a "real" financial
library must be able to handle real dates.  The NumPy financial functions do
not work with actual dates or calendars.  The preference for a more capable
library independent of NumPy was expressed several times in that thread.

In June, 2009, D. L. Goldsmith expressed concerns about the correctness of the
implementations of some of the financial functions [4]_.  It was suggested then
to move the financial functions out of NumPy to an independent package.

In a GitHub issue in 2013 [5]_, Nathaniel Smith suggested moving the financial
functions from the top-level namespace to ``numpy.financial``.  He also
suggested giving the functions better names.  Responses at that time included
the suggestion to deprecate them and move them from NumPy to a separate
package.  This issue is still open.

Later in 2013 [6]_, it was suggested on the mailing list that these functions
be removed from NumPy.

The arguments for the removal of these functions from NumPy:

* They are too specialized for NumPy.
* They are not actually useful for "real world" financial calculations, because
  they do not handle real dates and calendars.
* The definition of "correctness" for some of these functions seems to be a
  matter of convention, and the current NumPy developers do not have the
  background to judge their correctness.
* There has been little interest among past and present NumPy developers
  in maintaining these functions.

The main arguments for keeping the functions in NumPy are:

* Removing these functions will be disruptive for some users.  Current users
  will have to add the new ``numpy_financial`` package to their dependencies,
  and then modify their code to use the new package.
* The functions provided, while not "industrial strength", are apparently
  similar to functions provided by spreadsheets and some calculators.  Having
  them available in NumPy makes it easier for some developers to migrate their
  software to Python and NumPy.

It is clear from comments in the mailing list discussions and in the GitHub
issues that many current NumPy developers believe the benefits of removing
the functions outweigh the costs.  For example, from [5]_::

    The financial functions should probably be part of a separate package
    -- Charles Harris

    If there's a better package we can point people to we could just deprecate
    them and then remove them entirely... I'd be fine with that too...
    -- Nathaniel Smith

    +1 to deprecate them. If no other package exists, it can be created if
    someone feels the need for that.
    -- Ralf Gommers

    I feel pretty strongly that we should deprecate these. If nobody on numpy’s
    core team is interested in maintaining them, then it is purely a drag on
    development for NumPy.
    -- Stephan Hoyer

And from the 2013 mailing list discussion, about removing the functions from
NumPy::

    I am +1 as well, I don't think they should have been included in the first
    place.
    -- David Cournapeau

But not everyone was in favor of removal::

    The fin routines are tiny and don't require much maintenance once
    written.  If we made an effort (putting up pages with examples of common
    financial calculations and collecting those under a topical web page,
    then linking to that page from various places and talking it up), I
    would think they could attract users looking for a free way to play with
    financial scenarios.  [...]
    So, I would say we keep them.  If ours are not the best, we should bring
    them up to snuff.
    -- Joe Harrington

For an idea of the maintenance burden of the financial functions, one can
look for all the GitHub issues [7]_ and pull requests [8]_ that have the tag
``component: numpy.lib.financial``.

One method for measuring the effect of removing these functions is to find
all the packages on GitHub that use them.  Such a search can be performed
with the ``python-api-inspect`` service [9]_.  A search for all uses of the
NumPy financial functions finds just eight repositories.  (See the comments
in [5]_ for the actual SQL query.)


Implementation
--------------

* Create a new Python package, ``numpy_financial``, to be maintained in the
  top-level NumPy github organization.  This repository will contain the
  definitions and unit tests for the financial functions.  The package will
  be added to PyPI so it can be installed with ``pip``.
* Deprecate the financial functions in the ``numpy`` namespace, beginning in
  NumPy version 1.18. Remove the financial functions from NumPy version 1.20.


Backward compatibility
----------------------

The removal of these functions breaks backward compatibility, as explained
earlier.  The effects are mitigated by providing the ``numpy_financial``
library.


Alternatives
------------

The following alternatives were mentioned in [5]_:

* *Maintain the functions as they are (i.e. do nothing).*
  A review of the history makes clear that this is not the preference of many
  NumPy developers.  A recurring comment is that the functions simply do not
  belong in NumPy.  When that sentiment is combined with the history of bug
  reports and the ongoing questions about the correctness of the functions, the
  conclusion is that the cleanest solution is deprecation and removal.
* *Move the functions from the ``numpy`` namespace to ``numpy.financial``.*
  This was the initial suggestion in [5]_.  Such a change does not address the
  maintenance issues, and doesn't change the misfit that many developers see
  between these functions and NumPy.  It causes disruption for the current
  users of these functions without addressing what many developers see as the
  fundamental problem.


Discussion
----------

Links to past mailing list discussions, and to relevant GitHub issues and pull
requests, have already been given.  The announcement of this NEP was made on
the NumPy-Discussion mailing list on 3 September 2019 [10]_, and on the
PyData mailing list on 8 September 2019 [11]_.  The formal proposal to accept
the NEP was made on 19 September 2019 [12]_; a notification was also sent to
PyData (same thread as [11]_).  There have been no substantive objections.


References and footnotes
------------------------

.. [1] Financial functions,
   https://numpy.org/doc/1.17/reference/routines.financial.html

.. [2] Numpy-discussion mailing list, "Simple financial functions for NumPy",
   https://mail.python.org/pipermail/numpy-discussion/2008-April/032353.html

.. [3] Numpy-discussion mailing list, "add xirr to numpy financial functions?",
   https://mail.python.org/pipermail/numpy-discussion/2009-May/042645.html

.. [4] Numpy-discussion mailing list, "Definitions of pv, fv, nper, pmt, and rate",
   https://mail.python.org/pipermail/numpy-discussion/2009-June/043188.html

.. [5] Get financial functions out of main namespace,
   https://github.com/numpy/numpy/issues/2880

.. [6] Numpy-discussion mailing list, "Deprecation of financial routines",
   https://mail.python.org/pipermail/numpy-discussion/2013-August/067409.html

.. [7] ``component: numpy.lib.financial`` issues,
   https://github.com/numpy/numpy/issues?utf8=%E2%9C%93&q=is%3Aissue+label%3A%22component%3A+numpy.lib.financial%22+

.. [8] ``component: numpy.lib.financial`` pull requests,
   https://github.com/numpy/numpy/pulls?utf8=%E2%9C%93&q=is%3Apr+label%3A%22component%3A+numpy.lib.financial%22+

.. [9] Quansight-Labs/python-api-inspect,
   https://github.com/Quansight-Labs/python-api-inspect/

.. [10] Numpy-discussion mailing list, "NEP 32: Remove the financial functions
   from NumPy"
   https://mail.python.org/pipermail/numpy-discussion/2019-September/079965.html

.. [11] PyData mailing list (pydata@googlegroups.com), "NumPy proposal to
   remove the financial functions.
   https://mail.google.com/mail/u/0/h/1w0mjgixc4rpe/?&th=16d5c38be45f77c4&q=nep+32&v=c&s=q

.. [12] Numpy-discussion mailing list, "Proposal to accept NEP 32: Remove the
   financial functions from NumPy"
   https://mail.python.org/pipermail/numpy-discussion/2019-September/080074.html

Copyright
---------

This document has been placed in the public domain.
