==================
NEP 36 — Fair Play
==================

:Author: Stéfan van der Walt <stefanv@berkeley.edu>
:Status: Draft
:Type: Informational
:Created: 2019-10-24
:Resolution: Draft


Abstract
--------

This document describes Rules of Play for engaging with the NumPy
project as an external entity.  It discusses usage of the name NumPy,
modified distributions of NumPy, and how to best communicate
with the project when patched versions are deployed.

Companies and developers will know after reading this NEP what kinds
of behavior the community would like to see, and which we consider
troublesome, bothersome, and unacceptable.

Motivation
----------

It has come to our attention that modified versions of NumPy are
distributed by external entities.  Sometimes, the patches are
innocuous, such as when they allow NumPy to function on the target
system.  In other cases, they add new functionality which causes
problems for the community.

For example, in December 2018, through the filing of `Issue #12515
<https://github.com/numpy/numpy/issues/12515>`__, it became known to
the NumPy project that an external company had published a version of
NumPy with an extended footprint.  In this specific instance, a bug
was filed against the `np.erf` function---a function that NumPy itself
had never provided.

In the months that followed, several conversations were had with the
entity involved, and a better understanding emerged of needs on either
side.  The entity agreed to publish the patches made against Python,
and a special label was added on the NumPy issue tracker to identify
issues pertaining to this specific distribution.

In another example, a security issue (CVE-2019-6446) was filed against
NumPy.  This forced many distributions to take action, most often by
changing a default keyword value, unfortunately again leading to
multiple versions of the NumPy API being deployed, varying across
distribution.

These are only two of many examples of such modified distributions,
which cause wasted time for our developers: identifying the release
from which the error came, how the release was modified from the
version we provide, and whether we are able to address related issues
(if not, where to relay the report to).

During a community call on `October 16th, 2019 
<https://github.com/numpy/archive/blob/master/status_meetings/status-2019-10-16.md>`__
the community decided to draft a
set of guidelines for companies and external developers to use when
they modify and redistribute NumPy.

Scope
-----

This document aims to define a minimal set of rules that, when
followed, will placate the NumPy developers.  Our hope, however, is
that developers who think that they need to modify NumPy will consider
contributing to the project, or using one of the many mechanisms we
have for allowing our APIs to operate on externally defined objects.

We encourage and exhort developers to first discuss their problem with
the NumPy developers on the `mailing list <>`__ before resorting to
patching and releasing their own version of the library.

Fair Play Rules
---------------

1. Do not reuse the NumPy name for projects not developed by the NumPy
   community.

   At time of writing, there are only a handful of `numpy`-named
   packages developed by the community, including `numpy`,
   `numpy-financial`, and `unumpy`.  We ask that external packages not
   include the phrase `numpy`, i.e., avoid names such as
   `mycompany_numpy`.

   NumPy is a trademark owned by NumFOCUS.

2. Do not republish modified versions of NumPy.

   Modified versions of NumPy make it very difficult for the
   developers to address bug reports, since we typically do not know
   which parts of NumPy have been modified.

   If you absolutely have to break this rule (and we implore you not
   to!), then make it absolutely clear in the `__version__` tag that
   you have modified NumPy, e.g.::

     >>> print(np.__version__)
     '1.17.2+mycompany.15`

   We understand that minor patches are often required to make a
   library work under a certain distribution.  E.g., Debian may patch
   NumPy so that it searches for optimized BLAS libraries in the
   correct locations.  But we ask that no substantive changes are
   made.

3. Do not extend NumPy's API footprint.

   If you absolutely have to break rule two, please do not add
   additional functions to the namespace.  NumPy's API is already
   quite large, and we are working hard to reduce it where feasible.
   Having additional functions exposed in distributed versions is
   confusing for users and developers alike.

4. *DO* use official mechanism to engage with the API.

   Protocols such as `__array_ufunc__
   <https://numpy.org/neps/nep-0013-ufunc-overrides.html>`__ and
   `__array_function__
   <https://numpy.org/neps/nep-0018-array-function-protocol.html>`__
   were designed to help external packages interact more easily with
   NumPy.  E.g., the latter allows objects from foreign libraries to
   pass through NumPy unharmed.  We actively encourage using any of
   these "officialy sanctioned" mechanisms for overriding or
   interacting with NumPy.

   If these mechanisms are deemed insufficient, please start a
   discussion on the mailing list before monkeypatching NumPy.

Questions & Answers
-------------------

**Q:** We would like to distribute an optimized version of NumPy that
utilizes special instructions for our company's CPU.  You recommend
against that, so what are we to do?

**A:** Please consider including the patches required in the official
NumPy repository.  Not only do we encourage such contributions, but we
already have optimized loops for some platforms available.

**Q:** We would like to ship a much faster version of FFT than NumPy
provides, but NumPy has no mechanism for overriding its FFT routines.
How do we proceed?

**A:** There are two solutions that we approve of: let the users
install your optimizations using a piece of code, such as::

  from my_company_accel import patch_numpy_fft
  patch_numpy_fft()

or have your distribution automatically perform the above, but print a
message to the terminal clearly stating what is happening::

  We are now patching NumPy for optimal performance under MyComp
  Special Platform.  Please direct all bug reports to
  https://mycomp.com/numpy-bugs

If you require additional mechanisms for overriding code, please
discuss this with the development team on the mailing list.

**Q:** We would like to distribute NumPy with faster linear algebra
routines. Are we allowed to do this?

**A:** Yes, this is explicitly supported by linking to a different
version of BLAS.

Discussion
----------

References and Footnotes
------------------------

Copyright
---------

This document has been placed in the public domain. [1]_
