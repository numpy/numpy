=====================================================
NEP 3 — Cleaning the math configuration of numpy.core
=====================================================

:Author: David Cournapeau
:Contact: david@ar.media.kyoto-u.ac.jp
:Date: 2008-09-04
:Status: Deferred

Executive summary
=================

Before building numpy.core, we use some configuration tests to gather some
information about available math functions. Over the years, the configuration
became convoluted, to the point it became difficult to support new platforms
easily.

The goal of this proposal is to clean the configuration of the math
capabilities for easier maintenance.

Current problems
================

Currently, the math configuration mainly test for some math functions, and
configure numpy accordingly. But instead of testing each desired function
independently, the current system has been developed more as workarounds
particular platform oddities, using platform implicit knowledge. This is
against the normal philosophy of testing for capabilities only, which is the
autoconf philosophy, which showed the path toward portability (on Unix at
least) [1] This causes problems because modifying or adding configuration on
existing platforms break the implicit assumption, without a clear solution.

For example, on windows, when numpy is built with mingw, it would be nice to
enforce the configuration sizeof(long double) == sizeof(double) because mingw
uses the MS runtime, and the MS runtime does not support long double.
Unfortunately, doing so breaks the mingw math function detection, because of
the implicit assumption that mingw has a configuration sizeof(long double) !=
sizeof(double).

Another example is the testing for set of functions using only one function: if
expf is found, it is assumed that all basic float functions are available.
Instead, each function should be tested independently (expf, sinf, etc...).

Requirements
============

We have two strong requirements:
	- it should not break any currently supported platform
	- it should not make the configuration much slower (1-2 seconds are
	  acceptable)

Proposal
========

We suggest to break any implicit assumption, and test each math function
independently from each other, as usually done by autoconf. Since testing for a
vast set of functions can be time consuming, we will use a scheme similar to
AC_CHECK_FUNCS_ONCE in autoconf, that is test for a set of function at once,
and only in the case it breaks, do the per function check. When the first check
works, it should be as fast as the current scheme, except that the assumptions
are explicitly checked (all functions implied by HAVE_LONGDOUBLE_FUNCS would
be checked together, for example).

Issues
======

Static vs non static ? For basic functions, shall we define them static or not ?

License
=======

This document has been placed in the public domain.

[1]: Autobook here
