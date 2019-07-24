==========================================================================================
NEP 29 — A standard community policy for dropping support of old Python and NumPy versions
==========================================================================================


:Author: Thomas A Caswell <tcaswell@gmail.com>, Andreas Mueller, Brian Granger, Madicken Munk, Ralf Gommers, Matt Haberland <mhaberla@calpoly.edu>, Matthias Bussonnier <bussonniermatthias@gmail.com>, Stefan van der Walt
:Status: Draft
:Type: Informational Track
:Created: 2019-07-13


Abstract
--------

All projects across the ecosystem should adopt a common time window
based policy for increasing the minimum version of Python and NumPy
that downstream projects support.  By standardizing this policy
across the community we will make it easier for downstream projects to
plan.

This is an unusual NEP in that it is recommendations for community-wide
policy and not for changes to NumPy itself.  However, we do not
currently have a place for SPEEPs (Scientific Python Ecosystem
Enhancement Proposals) and given NumPy's central role in our community
this seems like the proper place to document this.


This is being put forward by maintainers of Matplotlib, scikit-learn,
IPython, Jupyter, yt, SciPy, NumPy, and scikit-image.



Detailed description
--------------------

When making a new major or minor release, projects should support all
minor versions of ``Python`` initially released in the prior 42
months from their anticipated release date and all minor version of
NumPy released in the prior 24 months.


The diagram::

       Jan 16      Jan 17      Jan 18      Jan 19      Jan 20
       |           |           |           |           |
  +++++|+++++++++++|+++++++++++|+++++++++++|+++++++++++|++++++++++++
   |              |                  |               |
   py 3.5.0       py 3.6.0           py 3.7.0        py 3.8.0
  |-----------------------------------------> Feb19
            |-----------------------------------------> Dec19
                      |-----------------------------------------> Nov20

shows the support windows for ``Python``.  A project with a major or
minor release in Feb19 should support py35 and newer, a project with a
major or minor release in Dec19 should support py36 and newer, and a
project with a major or minor release in Nov20 should support py37 and
newer.

The current Python release cadence is 18 months so a 42 month window
ensures that there will always be at least two versions of ``Python``
in the window.  By padding the window by 6 months from the anticipated
``Python`` cadence we avoid the edge cases where a project releases
the month after ``Python`` and would effectively only support one
version of ``Python`` that has an install base (and then the window
is 42).  This buffer also makes us resilient to minor fluctuations /
delays in the ``Python`` release schedule.

Because the threshold for dropping a version of ``Python`` is based
on historical release dates and a project's plans, the decision to drop
support for a given version of ``Python`` can be made very early in
the release process.

There will be some unavoidable mismatch in supported version of
``Python`` between packages immediately after a version of
``Python`` ages out.  However this should not last longer than one
release cycle of each of the projects, and when a given project
does a minor or major release, it is guaranteed that there will be a
stable release of all other projects that support the set of
``Python`` the new release will support.


Implementation
--------------

We suggest that all projects adopt the following language into their
development guidelines:


   - support minor versions of ``Python`` initially released
     42 months prior to our planned release date
   - always support at least the 2 latest minor versions of ``Python``
   - support minor versions of ``numpy`` initially released in the 24
     months prior to our planned release date or oldest that supports the
     minimum Python version (whichever is higher)

   The minimum supported version of ``Python`` will be set to
   ``python_requires`` in ``setup`` and all supported versions of
   Python will be in the test matrix and have binary artifacts built
   for releases.

   We will bump the minimum Python and NumPy versions as we can on
   every minor and major release, but never on a patch release.

For other dependencies, adopt similar time windows of 24 months or
shorter.


Backward compatibility
----------------------

No issues other than the intentional dropping of old version of
``Python`` and dependencies.


Alternatives
------------

Ad-Hoc
~~~~~~

Projects could on every release evaluate if they want to increase
the minimum version of Python supported.  While this is a notionally
simple policy, it makes it hard for downstream users to predict what
the future minimum versions will be.  As there is no objective threshold
to when the minimum version should be dropped, it is easy for these
discussions to devolve into [bike shedding](https://en.wikipedia.org/wiki/Wikipedia:Avoid_Parkinson%27s_bicycle-shed_effect) and acrimony.


All Python Software Foundation supported versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a very clear and conservative approach.  However, it means that
there is 4 year lag between when new language features come into the
language and when the projects are able to use them.  Additionally,
for projects that have a significant component of compiled extensions
this requires building many binary artifacts for each release.

For the case of NumPy, many projects carry workarounds to bugs that
are fixed in subsequent versions of NumPy.  Being proactive about
increasing the minimum version of NumPy will allow downstream
packages to carry fewer version-specific patches.



Default version on Linux distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The policy could be to support the version of Python that ships by
default in the latest Ubuntu LTS or CentOS/RHEL release.  However, we
would still have to standardize across the community which
distribution we are following.

By following the versions supported by major Linux distributions, we
are giving up technical control of our projects to external
organizations that may have different motivations and concerns than we
do.

N minor versions of Python
~~~~~~~~~~~~~~~~~~~~~~~~~~

Given the current release cadence of the Python, the proposed time
(42 months) is roughly equivalent to "the last two" Python minor
versions.  However, if Python changes their release cadence, any rule
based on the number of minor releases will need to be changed.


Time window on the X.Y.1 Python release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As the first bug fix release is typically a few months after the
initial release, you can achieve the same effect by using a large delay
from the X.Y.0 release which seems simpler to explain.


Discussion
----------


References and Footnotes
------------------------


Copyright
---------

This document has been placed in the public domain.
