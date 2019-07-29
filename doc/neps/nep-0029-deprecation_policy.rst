==================================================================================
NEP 29 — Recommend Python and Numpy version support as a community policy standard
==================================================================================


:Author: Thomas A Caswell <tcaswell@gmail.com>, Andreas Mueller, Brian Granger, Madicken Munk, Ralf Gommers, Matt Haberland <mhaberla@calpoly.edu>, Matthias Bussonnier <bussonniermatthias@gmail.com>, Stefan van der Walt
:Status: Draft
:Type: Informational Track
:Created: 2019-07-13


Abstract
--------

This NEP recommends and encourages all projects across the Scientific
Python ecosystem to adopt a common "time window-based" policy for
support of Python and NumPy versions. Standardizing a recommendation
for project support of minimum Python and NumPy versions will improve
downstream project planning.

This is an unusual NEP in that it offers recommendations for
community-wide policy and not for changes to NumPy itself.  Since a
common place for SPEEPs (Scientific Python Ecosystem Enhancement
Proposals) does not exist and given NumPy's central role in the
ecosystem, a NEP provides a visible place to document the proposed
policy.

This NEP is being put forward by maintainers of Matplotlib, scikit-learn,
IPython, Jupyter, yt, SciPy, NumPy, and scikit-image.



Detailed description
--------------------

For the purposes of this NEP we assume semantic versioning and define:

*major version*
   A release that change the first number (e.g. X.0.0)

*minor version*
   A release that changes the second number (e.g x.Y.0)

*patch version*
   A release that changes the third number (e.g. x.y.Z)


When a project creates a new major or minor version, we recommend that
the project should support all minor versions of Python introduced
and released in the prior 42 months ~~from their anticipated release
date~~ and all minor versions of NumPy released in the prior 24
months.


The diagram::

       Jan 16      Jan 17      Jan 18      Jan 19      Jan 20
       |           |           |           |           |
  +++++|+++++++++++|+++++++++++|+++++++++++|+++++++++++|++++++++++++
   |              |                  |               |
   py 3.5.0       py 3.6.0           py 3.7.0        py 3.8.0
  |-----------------------------------------> Feb19
            |-----------------------------------------> Dec19
                      |-----------------------------------------> Nov20

shows the 42 month support windows for Python.  A project with a
major or minor version release in Feb19 should support py35 and newer,
a project with a major or minor version release in Dec19 should
support py36 and newer, and a project with a major or minor version
release in Nov20 should support py37 and newer.

The current Python release cadence is 18 months so a 42 month window
ensures that there will always be at least two versions of Python
in the window.  By padding the window by 6 months from the anticipated
Python cadence we avoid the edge cases where a project releases
the month after Python and would effectively only support one
version of Python that has an installed base.
This six month buffer provides resilience to minor fluctuations /
delays in the Python release schedule.

Because Python version support is based on historical release
dates, a 42 month time window, and a project's plans, a project can
decide to drop a given version of Python very early in the release
process.

While there will be some unavoidable mismatch in supported version of
Python between if a project release occurs immediately after a
version of Python ages out.  This should not last longer than one
release cycle of each of the projects, and when a given project does a
minor or major release, it is guaranteed that there will be a stable
release of all other projects that support the set of Python the
new release will support.

If there is a Python 4 or a NumPy 2 this policy will have to be
reviewed in light of the community's and projects' best interests.


Support Table
~~~~~~~~~~~~~

============ ====== =====
Date         Python NumPy
------------ ------ -----
Jan 16, 2019 3.5+   1.13+
Mar 14, 2019 3.6+   1.13+
Jun 08, 2019 3.6+   1.14+
Jan 07, 2020 3.6+   1.15+
Jun 23, 2020 3.7+   1.15+
Jul 23, 2020 3.7+   1.16+
Jan 13, 2021 3.7+   1.17+
Dec 26, 2021 3.8+   1.17+
============ ====== =====


Drop Schedule
~~~~~~~~~~~~~

::

  On Jan 16, 2019 drop support for Numpy 1.12 (initially released on Jan 15, 2017)
  On Mar 14, 2019 drop support for Python 3.5 (initially released on Sep 13, 2015)
  On Jun 08, 2019 drop support for Numpy 1.13 (initially released on Jun 07, 2017)
  On Jan 07, 2020 drop support for Numpy 1.14 (initially released on Jan 06, 2018)
  On Jun 23, 2020 drop support for Python 3.6 (initially released on Dec 23, 2016)
  On Jul 23, 2020 drop support for Numpy 1.15 (initially released on Jul 23, 2018)
  On Jan 13, 2021 drop support for Numpy 1.16 (initially released on Jan 13, 2019)
  On Dec 26, 2021 drop support for Python 3.7 (initially released on Jun 27, 2018)


Implementation
--------------

We suggest that all projects adopt the following language into their
development guidelines:


   - This project supports minor versions of Python initially released
     42 months prior to a planned project release date.
   - The project will always support at least the 2 latest minor
     versions of Python
   - support minor versions of ``numpy`` initially released in the 24
     months prior to a planned project release date or the oldest
     version that supports the minimum Python version (whichever is
     higher)

   The minimum supported version of Python will be set to
   ``python_requires`` in ``setup``.  All supported versions of
   Python will be in the test matrix and have binary artifacts built
   for releases.

   The project will bump (adjust upward) the minimum Python and NumPy
   version support on every minor and major release, but never on a
   patch release.

For other dependencies, adopt similar time windows of 24 months or
shorter.


Backward compatibility
----------------------

No backward compatibility issues.

Alternatives
------------

Ad-Hoc version support
~~~~~~~~~~~~~~~~~~~~~~

A project could on every release evaluate whether to increase
the minimum version of Python supported.
As a major downside, an ad-hoc approach makes it hard for downstream users to predict what
the future minimum versions will be.  As there is no objective threshold
to when the minimum version should be dropped, it is easy for these
version support discussions to devolve into [bike shedding](https://en.wikipedia.org/wiki/Wikipedia:Avoid_Parkinson%27s_bicycle-shed_effect) and acrimony.


All CPython supported versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CPython supported versions of Python are listed in the Python
Developers Guide and the Python PEPs. Supporting these are a very
clear and conservative approach.  However, it means that there is 4
year lag between when new language features come into the language and
when the projects are able to use them.  Additionally, for projects
that have a significant component of compiled extensions this requires
building many binary artifacts for each release.

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
versions.  However, if Python changes their release cadence substantially, any rule
based solely on the number of minor releases may need to be changed to remain sensible.


Time window on the X.Y.1 Python release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As the first bug fix release is typically a few months after the
initial release, you can achieve the same effect by using a large delay
from the X.Y.0 release which seems simpler to explain.


Discussion
----------


References and Footnotes
------------------------

Code to generate support and drop schedule tables ::

  from datetime import datetime, timedelta

  data = """Jan 15, 2017: Numpy 1.12
  Sep 13, 2015: Python 3.5
  Jun 27, 2018: Python 3.7
  Dec 23, 2016: Python 3.6
  Jun 07, 2017: Numpy 1.13
  Jan 06, 2018: Numpy 1.14
  Jul 23, 2018: Numpy 1.15
  Jan 13, 2019: Numpy 1.16
  """

  releases = []

  plus42 = timedelta(days=int(365*3.5 + 1))
  plus24 = timedelta(days=int(365*2 + 1))

  for line in data.splitlines():
      date, project_version = line.split(':')
      project, version = project_version.strip().split(' ')
      release = datetime.strptime(date, '%b %d, %Y')
      if project.lower() == 'numpy':
          drop = release + plus24
      else:
          drop = release + plus42
      releases.append((drop, project, version, release))

  releases = sorted(releases, key=lambda x: x[0])

  minpy = '3.8+'
  minnum = '1.17+'

  toprint_drop_dates = ['']
  toprint_support_table = []
  for d, p, v, r in releases[::-1]:
      df = d.strftime('%b %d, %Y')
      toprint_drop_dates.append(
          f'On {df} drop support for {p} {v} '
          f'(initially released on {r.strftime("%b %d, %Y")})')
      toprint_support_table.append(f'{df} {minpy:<6} {minnum:<5}')
      if p.lower() == 'numpy':
          minnum = v+'+'
      else:
          minpy = v+'+'

  for e in toprint_drop_dates[::-1]:
      print(e)

  print('============ ====== =====')
  print('Date         Python NumPy')
  print('------------ ------ -----')
  for e in toprint_support_table[::-1]:
      print(e)
  print('============ ====== =====')


Copyright
---------

This document has been placed in the public domain.
