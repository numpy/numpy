==================================================================================
NEP 29 — Recommend Python and Numpy version support as a community policy standard
==================================================================================


:Author: Thomas A Caswell <tcaswell@gmail.com>, Andreas Mueller, Brian Granger, Madicken Munk, Ralf Gommers, Matt Haberland <mhaberla@calpoly.edu>, Matthias Bussonnier <bussonniermatthias@gmail.com>, Stefan van der Walt <stefanv@berkeley.edu>
:Status: Final
:Type: Informational
:Created: 2019-07-13
:Resolution: https://mail.python.org/pipermail/numpy-discussion/2019-October/080128.html


Abstract
--------

This NEP recommends that all projects across the Scientific
Python ecosystem adopt a common "time window-based" policy for
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
   A release that changes the first number (e.g. X.0.0)

*minor version*
   A release that changes the second number (e.g 1.Y.0)

*patch version*
   A release that changes the third number (e.g. 1.1.Z)


When a project releases a new major or minor version, we recommend that
they support at least all minor versions of Python
introduced and released in the prior 42 months *from the
anticipated release date* with a minimum of 2 minor versions of
Python, and all minor versions of NumPy released in the prior 24
months *from the anticipated release date* with a minimum of 3
minor versions of NumPy.


Consider the following timeline::

       Jan 16      Jan 17      Jan 18      Jan 19      Jan 20
       |           |           |           |           |
  +++++|+++++++++++|+++++++++++|+++++++++++|+++++++++++|++++++++++++
   |              |                  |               |
   py 3.5.0       py 3.6.0           py 3.7.0        py 3.8.0
  |-----------------------------------------> Feb19
            |-----------------------------------------> Dec19
                      |-----------------------------------------> Nov20

It shows the 42 month support windows for Python.  A project with a
major or minor version release in February 2019 should support Python 3.5 and newer,
a project with a major or minor version released in December 2019 should
support Python 3.6 and newer, and a project with a major or minor version
release in November 2020 should support Python 3.7 and newer.

The current Python release cadence is 18 months so a 42 month window
ensures that there will always be at least two minor versions of Python
in the window.  The window is extended 6 months beyond the anticipated two-release
interval for Python to provides resilience against small fluctuations /
delays in its release schedule.

Because Python minor version support is based only on historical
release dates, a 42 month time window, and a planned project release
date, one can predict with high confidence when a project will be able
to drop any given minor version of Python.  This, in turn, could save
months of unnecessary maintenance burden.

If a project releases immediately after a minor version of Python
drops out of the support window, there will inevitably be some
mismatch in supported versions—but this situation should only last
until other projects in the ecosystem make releases.

Otherwise, once a project does a minor or major release, it is
guaranteed that there will be a stable release of all other projects
that, at the source level, support the same set of Python versions
supported by the new release.

If there is a Python 4 or a NumPy 2 this policy will have to be
reviewed in light of the community's and projects' best interests.


Support Table
~~~~~~~~~~~~~

============ ====== =====
Date         Python NumPy
------------ ------ -----
Jan 07, 2020 3.6+   1.15+
Jun 23, 2020 3.7+   1.15+
Jul 23, 2020 3.7+   1.16+
Jan 13, 2021 3.7+   1.17+
Jul 26, 2021 3.7+   1.18+
Dec 26, 2021 3.8+   1.18+
Apr 14, 2023 3.9+   1.18+
============ ====== =====


Drop Schedule
~~~~~~~~~~~~~

::

  On next release, drop support for Python 3.5 (initially released on Sep 13, 2015)
  On Jan 07, 2020 drop support for Numpy 1.14 (initially released on Jan 06, 2018)
  On Jun 23, 2020 drop support for Python 3.6 (initially released on Dec 23, 2016)
  On Jul 23, 2020 drop support for Numpy 1.15 (initially released on Jul 23, 2018)
  On Jan 13, 2021 drop support for Numpy 1.16 (initially released on Jan 13, 2019)
  On Jul 26, 2021 drop support for Numpy 1.17 (initially released on Jul 26, 2019)
  On Dec 26, 2021 drop support for Python 3.7 (initially released on Jun 27, 2018)
  On Apr 14, 2023 drop support for Python 3.8 (initially released on Oct 14, 2019)


Implementation
--------------

We suggest that all projects adopt the following language into their
development guidelines:

   This project supports:

   - All minor versions of Python released 42 months prior to the
     project, and at minimum the two latest minor versions.
   - All minor versions of ``numpy`` released in the 24 months prior
     to the project, and at minimum the last three minor versions.

   In ``setup.py``, the ``python_requires`` variable should be set to
   the minimum supported version of Python.  All supported minor
   versions of Python should be in the test matrix and have binary
   artifacts built for the release.

   Minimum Python and NumPy version support should be adjusted upward
   on every major and minor release, but never on a patch release.


Backward compatibility
----------------------

No backward compatibility issues.

Alternatives
------------

Ad-Hoc version support
~~~~~~~~~~~~~~~~~~~~~~

A project could, on every release, evaluate whether to increase
the minimum version of Python supported.
As a major downside, an ad-hoc approach makes it hard for downstream users to predict what
the future minimum versions will be.  As there is no objective threshold
to when the minimum version should be dropped, it is easy for these
version support discussions to devolve into `bike shedding <https://en.wikipedia.org/wiki/Wikipedia:Avoid_Parkinson%27s_bicycle-shed_effect>`_ and acrimony.


All CPython supported versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CPython supported versions of Python are listed in the Python
Developers Guide and the Python PEPs. Supporting these is a very clear
and conservative approach.  However, it means that there exists a four
year lag between when a new features is introduced into the language
and when a project is able to use it.  Additionally, for projects with
compiled extensions this requires building many binary artifacts for
each release.

For the case of NumPy, many projects carry workarounds to bugs that
are fixed in subsequent versions of NumPy.  Being proactive about
increasing the minimum version of NumPy allows downstream
packages to carry fewer version-specific patches.



Default version on Linux distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The policy could be to support the version of Python that ships by
default in the latest Ubuntu LTS or CentOS/RHEL release.  However, we
would still have to standardize across the community which
distribution to follow.

By following the versions supported by major Linux distributions, we
are giving up technical control of our projects to external
organizations that may have different motivations and concerns than we
do.


N minor versions of Python
~~~~~~~~~~~~~~~~~~~~~~~~~~

Given the current release cadence of the Python, the proposed time (42
months) is roughly equivalent to "the last two" Python minor versions.
However, if Python changes their release cadence substantially, any
rule based solely on the number of minor releases may need to be
changed to remain sensible.

A more fundamental problem with a policy based on number of Python
releases is that it is hard to predict when support for a given minor
version of Python will be dropped as that requires correctly
predicting the release schedule of Python for the next 3-4 years.  A
time-based rule, in contrast, only depends on past events
and the length of the support window.


Time window from the X.Y.1 Python release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is equivalent to a few month longer support window from the X.Y.0
release.  This is because X.Y.1 bug-fix release is typically a few
months after the X.Y.0 release, thus a N month window from X.Y.1 is
roughly equivalent to a N+3 month from X.Y.0.

The X.Y.0 release is naturally a special release.  If we were to
anchor the window on X.Y.1 we would then have the discussion of why
not X.Y.M?


Discussion
----------


References and Footnotes
------------------------

Code to generate support and drop schedule tables ::

  from datetime import datetime, timedelta

  data = """Jan 15, 2017: Numpy 1.12
  Sep 13, 2015: Python 3.5
  Dec 23, 2016: Python 3.6
  Jun 27, 2018: Python 3.7
  Jun 07, 2017: Numpy 1.13
  Jan 06, 2018: Numpy 1.14
  Jul 23, 2018: Numpy 1.15
  Jan 13, 2019: Numpy 1.16
  Jul 26, 2019: Numpy 1.17
  Oct 14, 2019: Python 3.8
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

  minpy = '3.9+'
  minnum = '1.18+'

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
