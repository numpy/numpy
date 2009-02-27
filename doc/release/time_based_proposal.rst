.. vim:syntax=rst

Introduction
============

This document proposes some enhancements for numpy and scipy releases.
Successive numpy and scipy releases are too far apart from a time point of
view - some people who are in the numpy release team feel that it cannot
improve without a bit more formal release process. The main proposal is to
follow a time-based release, with expected dates for code freeze, beta and rc.
The goal is two folds: make release more predictable, and move the code forward.

Rationale
=========

Right now, the release process of numpy is relatively organic. When some
features are there, we may decide to make a new release. Because there is not
fixed schedule, people don't really know when new features and bug fixes will
go into a release. More significantly, having an expected release schedule
helps to *coordinate* efforts: at the beginning of a cycle, everybody can jump
in and put new code, even break things if needed. But after some point, only
bug fixes are accepted: this makes beta and RC releases much easier; calming
things down toward the release date helps focusing on bugs and regressions

Proposal
========

Time schedule
-------------

The proposed schedule is to release numpy every 9 weeks - the exact period can
be tweaked if it ends up not working as expected. There will be several stages
for the cycle:

        * Development: anything can happen (by anything, we mean as currently
          done). The focus is on new features, refactoring, etc...

        * Beta: no new features. No bug fixing which requires heavy changes.
          regression fixes which appear on supported platforms and were not
          caught earlier.

        * Polish/RC: only docstring changes and blocker regressions are allowed.

The schedule would be as follows:

        +------+-----------------+-----------------+------------------+
        | Week |     1.3.0       |      1.4.0      |  Release time    |
        +======+=================+=================+==================+
        |  1   |  Development    |                 |                  |
        +------+-----------------+-----------------+------------------+
        |  2   |  Development    |                 |                  |
        +------+-----------------+-----------------+------------------+
        |  3   |  Development    |                 |                  |
        +------+-----------------+-----------------+------------------+
        |  4   |  Development    |                 |                  |
        +------+-----------------+-----------------+------------------+
        |  5   |  Development    |                 |                  |
        +------+-----------------+-----------------+------------------+
        |  6   |  Development    |                 |                  |
        +------+-----------------+-----------------+------------------+
        |  7   |  Beta           |                 |                  |
        +------+-----------------+-----------------+------------------+
        |  8   |  Beta           |                 |                  |
        +------+-----------------+-----------------+------------------+
        |  9   |  Beta           |                 |  1.3.0 released  |
        +------+-----------------+-----------------+------------------+
        |  10  |  Polish         |   Development   |                  |
        +------+-----------------+-----------------+------------------+
        |  11  |  Polish         |   Development   |                  |
        +------+-----------------+-----------------+------------------+
        |  12  |  Polish         |   Development   |                  |
        +------+-----------------+-----------------+------------------+
        |  13  |  Polish         |   Development   |                  |
        +------+-----------------+-----------------+------------------+
        |  14  |                 |   Development   |                  |
        +------+-----------------+-----------------+------------------+
        |  15  |                 |   Development   |                  |
        +------+-----------------+-----------------+------------------+
        |  16  |                 |   Beta          |                  |
        +------+-----------------+-----------------+------------------+
        |  17  |                 |   Beta          |                  |
        +------+-----------------+-----------------+------------------+
        |  18  |                 |   Beta          |  1.4.0 released  |
        +------+-----------------+-----------------+------------------+

Each stage can be defined as follows:

        +------------------+-------------+----------------+----------------+
        |                  | Development |      Beta      |    Polish      |
        +==================+=============+================+================+
        | Python Frozen    |             |     slushy     |       Y        |
        +------------------+-------------+----------------+----------------+
        | Docstring Frozen |             |     slushy     |  thicker slush |
        +------------------+-------------+----------------+----------------+
        | C code Frozen    |             | thicker slush  |  thicker slush |
        +------------------+-------------+----------------+----------------+

Terminology:

        * slushy: you can change it if you beg the release team and it's really
          important and you coordinate with docs/translations; no "big"
          changes.

        * thicker slush: you can change it if it's an open bug marked
          showstopper for the Polish release, you beg the release team, the
          change is very very small yet very very important, and you feel
          extremely guilty about your transgressions.

The different frozen states are intended to be gradients. The exact meaning is
decided by the release manager: he has the last word on what's go in, what
doesn't.  The proposed schedule means that there would be at most 12 weeks
between putting code into the source code repository and being released.

Release team
------------

For every release, there would be at least one release manager. We propose to
rotate the release manager: rotation means it is not always the same person
doing the dirty job, and it should also keep the release manager honest.

References
==========

        * Proposed schedule for Gnome from Havoc Pennington (one of the core
          GTK and Gnome manager):
          http://mail.gnome.org/archives/gnome-hackers/2002-June/msg00041.html
          The proposed schedule is heavily based on this email

        * http://live.gnome.org/ReleasePlanning/Freezes
