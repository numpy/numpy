===================================================
NEP 6 — Replacing Trac with a different bug tracker
===================================================

:Author: David Cournapeau, Stefan van der Walt
:Status: Deferred

Some release managers of both numpy and scipy are becoming more and more
dissatisfied with the current development workflow, in particular for bug
tracking. This document is a tentative to explain some problematic scenario,
current trac limitations, and what can be done about it.

Scenario
========

new release
-----------

The workflow for a release is roughly as follows:

	* find all known regressions from last release, and fix them

        * get an idea of all bugs reported since last release

        * triage bugs in regressions/blocker issues/etc..., and assign them in
          the according roadmap, subpackage and maintainers

	* pinging subpackage maintainers

Most of those tasks are quite inefficient in the current trac as used on scipy:

        * it is hard to keep track of issues. In particular, every time one goes
          to trac, we don't really know what's new from what's not. If you
          think of issues as emails, the current situation would be like not
          having read/unread feature.

        * Batch handling of issues: changing characteristics of several issues
          at the same time is difficult, because the only available UI is
          web-based. Command-line based UI are much more efficient for this
          kind of scenario

More generally, making useful reports is very awkward with the currently
deployed trac. Trac 0.11 may solve of those problems, but it has to be much
better than the actually deployed version on scipy website. Finding issues with
patches, old patches, etc... and making reports has to be much more streamlined
that it is now.

subcomponent maintainer
-----------------------

Say you are the maintainer of scipy.foo, then you are mostly interested in
getting bugs concerning scipy.foo only. But it should be easy for the general
team to follow your work - it should also be easy for casual users (e.g. not
developers) to follow some new features development pace.

Review, newcoming code
----------------------

The goal is simple: make the bar as low as possible, and make sure people know
what to do at every step to contribute to numpy or scipy:

        * Right now, patches languish for too long in trac. Of course, lack of
          time is one big reason; but the process of following new contributes
          could be made much simpler

        * It should be possible to be pinged only for reviews one a subset of
          numpy/scipy.

        * It should be possible for people interested in the patches to follow
          its progression. Comments, but also 'mini' timelines could be useful,
          particularly for massive issues (massive from a coding POV).

Current trac limitation
=======================

Note: by trac, we mean the currently deployed one. Some more recent versions
may solve some of the issues.

        * Multi-project support: we have three trac instances, one for scipy,
          one for numpy, one for scikits. Creating accounts, maintaining and
          updating each of them is a maintenance burden. Nobody likes to do
          this kind of work, so anything which can reduce the burden is a plus.
          Also, it happens quite frequently that a bug against numpy is filled
          on scipy trac and vice and versa. You have to handle this manually,
          currently.

        * Clients not based on the web-ui. This can be made through the xmlrpc
          plugin + some clients. In particular, something like
          http://tracexplorer.devjavu.com/ can be interesting for people who
          like IDE. At least one person expressed his desire to have as much
          integration as possible with Eclipse.

        * Powerful queries: it should be possible to quickly find issues
          between two releases, the new issues from a given date, issues with
          patch, issues waiting for reviews, etc... The issues data have to be
          customizable, because most bug-tracker do not support things like
          review, etc... so we need to handle this ourselves (through tags,
          etc...)

        * Marking issues as read/unread. It should also be possible for any
          user to 'mask' issues to ignore them.

        * ticket dependency. This is quite helpful in my experience for big
          features which can be split into several issues. Roadmap can only be
          created by trac admin, and they are kind of heavy-weight.

Possible candidates
===================

Updated trac + plugins
----------------------

Pros:

        * Same system

        * In python, so we can hack it if we want

Cons:

        * Trac is aimed at being basic, and extended with plugins. But most
          plugins are broken, or not up to date. The information on which
          plugins are mature is not easily available.

        * At least the scipy.org trac was slow, and needed to be restarted
          constantly. This is simply not acceptable.

Redmine
-------

Pros:

        * Support most features (except xmlrpc ?). Multi-project, etc...

        * (subjective): I (cdavid) find the out-of-the-box experience with
          redmine much more enjoyable. More information is available easily,
          less clicks, more streamlined. See
          http://www.redmine.org/wiki/redmine/TheyAreUsingRedmine for examples

        * Conversion scripts from trac (no experience with it yet for numpy/scipy).

        * Community seems friendly and gets a lof of features done

Cons:

        * new system, less mature ?

        * in Ruby: since we are a python project, most of dev are familiar with
          python.

        * Wiki integration, etc... ?

Unknown:

        * xmlrpc API
        * performances
        * maintenance cost

Roundup
-------

TODO
