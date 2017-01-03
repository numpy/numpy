================================================================
  NumPy project governance and decision-making
================================================================

The purpose of this document is to formalize the governance process
used by the NumPy project in both ordinary and extraordinary
situations, and to clarify how decisions are made and how the various
elements of our community interact, including the relationship between
open source collaborative development and work that may be funded by
for-profit or non-profit entities.

Summary
=======

NumPy is a community-owned and community-run project. To the maximum
extent possible, decisions about project direction are made by community
consensus (but note that "consensus" here has a somewhat technical
meaning that might not match everyone's expectations -- see below). Some
members of the community additionally contribute by serving on the NumPy
steering council, where they are responsible for facilitating the
establishment of community consensus, for stewarding project resources,
and -- in extreme cases -- for making project decisions if the normal
community-based process breaks down.

The Project
===========

The NumPy Project (The Project) is an open source software project
affiliated with the 501(c)3 NumFOCUS Foundation. The goal of The Project
is to develop open source software for array-based computing in Python,
and in particular the ``numpy`` package, along with related software
such as ``f2py`` and the NumPy Sphinx extensions. The Software developed
by The Project is released under the BSD (or similar) open source
license, developed openly and hosted on public GitHub repositories under
the ``numpy`` GitHub organization.

The Project is developed by a team of distributed developers, called
Contributors. Contributors are individuals who have contributed code,
documentation, designs or other work to the Project. Anyone can be a
Contributor. Contributors can be affiliated with any legal entity or
none. Contributors participate in the project by submitting, reviewing
and discussing GitHub Pull Requests and Issues and participating in open
and public Project discussions on GitHub, mailing lists, and other
channels. The foundation of Project participation is openness and
transparency.

The Project Community consists of all Contributors and Users of the
Project. Contributors work on behalf of and are responsible to the
larger Project Community and we strive to keep the barrier between
Contributors and Users as low as possible.

The Project is formally affiliated with the 501(c)3 NumFOCUS Foundation
(http://numfocus.org), which serves as its fiscal sponsor, may hold
project trademarks and other intellectual property, helps manage project
donations and acts as a parent legal entity. NumFOCUS is the only legal
entity that has a formal relationship with the project (see
Institutional Partners section below).

Governance
==========

This section describes the governance and leadership model of The
Project.

The foundations of Project governance are:

-  Openness & Transparency
-  Active Contribution
-  Institutional Neutrality

Consensus-based decision making by the community
------------------------------------------------

Normally, all project decisions will be made by consensus of all
interested Contributors. The primary goal of this approach is to ensure
that the people who are most affected by and involved in any given
change can contribute their knowledge in the confidence that their
voices will be heard, because thoughtful review from a broad community
is the best mechanism we know of for creating high-quality software.

The mechanism we use to accomplish this goal may be unfamiliar for those
who are not experienced with the cultural norms around free/open-source
software development. We provide a summary here, and highly recommend
that all Contributors additionally read `Chapter 4: Social and Political
Infrastructure <http://producingoss.com/en/producingoss.html#social-infrastructure>`_
of Karl Fogel's classic *Producing Open Source Software*, and in
particular the section on `Consensus-based
Democracy <http://producingoss.com/en/producingoss.html#consensus-democracy>`_,
for a more detailed discussion.

In this context, consensus does *not* require:

-  that we wait to solicit everybody's opinion on every change,
-  that we ever hold a vote on anything,
-  or that everybody is happy or agrees with every decision.

For us, what consensus means is that we entrust *everyone* with the
right to veto any change if they feel it necessary. While this may sound
like a recipe for obstruction and pain, this is not what happens.
Instead, we find that most people take this responsibility seriously,
and only invoke their veto when they judge that a serious problem is
being ignored, and that their veto is necessary to protect the project.
And in practice, it turns out that such vetoes are almost never formally
invoked, because their mere possibility ensures that Contributors are
motivated from the start to find some solution that everyone can live
with -- thus accomplishing our goal of ensuring that all interested
perspectives are taken into account.

How do we know when consensus has been achieved? In principle, this is
rather difficult, since consensus is defined by the absence of vetos,
which requires us to somehow prove a negative. In practice, we use a
combination of our best judgement (e.g., a simple and uncontroversial
bug fix posted on GitHub and reviewed by a core developer is probably
fine) and best efforts (e.g., all substantive API changes must be posted
to the mailing list in order to give the broader community a chance to
catch any problems and suggest improvements; we assume that anyone who
cares enough about NumPy to invoke their veto right should be on the
mailing list). If no-one bothers to comment on the mailing list after a
few days, then it's probably fine. And worst case, if a change is more
controversial than expected, or a crucial critique is delayed because
someone was on vacation, then it's no big deal: we apologize for
misjudging the situation, `back up, and sort things
out <http://producingoss.com/en/producingoss.html#version-control-relaxation>`_.

If one does need to invoke a formal veto, then it should consist of:

-  an unambiguous statement that a veto is being invoked,
-  an explanation of why it is being invoked, and
-  a description of what conditions (if any) would convince the vetoer
   to withdraw their veto.

If all proposals for resolving some issue are vetoed, then the status
quo wins by default.

In the worst case, if a Contributor is genuinely misusing their veto in
an obstructive fashion to the detriment of the project, then they can be
ejected from the project by consensus of the Steering Council -- see
below.

Steering Council
----------------

The Project will have a Steering Council that consists of Project
Contributors who have produced contributions that are substantial in
quality and quantity, and sustained over at least one year. The overall
role of the Council is to ensure, with input from the Community, the
long-term well-being of the project, both technically and as a
community.

During the everyday project activities, council members participate in
all discussions, code review and other project activities as peers with
all other Contributors and the Community. In these everyday activities,
Council Members do not have any special power or privilege through their
membership on the Council. However, it is expected that because of the
quality and quantity of their contributions and their expert knowledge
of the Project Software and Services that Council Members will provide
useful guidance, both technical and in terms of project direction, to
potentially less experienced contributors.

The Steering Council and its Members play a special role in certain
situations. In particular, the Council may, if necessary:

-  Make decisions about the overall scope, vision and direction of the
   project.
-  Make decisions about strategic collaborations with other
   organizations or individuals.
-  Make decisions about specific technical issues, features, bugs and
   pull requests. They are the primary mechanism of guiding the code
   review process and merging pull requests.
-  Make decisions about the Services that are run by The Project and
   manage those Services for the benefit of the Project and Community.
-  Update policy documents such as this one.
-  Make decisions when regular community discussion doesn’t produce
   consensus on an issue in a reasonable time frame.

However, the Council's primary responsibility is to facilitate the
ordinary community-based decision making procedure described above. If
we ever have to step in and formally override the community for the
health of the Project, then we will do so, but we will consider reaching
this point to indicate a failure in our leadership.

Council decision making
~~~~~~~~~~~~~~~~~~~~~~~

If it becomes necessary for the Steering Council to produce a formal
decision, then they will use a form of the `Apache Foundation voting
process <https://www.apache.org/foundation/voting.html>`_. This is a
formalized version of consensus, in which +1 votes indicate agreement,
-1 votes are vetoes (and must be accompanied with a rationale, as
above), and one can also vote fractionally (e.g. -0.5, +0.5) if one
wishes to express an opinion without registering a full veto. These
numeric votes are also often used informally as a way of getting a
general sense of people's feelings on some issue, and should not
normally be taken as formal votes. A formal vote only occurs if
explicitly declared, and if this does occur then the vote should be held
open for long enough to give all interested Council Members a chance to
respond -- at least one week.

In practice, we anticipate that for most Steering Council decisions
(e.g., voting in new members) a more informal process will suffice.

Council membership
~~~~~~~~~~~~~~~~~~

A list of current Steering Council Members is maintained at the
page :ref:`governance-people`.

To become eligible to join the Steering Council, an individual must be
a Project Contributor who has produced contributions that are
substantial in quality and quantity, and sustained over at least one
year. Potential Council Members are nominated by existing Council
members, and become members following consensus of the existing
Council members, and confirmation that the potential Member is
interested and willing to serve in that capacity. The Council will be
initially formed from the set of existing Core Developers who, as of
late 2015, have been significantly active over the last year.

When considering potential Members, the Council will look at candidates
with a comprehensive view of their contributions. This will include but
is not limited to code, code review, infrastructure work, mailing list
and chat participation, community help/building, education and outreach,
design work, etc. We are deliberately not setting arbitrary quantitative
metrics (like “100 commits in this repo”) to avoid encouraging behavior
that plays to the metrics rather than the project’s overall well-being.
We want to encourage a diverse array of backgrounds, viewpoints and
talents in our team, which is why we explicitly do not define code as
the sole metric on which council membership will be evaluated.

If a Council member becomes inactive in the project for a period of one
year, they will be considered for removal from the Council. Before
removal, inactive Member will be approached to see if they plan on
returning to active participation. If not they will be removed
immediately upon a Council vote. If they plan on returning to active
participation soon, they will be given a grace period of one year. If
they don’t return to active participation within that time period they
will be removed by vote of the Council without further grace period. All
former Council members can be considered for membership again at any
time in the future, like any other Project Contributor. Retired Council
members will be listed on the project website, acknowledging the period
during which they were active in the Council.

The Council reserves the right to eject current Members, if they are
deemed to be actively harmful to the project’s well-being, and attempts
at communication and conflict resolution have failed. This requires the
consensus of the remaining Members.


Conflict of interest
~~~~~~~~~~~~~~~~~~~~

It is expected that the Council Members will be employed at a wide range
of companies, universities and non-profit organizations. Because of
this, it is possible that Members will have conflict of interests. Such
conflict of interests include, but are not limited to:

-  Financial interests, such as investments, employment or contracting
   work, outside of The Project that may influence their work on The
   Project.
-  Access to proprietary information of their employer that could
   potentially leak into their work with the Project.

All members of the Council shall disclose to the rest of the Council any
conflict of interest they may have. Members with a conflict of interest
in a particular issue may participate in Council discussions on that
issue, but must recuse themselves from voting on the issue.

Private communications of the Council
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To the maximum extent possible, Council discussions and activities
will be public and done in collaboration and discussion with the
Project Contributors and Community. The Council will have a private
mailing list that will be used sparingly and only when a specific
matter requires privacy. When private communications and decisions are
needed, the Council will do its best to summarize those to the
Community after eliding personal/private/sensitive information that
should not be posted to the public internet.

Subcommittees
~~~~~~~~~~~~~

The Council can create subcommittees that provide leadership and
guidance for specific aspects of the project. Like the Council as a
whole, subcommittees should conduct their business in an open and public
manner unless privacy is specifically called for. Private subcommittee
communications should happen on the main private mailing list of the
Council unless specifically called for.

NumFOCUS Subcommittee
~~~~~~~~~~~~~~~~~~~~~

The Council will maintain one narrowly focused subcommittee to manage
its interactions with NumFOCUS.

-  The NumFOCUS Subcommittee is comprised of 5 persons who manage
   project funding that comes through NumFOCUS. It is expected that
   these funds will be spent in a manner that is consistent with the
   non-profit mission of NumFOCUS and the direction of the Project as
   determined by the full Council.
-  This Subcommittee shall NOT make decisions about the direction, scope
   or technical direction of the Project.
-  This Subcommittee will have 5 members, 4 of whom will be current
   Council Members and 1 of whom will be external to the Steering
   Council. No more than 2 Subcommitee Members can report to one person
   through employment or contracting work (including the reportee, i.e.
   the reportee + 1 is the max). This avoids effective majorities
   resting on one person.

The current membership of the NumFOCUS Subcommittee is listed at the
page :ref:`governance-people`.


Institutional Partners and Funding
==================================

The Steering Council are the primary leadership for the project. No
outside institution, individual or legal entity has the ability to own,
control, usurp or influence the project other than by participating in
the Project as Contributors and Council Members. However, because
institutions can be an important funding mechanism for the project, it
is important to formally acknowledge institutional participation in the
project. These are Institutional Partners.

An Institutional Contributor is any individual Project Contributor who
contributes to the project as part of their official duties at an
Institutional Partner. Likewise, an Institutional Council Member is any
Project Steering Council Member who contributes to the project as part
of their official duties at an Institutional Partner.

With these definitions, an Institutional Partner is any recognized legal
entity in the United States or elsewhere that employs at least 1
Institutional Contributor of Institutional Council Member. Institutional
Partners can be for-profit or non-profit entities.

Institutions become eligible to become an Institutional Partner by
employing individuals who actively contribute to The Project as part of
their official duties. To state this another way, the only way for a
Partner to influence the project is by actively contributing to the open
development of the project, in equal terms to any other member of the
community of Contributors and Council Members. Merely using Project
Software in institutional context does not allow an entity to become an
Institutional Partner. Financial gifts do not enable an entity to become
an Institutional Partner. Once an institution becomes eligible for
Institutional Partnership, the Steering Council must nominate and
approve the Partnership.

If at some point an existing Institutional Partner stops having any
contributing employees, then a one year grace period commences. If at
the end of this one year period they continue not to have any
contributing employees, then their Institutional Partnership will
lapse, and resuming it will require going through the normal process
for new Partnerships.

An Institutional Partner is free to pursue funding for their work on The
Project through any legal means. This could involve a non-profit
organization raising money from private foundations and donors or a
for-profit company building proprietary products and services that
leverage Project Software and Services. Funding acquired by
Institutional Partners to work on The Project is called Institutional
Funding. However, no funding obtained by an Institutional Partner can
override the Steering Council. If a Partner has funding to do NumPy work
and the Council decides to not pursue that work as a project, the
Partner is free to pursue it on their own. However in this situation,
that part of the Partner’s work will not be under the NumPy umbrella and
cannot use the Project trademarks in a way that suggests a formal
relationship.

Institutional Partner benefits are:

-  Acknowledgement on the NumPy websites, in talks and T-shirts.
-  Ability to acknowledge their own funding sources on the NumPy
   websites, in talks and T-shirts.
-  Ability to influence the project through the participation of their
   Council Member.
-  Council Members invited to NumPy Developer Meetings.

A list of current Institutional Partners is maintained at the page
:ref:`governance-people`.


Document history
================

https://github.com/numpy/numpy/commits/master/doc/source/dev/governance/governance.rst

Acknowledgements
================

Substantial portions of this document were adapted from the
`Jupyter/IPython project's governance document
<https://github.com/jupyter/governance/blob/master/governance.md>`_.

License
=======

To the extent possible under law, the authors have waived all
copyright and related or neighboring rights to the NumPy project
governance and decision-making document, as per the `CC-0 public
domain dedication / license
<https://creativecommons.org/publicdomain/zero/1.0/>`_.
