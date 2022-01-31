.. _NEP48:

=====================================
NEP 48 — Spending NumPy project funds
=====================================

:Author: Ralf Gommers <ralf.gommers@gmail.com>
:Author: Inessa Pawson <inessa@albuscode.org>
:Author: Stefan van der Walt <stefanv@berkeley.edu>
:Status: Draft
:Type: Informational
:Created: 2021-02-07
:Resolution:


Abstract
--------

The NumPy project has historically never received significant **unrestricted**
funding. However, that is starting to change.  This NEP aims to provide
guidance about spending NumPy project unrestricted funds by formulating a set
of principles about *what* to pay for and *who* to pay. It will also touch on
how decisions regarding spending funds get made, how funds get administered,
and transparency around these topics.


Motivation and Scope
--------------------

NumPy is a fiscally sponsored project of NumFOCUS, a 501(c)(3) nonprofit
organization headquartered in Austin, TX. Therefore, for all legal and
accounting matters the NumPy project has to follow the rules and regulations
for US nonprofits. All nonprofit donations are classified into two categories:
**unrestricted funds** which may be used for any legal purpose appropriate
to the organization and **restricted funds**, monies set aside for a
particular purpose (e.g., project, educational program, etc.).

For the detailed timeline of NumPy funding refer to
:ref:`numpy-funding-history`.

Since its inception and until 2020, the NumPy project has only spent on the order of
$10,000 USD of funds that were not restricted to a particular program.  Project
income of this type has been relying on donations from individuals and, from
mid 2019, recurring monthly contributions from Tidelift. By the end of 2020,
the Tidelift contributions increased to $3,000/month, and there's also a
potential for an increase of donations and grants going directly to the
project. Having a clear set of principles around how to use these funds will
facilitate spending them fairly and effectively. Additionally, it will make it
easier to solicit donations and other contributions.

A key assumption this NEP makes is that NumPy remains a largely
volunteer-driven project, and that the project funds are not enough to employ
maintainers full-time. If funding increases to the point where that assumption
is no longer true, this NEP should be updated.

In scope for this NEP are:

- Principles of spending project funds: what to pay for, and who to pay.
- Describing how NumPy's funds get administered.
- Describing how decisions to spend funds get proposed and made.

Out of scope for this NEP are:

- Making any decisions about spending project funds on a specific project or
  activity.
- Principles for spending funds that are intended for NumPy development, but
  don't fall in the category of NumPy unrestricted funds. This includes most of
  the grant funding, which is usually earmarked for certain
  activities/deliverables and goes to an Institutional Partner rather than
  directly to the NumPy project, and companies or institutions funding specific
  features.
  *Rationale: As a project, we have no direct control over how this work gets
  executed (at least formally, until issues or PRs show up). In some cases, we
  may not even know the contributions were funded or done by an employee on
  work time. (Whether that's the case or not should not change how we approach
  a contribution).  For grants though, we do expect the research/project leader
  and funded team to align their work with the needs of NumPy and be
  receptive to feedback from other NumPy maintainers and contributors.*


Principles of spending project funds
------------------------------------

NumPy will likely always be a project with many times more volunteer
contributors than funded people. Therefore having those funded people operate
in ways that attract more volunteers and enhance their participation experience
is critical. That key principle motivates many of the more detailed principles
given below for what to pay for and whom to pay.

The approach for spending funds will be:

- first figure out what we want to fund,
- then look for a great candidate,
- after that's settled, determine a fair compensation level.

The next sections go into detail on each of these three points.

.. _section-what-to-pay-for:

What to pay for
```````````````

1. Pay for things that are important *and* otherwise won't get done.
   *Rationale: there is way more to be done than there are funds to do all
   those things. So count on interested volunteers or external sponsored work
   to do many of those things.*
2. Plan for sustainability. Don't rely on money always being there.
3. Consider potential positive benefits for NumPy maintainers and contributors,
   maintainers of other projects, end users, and other stakeholders like
   packagers and educators.
4. Think broadly. There's more to a project than code: websites, documentation,
   community building, governance - it's all important.
5. For proposed funded work, include paid time for others to review your work
   if such review is expected to be significant effort - do not just increase
   the load on volunteer maintainers.
   *Rationale: we want the effect of spending funds to be positive for
   everyone, not just for the people getting paid. This is also a matter of
   fairness.*

When considering development work, principle (1) implies that priority should
be giving to (a) the most boring/painful tasks that no one likes doing, and to
necessary structural changes to the code base that are too large to be done by
a volunteer in a reasonable amount of time.

There are also many tasks, activities, and projects outside of
development work that are important and could enhance the project or community
- think of, for example, user surveys, translations, outreach, dedicated
mentoring of newcomers, community organizating, website improvements, and
administrative tasks.

Time of people to perform tasks is also not the only thing that funds can be
used for: expenses for in-person developer meetings or sprints, hosted hardware
for benchmarking or development work, and CI or other software services could
all be good candidates to spend funds on.

Whom to pay
```````````

1. All else being equal, give preference to existing maintainers/contributors.
2. When looking outside of the current team, consider this an opportunity to
   make the project more diverse.
3. Pay attention to the following when considering paying someone:

   - the necessary technical or domain-specific skills to execute the tasks,
   - communication and self-management skills,
   - experience contributing to and working with open source projects.

It will likely depend on the project/tasks whether there's already a clear best
candidate within the NumPy team, or whether we look for new people to get
involved. Before making any decisions, the decision makers (according to the
NumPy governance document - currently that's the Steering Council) should think
about whether an opportunity should be advertised to give a wider group of
people a chance to apply for it.

Compensating fairly
```````````````````

.. note::

   This section on compensating fairly will be considered *Draft* even if this
   NEP as a whole is accepted. Once we have applied the approach outlined here
   at least 2-3 times and we are happy with it, will we remove this note and
   consider this section *Accepted*.

Paying people fairly is a difficult topic, especially when it comes to
distributed teams. Therefore, we will only offer some guidance here. Final
decisions will always have to be considered and approved by the group of people
that bears this responsibility (according to the current NumPy governance
structure, this would be the NumPy Steering Council).

Discussions on remote employee compensation tend to be dominated by two
narratives: "pay local market rates" and "same work -- same pay".

We consider them both extreme:

- "Same work -- same pay" is unfair to people living in locations with a higher
  cost of living. For example, the average rent for a single family apartment
  can differ by a large factor (from a few hundred dollars to thousands of
  dollars per month).
- "Pay local market rates" bakes in existing inequalities between countries
  and makes fixed-cost items like a development machine or a holiday trip
  abroad relatively harder to afford in locations where market rates are lower.

We seek to find a middle ground between these two extremes.

Useful points of reference include companies like GitLab and
Buffer who are transparent about their remuneration policies ([3]_, [4]_),
Google Summer of Code stipends ([5]_), other open source projects that manage
their budget in a transparent manner (e.g., Babel and Webpack on Open
Collective ([6]_, [7]_)), and standard salary comparison sites.

Since NumPy is a not-for-profit project, we also looked to the nonprofit sector
for guidelines on remuneration policies and compensation levels. Our findings
show that most smaller non-profits tend to pay a median salary/wage. We
recognize merit in this approach: applying candidates are likely to have a
genuine interest in open source, rather than to be motivated purely by
financial incentives.

Considering all of the above, we will use the following guidelines for
determining compensation:

1. Aim to compensate people appropriately, up to a level that's expected for
   senior engineers or other professionals as applicable.
2. Establish a compensation cap of $125,000 USD that cannot be exceeded even
   for the residents from the most expensive/competitive locations ([#f-pay]_).
3. For equivalent work and seniority,  a pay differential between locations
   should never be more than 2x.
   For example, if we pay $110,000 USD to a senior-level developer from New
   York, for equivalent work a senior-level developer from South-East Asia
   should be paid at least $55,000 USD. To compare locations, we will use
   `Numbeo Cost of Living calculator <https://www.numbeo.com/cost-of-living/>`__
   (or its equivalent).

Some other considerations:

- Often, compensated work is offered for a limited amount of hours or fixed
  term. In those cases, consider compensation equivalent to a remuneration
  package that comes with permanent employment (e.g., one month of work should
  be compensated by at most 1/12th of a full-year salary + benefits).
- When comparing rates, an individual contractor should typically make 20% more
  than someone who is employed since they have to take care of their benefits
  and accounting on their own.
- Some people may be happy with one-off payments towards a particular
  deliverable (e.g., "triage all open issues for label X for $x,xxx").
  This should be compensated at a lower rate compared to an individual
  contractor. Or they may motivate lower amounts for another reason (e.g., "I
  want to receive $x,xxx to hire a cleaner or pay for childcare, to free up
  time for work on open source).
- When funding someone's time through their employer, that employer may want to
  set the compensation level based on its internal rules (e.g., overhead rates).
  Small deviations from the guidelines in this NEP may be needed in such cases,
  however they should be within reason.
- It's entirely possible that another strategy rather than paying people for
  their time on certain tasks may turn out to be more effective. Anything that
  helps the project and community grow and improve is worth considering.
- Transparency helps. If everyone involved is comfortable sharing their
  compensation levels with the rest of the team (or better make it public),
  it's least likely to be way off the mark for fairness.

We highly recommend that the individuals involved in decision-making about
hiring and compensation peruse the content of the References section of this
NEP. It offers a lot of helpful advice on this topic.


Defining fundable activities and projects
-----------------------------------------

We'd like to have a broader set of fundable ideas that we will prioritize with
input from NumPy team members and the wider community. All ideas will be
documented on a single wiki page. Anyone may propose an idea. Only members of a
NumPy team may edit the wiki page.

Each listed idea must meet the following requirements:

1. It must be clearly scoped: its description must explain the importance to
   the project, referencing the NumPy Roadmap if possible, the items to pay for
   or activities and deliverables, and why it should be a funded activity (see
   :ref:`section-what-to-pay-for`).
2. It must contain the following metadata: title, cost, time duration or effort
   estimate, and (if known) names of the team member(s) to execute or coordinate.
3. It must have an assigned priority (low, medium, or high). This discussion
   can originate at a NumPy community meeting or on the mailing list. However,
   it must be finalized on the mailing list allowing everyone to weigh in.

If a proposed idea has been assigned a high priority level, a decision on
allocating funding for it will be made on the private NumPy Steering Council
mailing list. *Rationale: these will often involve decisions about individuals,
which is typically hard to do in public. This is the current practice that
seems to be working well.*

Sometimes, it may be practical to make a single funding decision ad-hoc (e.g.,
"Here's a great opportunity plus the right person to execute it right now”).
However, this approach to decision-making should be used rarely.


Strategy for spending/saving funds
----------------------------------

There is an expectation from NumPy individual, corporate, and institutional
donors that the funds will be used for the benefit of the project and the
community. Therefore, we should spend available funds, thoughtfully,
strategically, and fairly, as they come in. For emergencies, we should keep a
$10,000 - $15,000 USD reserve which could cover, for example, a year of CI and
hosting services, 1-2 months of full-time maintenance work, or contracting a
consultant for a specific need.


How project funds get administered
----------------------------------

We will first summarize how administering of funds works today, and then
discuss how to make this process more efficient and transparent.

Currently, the project funds are held by NumFOCUS in a dedicated account.
NumFOCUS has a small accounting team, which produces an account overview as a
set of spreadsheets on a monthly basis. These land in a shared drive, typically
with about a one month delay (e.g., the balance and transactions for February
are available at the end of March), where a few NumPy team members can access
them. Expense claims and invoices are submitted through the NumFOCUS website.
Those then show up in another spreadsheet, where a NumPy team member must
review and approve each of them before payments are made. Following NumPy
bylaws, the NumFOCUS finance subcommittee, consisting of five people, meets
every six months to review all the project related transactions. (In practice,
there have been so few transactions that we skipped some of these meetings.)

The existing process is time-consuming and error-prone. More transparency and
automation are desirable.


Transparency about project funds and in decision making
```````````````````````````````````````````````````````

**To discuss: do we want full transparency by publishing our accounts,
transparency to everyone on a NumPy team, or some other level?**

Ralf: I'd personally like it to be fully transparent, like through Open
Collective, so the whole community can see current balance, income and expenses
paid out at any moment in time. Moving to Open Collective is nontrivial,
however we can publish the data elsewhere for now if we'd want to.
*Note: Google Season of Docs this year requires having an Open Collective
account, so this is likely to happen soon enough.*

Stefan/Inessa: at least a summary overview should be fully public, and all
transactions should be visible to the Steering Council. Full transparency of
all transactions is probably fine, but not necessary.

*The options here may be determined by the accounting system and amount of
effort required.*


.. _numpy-funding-history:

NumPy funding – history and current status
------------------------------------------

The NumPy project received its first major funding in 2017. For an overview of
the early history of NumPy (and SciPy), including some institutions sponsoring
time for their employees or contractors to work on NumPy, see [1]_ and [2]_. To
date, NumPy has received four grants:

- Two grants, from the Alfred P. Sloan Foundation and the Gordon and Betty
  Moore Foundation respectively, of about $1.3M combined to the Berkeley
  Institute of Data Science. Work performed during the period 2017-2020;
  PI Stéfan van der Walt.
- Two grants from the Chan Zuckerberg Foundation to NumFOCUS, for a combined
  amount of $335k. Work performed during the period 2020-2021; PI's Ralf
  Gommers (first grant) and Melissa Mendonça (second grant).

From 2012 onwards NumPy has been a fiscally sponsored project of NumFOCUS.
Note that fiscal sponsorship doesn't mean NumPy gets funding, rather that it
can receive funds under the umbrella of a nonprofit. See `NumFOCUS Project
Support <https://numfocus.org/projects-overview>`__ for more details.

Only since 2017 has the NumPy website displayed a "Donate" button, and since
2019 the NumPy repositories have had the GitHub Sponsors button. Before that,
it was possible to donate to NumPy on the NumFOCUS website. The sum total of
donations from individuals to NumPy for 2017-2020 was about $6,100.

From May 2019 onwards, Tidelift has supported NumPy financially as part of
its "managed open source" business model. From May 2019 till July 2020 this was
$1,000/month, and it started steadily growing after that to about $3,000/month
(as of Feb 2021).

Finally, there has been other incidental project income, for example, some book
royalties from Packt Publishing, GSoC mentoring fees from Google, and
merchandise sales revenue through the NumFOCUS web shop. All of these were
small (two or three figure) amounts.

This brings the total amount of project income which did not already have a
spending target to about $35,000. Most of that is recent, from Tidelift.
Over the past 1.5 years we spent about $10,000 for work on the new NumPy
website and Sphinx theme. Those spending decisions were made by the NumPy
Steering Council and announced on the mailing list.

That leaves about $25,000 in available funds at the time of writing, and
that amount is currently growing at a rate of about $3,000/month.


Related Work
------------

See references.  We assume that other open source projects have also developed
guidelines on spending project funds. However, we were unable to find any
examples at the time of writing.


Alternatives
------------

*Alternative spending strategy*: not having cash reserves. The rationale
being that NumPy is important enough that in a real emergency some person or
entity will likely jump in to help out. This is not a responsible approach to
financial stewardship of the project though. Hence, we decided against it.


Discussion
----------



References and Footnotes
------------------------

.. [1] Pauli Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific
       computing in Python", https://www.nature.com/articles/s41592-019-0686-2,
       2020

.. [2] Charles Harris et al., "Array programming with NumPy", https://www.nature.com/articles/s41586-020-2649-2, 2020

.. [3] https://remote.com/blog/remote-compensation

.. [4] https://about.gitlab.com/company/culture/all-remote/compensation/#how-do-you-decide-how-much-to-pay-people

.. [5] https://developers.google.com/open-source/gsoc/help/student-stipends

.. [6] Jurgen Appelo, "Compensation: what is fair?", https://blog.agilityscales.com/compensation-what-is-fair-38a65a822c29, 2016

.. [7] Project Include, "Compensating fairly", https://projectinclude.org/compensating_fairly

.. [#f-pay] This cap is derived from comparing with compensation levels at
            other open source projects (e.g., Babel, Webpack, Drupal - all in
            the $100,000 -- $125,000 range) and Partner Institutions.

- Nadia Eghbal, "Roads and Bridges: The Unseen Labor Behind Our Digital
  Infrastructure", 2016
- Nadia Eghbal, "Working in Public: The Making and Maintenance of Open
  Source", 2020
- https://github.com/nayafia/lemonade-stand
- Daniel Oberhaus, `"The Internet Was Built on the Free Labor of Open Source
  Developers. Is That Sustainable?"
  <https://www.vice.com/en/article/43zak3/the-internet-was-built-on-the-free-labor-of-open-source-developers-is-that-sustainable>`_, 2019
- David Heinemeier Hansson, `"The perils of mixing open source and money" <https://dhh.dk/2013/the-perils-of-mixing-open-source-and-money.html>`_, 2013
- Danny Crichton, `"Open source sustainability" <https://techcrunch.com/2018/06/23/open-source-sustainability/?guccounter=1>`_, 2018
- Nadia Eghbal, "Rebuilding the Cathedral", https://www.youtube.com/watch?v=VS6IpvTWwkQ, 2017
- Nadia Eghbal, "Where money meets open source", https://www.youtube.com/watch?v=bjAinwgvQqc&t=246s, 2017
- Eileen Uchitelle, ""The unbearable vulnerability of open source", https://www.youtube.com/watch?v=VdwO3LQ56oM, 2017 (the inverted triangle, open source is a funnel)
- Dries Buytaert, "Balancing Makers and Takers to scale and sustain Open Source", https://dri.es/balancing-makers-and-takers-to-scale-and-sustain-open-source, 2019
- Safia Abdalla, "Beyond Maintenance", https://increment.com/open-source/beyond-maintenance/, 2019
- Xavier Damman, "Money and Open Source Communities", https://blog.opencollective.com/money-and-open-source-communities/, 2016
- Aseem Sood, "Let's talk about money", https://blog.opencollective.com/lets-talk-about-money/, 2017
- Alanna Irving, "Has your open source community raised money? Here's how to spend it.", https://blog.opencollective.com/has-your-open-source-community-raised-money-heres-how-to-spend-it/, 2017
- Alanna Irving, "Funding open source, how Webpack reached $400k+/year", https://blog.opencollective.com/funding-open-source-how-webpack-reached-400k-year/, 2017
- Alanna Irving, "Babel's rise to financial sustainability", https://blog.opencollective.com/babels-rise-to-financial-sustainability/, 2019
- Devon Zuegel, "The city guide to open source", https://www.youtube.com/watch?v=80KTVu6GGSE, 2020 + blog: https://increment.com/open-source/the-city-guide-to-open-source/

GitHub Sponsors:

- https://github.blog/2019-05-23-announcing-github-sponsors-a-new-way-to-contribute-to-open-source/
- https://github.blog/2020-05-12-github-sponsors-is-out-of-beta-for-sponsored-organizations/
- https://blog.opencollective.com/on-github-sponsors/, 2019
- https://blog.opencollective.com/double-the-love/, 2020
- https://blog.opencollective.com/github-sponsors-for-companies-open-source-collective-for-people/


Copyright
---------

This document has been placed in the public domain.
