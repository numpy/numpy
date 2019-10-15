===================================
NEP 28 — numpy.org website redesign
===================================

:Author: Ralf Gommers <ralf.gommers@gmail.com>
:Author: Joe LaChance <joe@boldmetrics.com>
:Author: Shekhar Rajak <shekharrajak.1994@gmail.com>
:Status: Accepted
:Type: Informational
:Created: 2019-07-16
:Resolution: https://mail.python.org/pipermail/numpy-discussion/2019-August/079889.html


Abstract
--------

NumPy is the fundamental library for numerical and scientific computing with
Python. It is used by millions and has a large team of maintainers and
contributors. Despite that, its `numpy.org <http://numpy.org>`_ website has
never received the attention it needed and deserved. We hope and intend to
change that soon. This document describes ideas and requirements for how to
design a replacement for the current website, to better serve the needs of
our diverse community.

At a high level, what we're aiming for is:

- a modern, clean look
- an easy to deploy static site
- a structure that's easy to navigate
- content that addresses all types of stakeholders
- Possible multilingual translations / i18n

This website serves a couple of roles:

- it's the entry point to the project for new users
- it should link to the documentation (which is hosted separately, now on
  http://docs.scipy.org/ and in the near future on http://numpy.org/doc).
- it should address various aspects of the project (e.g. what NumPy is and
  why you'd want to use it, community, project organization, funding,
  relationship with NumFOCUS and possibly other organizations)
- it should link out to other places, so every type of stakeholder
  (beginning and advanced user, educators, packagers, funders, etc.)
  can find their way


Motivation and Scope
--------------------

The current numpy.org website has almost no content and its design is poor.
This affects many users, who come there looking for information. It also
affects many other aspects of the NumPy project, from finding new contributors
to fundraising.

The scope of the proposed redesign is the top-level numpy.org site, which
now contains only a couple of pages and may contain on the order of ten
pages after the redesign. Changing the documentation (user guide, reference
guide, and some other pages in the NumPy Manual) is out of scope for
this proposal.


Detailed description
--------------------

User Experience
~~~~~~~~~~~~~~~

Besides the NumPy logo, there is little that can or needs to be kept from the
current website. We will rely to a large extent on ideas and proposals by the
designer(s) of the new website.

As reference points we can use the `Jupyter website <https://jupyter.org/>`_,
which is probably the best designed site in our ecosystem, and the
`QuantEcon <https://quantecon.org>`_ and `Julia <https://julialang.org>`_
sites which are well-designed too.

The Website
~~~~~~~~~~~

A static site is a must. There are many high-quality static site generators.
The current website uses Sphinx, however that is not the best choice - it's
hard to theme and results in sites that are too text-heavy due to Sphinx'
primary aim being documentation.

The following should be considered when choosing a static site generator:

1. *How widely used is it?* This is important when looking for help maintaining
   or improving the site. More popular frameworks are usually also better
   maintained, so less chance of bugs or obsolescence.
2. *Ease of deployment.* Most generators meet this criterion, however things
   like built-in support for GitHub Pages helps.
3. *Preferences of who implements the new site.* Everyone has their own
   preferences. And it's a significant amount of work to build a new site.
   So we should take the opinion of those doing the work into account.

Traffic
```````

The current site receives on the order of 500,000 unique visitors per month.
With a redesigned site and relevant content, there is potential for visitor
counts to reach 5-6 million -- a similar level as
`scipy.org <http://scipy.org>`_ or `matplotlib.org <http://matplotlib.org>`_ --
or more.

Possible options for static site generators
```````````````````````````````````````````

1. *Jekyll.* This is a well maintained option with 855 Github contributors,
   with contributions within the last month. Jekyll is written in Ruby, and
   has a simple CLI interface. Jekyll also has a large directory of
   `themes <https://jekyllthemes.io>`__, although a majority cost money.
   There are several themes (`serif <https://jekyllthemes.io/theme/serif>`_,
   `uBuild <https://jekyllthemes.io/theme/ubuild-jekyll-theme>`_,
   `Just The Docs <https://jekyllthemes.io/theme/just-the-docs>`_) that are
   appropriate and free. Most themes are likely responsive for mobile, and
   that should be a requirement. Jekyll uses a combination of liquid templating
   and YAML to render HTML, and content is written in Markdown. i18n
   functionality is not native to Jekyll, but can be added easily.
   One nice benefit of Jekyll is that it can be run automatically by GitHub
   Pages, so deployment via a CI system doesn't need to be implemented.
2. *Hugo.* This is another well maintained option with 554 contributors, with
   contributions within the last month. Hugo is written in Go, and similar to
   Jekyll, has a simple to use CLI interface to generate static sites. Again,
   similar to Jekyll, Hugo has a large directory of
   `themes <https://themes.gohugo.io>`_. These themes appear to be free,
   unlike some of Jekyll's themes.
   (`Sample landing page theme <https://themes.gohugo.io/hugo-hero-theme>`_,
   `docs theme <https://themes.gohugo.io/hugo-whisper-theme>`_). Hugo uses Jade
   as its templating language, and content is also written in Markdown. i18n
   functionality is native to Hugo.
3. *Docusaurus.* Docusaurus is a responsive static site generator made by Facebook.
   Unlike the previous options, Docusaurus doesn't come with themes, and thus we
   would not want to use this for our landing page. This is an excellent docs
   option written in React. Docusaurus natively has support for i18n (via
   Crowdin_, document versioning, and document search.

Both Jekyll and Hugo are excellent options that should be supported into the
future and are good choices for NumPy. Docusaurus has several bonus features
such as versioning and search that Jekyll and Hugo don't have, but is likely
a poor candidate for a landing page - it could be a good option for a
high-level docs site later on though.

Deployment
~~~~~~~~~~

There is no need for running a server, and doing so is in our experience a
significant drain on the time of maintainers.

1. *Netlify.* Using netlify is free until 100GB of bandwidth is used. Additional
   bandwidth costs $20/100GB. They support a global CDN system, which will keep
   load times quick for users in other regions. Netlify also has Github integration,
   which will allow for easy deployment. When a pull request is merged, Netlify
   will automatically deploy the changes. DNS is simple, and HTTPS is also supported.
2. *Github Pages.* Github Pages also has a 100GB bandwidth limit, and is unclear if
   additional bandwidth can be purchased. It is also unclear where sites are deployed,
   and should be assumed sites aren't deployed globally. Github Pages has an easy to
   use CI & DNS, similar to to Netlify. HTTPS is supported.
3. *Cloudflare.* An excellent option, additional CI is likely needed for the same
   ease of deployment.

All of the above options are appropriate for the NumPy site based on current
traffic. Updating to a new deployment strategy, if needed, is a minor amount of
work compared to developing the website itself. If a provider such as
Cloudflare is chosen, additional CI may be required, such as CircleCI, to
have a similar deployment to GitHub Pages or Netlify.

Analytics
~~~~~~~~~

It's benefical to maintainers to know how many visitors are coming to
numpy.org. Google Analytics offers visitor counts and locations. This will
help to support and deploy more strategically, and help maintainers
understand where traffic is coming from.

Google Analytics is free. A script, provided by Google, must be added to the home page.

Website Structure
~~~~~~~~~~~~~~~~~

We aim to keep the first version of the new website small in terms of amount
of content. New pages can be added later on, it's more important right now to
get the site design right and get some essential information up. Note that in
the second half of 2019 we expect to get 1 or 2 tech writers involved in the
project via Google Season of Docs. They will likely help improve the content
and organization of that content.

We propose the following structure:

0. Front page: essentials of what NumPy is (compare e.g. jupyter.org), one or
   a couple key user stories (compare e.g. julialang.org)
1. Install
2. Documentation
3. Array computing
4. Community
5. Learning
6. About Us
7. Contribute
8. Donate

There may be a few other pages, e.g. a page on performance, that are linked
from one of the main pages.

Stakeholder Content
~~~~~~~~~~~~~~~~~~~

This should have as little content as possible *within the site*. Somewhere
on the site we should link out to content that's specific to:

- beginning users (quickstart, tutorial)
- advanced users
- educators
- packagers
- package authors that depend on NumPy
- funders (governance, roadmap)

Translation (multilingual / i18n)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NumPy has users all over the world. Most of those users are not native
English speakers, and many don't speak English well or at all. Therefore
having content in multiple languages is potentially addressing a large unmet
need. It would likely also help make the NumPy project more diverse and
welcoming.

On the other hand, there are good reasons why few projects have a
multi-lingual site. It's potentially a lot of extra work. Extra work for
maintainers is costly - they're already struggling to keep up with the work
load. Therefore we have to very carefully consider whether a multi-lingual
site is feasible and weight costs and benefits.

We start with an assertion: maintaining translations of all documentation, or
even the whole user guide, as part of the NumPy project is not feasible. One
simply has to look at the volume of our documentation and the frequency with
which we change it to realize that that's the case. Perhaps it will be
feasible though to translate just the top-level pages of the website. Those
do not change very often, and it will be a limited amount of content (order
of magnitude 5-10 pages of text).

We propose the following requirements for adding a language:

- The language must have a dedicated maintainer
- There must be a way to validate content changes (e.g. a second
  maintainer/reviewer, or high quality language support in a freely
  available machine translation tool)
- The language must have a reasonable size target audience (to be
  assessed by the NumPy maintainers)

Furthermore we propose a policy for when to remove support for a language again
(preferably by hiding it rather than deleting content). This may be done when
the language no longer has a maintainer, and coverage of translations falls
below an acceptable threshold (say 80%).

Benefits of having translations include:

- Better serve many existing and potential users
- Potentially attract a culturally and geographically more diverse set of contributors

The tradeoffs are:

- Cost of maintaining a more complex code base
- Cost of making decisions about whether or not to add a new language
- Higher cost to making content changes, creates work for language maintainers
- Any content change should be rolled out with enough delay to have translations in place

Can we define a small enough set of pages and content that it makes sense to do this?
Probably yes.

Is there an easy to use tool to maintain translations and add them to the website?
To be discussed - it needs investigating, and may depend on the choice of static site
generator. One potential option is Crowdin_, which is free for open source projects.


Style and graphic design
~~~~~~~~~~~~~~~~~~~~~~~~

Beyond the "a modern, clean look" goal we choose to not specify too much.  A
designer may have much better ideas than the authors of this proposal, hence we
will work with the designer(s) during the implementation phase.

The NumPy logo could use a touch-up.  The logo widely recognized and its colors and
design are good, however the look-and-feel is perhaps a little dated.


Other aspects
~~~~~~~~~~~~~

A search box would be nice to have.  The Sphinx documentation already has a
search box, however a search box on the main site which provides search results
for the docs, the website, and perhaps other domains that are relevant for
NumPy would make sense.


Backward compatibility
----------------------

Given a static site generator is chosen, we will migrate away from Sphinx for
numpy.org (the website, *not including the docs*). The current deployment can
be preserved until a future deprecation date is decided (potentially based on
the comfort level of our new site).

All site generators listed above have visibility into the HTML and Javascript
that is generated, and can continue to be maintained in the event a given
project ceases to be maintained.


Alternatives
------------

Alternatives we considered for the overall design of the website:

1. *Update current site.* A new Sphinx theme could be chosen. This would likely
   take the least amount of resources initially, however, Sphinx does not have
   the features we are looking for moving forward such as i18n, responsive design,
   and a clean, modern look.
   Note that updating the docs Sphinx theme is likely still a good idea - it's
   orthogonal to this NEP though.
2. *Create custom site.* This would take the most amount of resources, and is
   likely to have additional benefit in comparison to a static site generator.
   All features would be able to be added at the cost of developer time.


Discussion
----------

Mailing list thread discussing this NEP: TODO


References and Footnotes
------------------------
.. _Crowdin: https://crowdin.com/pricing#annual

Copyright
---------

This document has been placed in the public domain.
