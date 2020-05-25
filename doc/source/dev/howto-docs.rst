.. _howto-docs:

############################################
How to contribute to the NumPy documentation
############################################

This guide will help you decide what to contribute and how to submit it to the
official NumPy documentation.


**Documentation team meetings**

The NumPy community has set a firm goal of improving its documentation. Our
Documentation Team holds open meetings on Zoom every three weeks, and everyone
is welcome. Don't hesitate to reach out if you have questions or just need
someone to guide you through your first steps -- we're always happy to help.
Meetings are usually announced on the `numpy-discussion mailing list
<https://mail.python.org/mailman/listinfo/numpy-discussion>`__. Minutes are
taken `on hackmd.io <https://hackmd.io/oB_boakvRqKR-_2jRV-Qjg>`__ and stored
in the `NumPy Archive repository <https://github.com/numpy/archive>`__.

Overview
------------------------
:ref:`NumPy Documentation <numpy_docs_mainpage>` falls into two categories:

- **Hand-written:** Tutorials, How-Tos, Explanations. More on these categories below.

- **API Reference:** Generated automatically from the NumPy
  code `docstrings <https://www.python.org/dev/peps/pep-0257/>`__ when the
  NumPy documentation is :ref:`built from source<howto-build-docs>`.

First steps
--------------------

Fixes to technical inaccuracies take high priority -- a docstring missing a
parameter, a faulty description of a function/parameter/method, and so on.
Other "structural" defects like broken links also get priority. Problems
like these can be addressed by a `pull request (PR)
<https://numpy.org/devdocs/dev/index.html#devindex>`__ if you know how to do
one; otherwise please `open an issue <https://github.com/numpy/numpy/issues>`__.

Typos and misspellings fall on a lower rung; we welcome hearing about them but
may not be able to fix them promptly. These too can be handled as pull
requests or issues.

Obvious mistakes in wording (like leaving out a "not") fall in the typo
category, but other rewordings face a judgment call. Only if a change is
significant and compellingly better is it likely to go forward. Thus it's
best to lay it out first in an issue.

To propose major additions, such as a new tutorial, write to the `mailing list
<https://mail.python.org/mailman/listinfo/numpy-discussion>`__.

Our formal roadmap for documentation is a *NumPy Enhancement
Proposal (NEP)*,
`NEP 44 - Restructuring the NumPy Documentation
<https://www.numpy.org/neps/nep-0044-restructuring-numpy-docs>`__.
It identifies areas where our docs need help and lists several specific
additions we'd like to see, including Jupyter notebooks.

You can find larger planned and in-progress documentation improvement ideas `at
our GitHub project <https://github.com/orgs/numpy/projects/2>`__.

.. _tutorials_howtos_explanations:

Tutorials, how-to's, explanations -- what's the difference?
------------------------------------------------------------

On the battlefield of writing it's easy to forget that your goal is to
answer questions. Daily life has no one-size-fits-all answers ("I asked her
the time, and she told me how to build a watch"), and neither do projects. A
docs library, it turns out, needs to stock four sizes: ``tutorial``,
``how-to guide``, ``explanation``, and ``reference``. Daniele Procida has
written a `definitive, readable exposition
<https://documentation.divio.com/>`__ of how these differ and how to write
each type.


.. _contributing:

Contributing
------------

Don't worry if English is not your first language, or if you can only come up
with a rough draft. Open source is a community effort. Do your best -- we'll
help fix issues.

We encourage you to use real images and data (provided they are appropriately
licensed and available). This makes the material more engaging and can add
pedagogical value.

*Note: currently we cannot easily use data in other packages (except, e.g., from
SciPy or Matplotlib). We plan to create a dedicated datasets package, but that's
not ready yet - please discuss with us if you have data sources in mind.*

Documentation is stored in the NumPy source code tree, so to add your work to
the official documentation, you have to download the NumPy source code,
:ref:`build it <howto-build-docs>`, and submit your changes via a
:ref:`GitHub pull request <devindex>`.

If you are unfamiliar with git/GitHub or the process of submitting a pull
request (PR), check our :ref:`Contributor Guide <devindex>`.

The markup language for our docs is reStructuredText (rST), a more
comprehensive language than Markdown; the rST docs are then processed by Sphinx,
the tool most Python projects use to build and link project documentation. For
more on rST, see the `Quick reStructuredText Guide
<https://docutils.sourceforge.io/docs/user/rst/quickref.html>`__ or the
`reStructuredText Primer
<http://www.sphinx-doc.org/en/stable/usage/restructuredtext/basics.html>`__


Indirect ways to contribute
----------------------------

If you run across material that you think would be a useful addition,
`open an issue on GitHub
<https://github.com/numpy/numpy/issues>`__.

Similarly, if you can't find information you need on our site,
tell us!
`See this issue <https://github.com/numpy/numpy/issues/15760>`__
for an example.

Writing a tutorial on your blog, creating a YouTube video, or answering
questions on social media or Stack Overflow are also great contributions!


**Other interesting material**

- `writethedocs.org <https://www.writethedocs.org/>`__ has a lot of interesting
  resources for technical writing.
- Google offers two free `Technical Writing Courses
  <https://developers.google.com/tech-writing>`__
- `Software Carpentry <https://software-carpentry.org/software>`__ has a lot of
  nice recommendations for creating educational material.
