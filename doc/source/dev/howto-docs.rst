.. _howto-docs:

############################################
How to contribute to the NumPy documentation
############################################

This guide will help you decide what to contribute and how to submit it to the
official NumPy documentation.

***************************
Documentation team meetings
***************************

The NumPy community has set a firm goal of improving its documentation. We
hold regular documentation meetings on Zoom (dates are announced on the
`numpy-discussion mailing list
<https://mail.python.org/mailman/listinfo/numpy-discussion>`__), and everyone
is welcome. Reach out if you have questions or need
someone to guide you through your first steps -- we're happy to help.
Minutes are taken `on hackmd.io <https://hackmd.io/oB_boakvRqKR-_2jRV-Qjg>`__
and stored in the `NumPy Archive repository
<https://github.com/numpy/archive>`__.

*************
What's needed
*************

The :ref:`NumPy Documentation <numpy_docs_mainpage>` has the details covered.
API reference documentation is generated directly from
`docstrings <https://www.python.org/dev/peps/pep-0257/>`_ in the code when the
documentation is :ref:`built<howto-build-docs>`. Although we have mostly
complete reference documentation for each function and class exposed to users,
there is a lack of usage examples for some of them.

What we lack are docs with broader scope -- tutorials, how-tos, and
explanations. Reporting defects is another way to contribute. We discuss both.

******************
Contributing fixes
******************

We're eager to hear about and fix doc defects. But to attack the biggest
problems we end up having to defer or overlook some bug reports. Here are the
best defects to go after.

Top priority goes to **technical inaccuracies** -- a docstring missing a
parameter, a faulty description of a function/parameter/method, and so on.
Other "structural" defects like broken links also get priority. All these fixes
are easy to confirm and put in place. You can submit
a `pull request (PR) <https://numpy.org/devdocs/dev/index.html#devindex>`__
with the fix, if you know how to do that; otherwise please `open an issue
<https://github.com/numpy/numpy/issues>`__.

**Typos and misspellings** fall on a lower rung; we welcome hearing about them
but may not be able to fix them promptly. These too can be handled as pull
requests or issues.

Obvious **wording** mistakes (like leaving out a "not") fall into the typo
category, but other rewordings -- even for grammar -- require a judgment call,
which raises the bar. Test the waters by first presenting the fix as an issue.

**********************
Contributing new pages
**********************

Your frustrations using our documents are our best guide to what needs fixing.

If you write a missing doc you join the front line of open source, but it's
a meaningful contribution just to let us know what's missing. If you want to
compose a doc, run your thoughts by the `mailing list
<https://mail.python.org/mailman/listinfo/numpy-discussion>`__ for futher
ideas and feedback. If you want to alert us to a gap,
`open an issue <https://github.com/numpy/numpy/issues>`__. See
`this issue <https://github.com/numpy/numpy/issues/15760>`__ for an example.

If you're looking for subjects, our formal roadmap for documentation is a
*NumPy Enhancement Proposal (NEP)*,
`NEP 44 - Restructuring the NumPy Documentation <https://www.numpy.org/neps/nep-0044-restructuring-numpy-docs>`__.
It identifies areas where our docs need help and lists several
additions we'd like to see, including :ref:`Jupyter notebooks <numpy_tutorials>`.

.. _tutorials_howtos_explanations:

Documentation framework
=======================

There are formulas for writing useful documents, and four formulas
cover nearly everything. There are four formulas because there are four
categories of document -- ``tutorial``, ``how-to guide``, ``explanation``,
and ``reference``. The insight that docs divide up this way belongs to
Daniele Procida and his `Diátaxis Framework <https://diataxis.fr/>`__. When you
begin a document or propose one, have in mind which of these types it will be.

.. _numpy_tutorials:

NumPy tutorials
===============

In addition to the documentation that is part of the NumPy source tree, you can
submit content in Jupyter Notebook format to the
`NumPy Tutorials <https://numpy.org/numpy-tutorials>`__ page. This
set of tutorials and educational materials is meant to provide high-quality
resources by the NumPy project, both for self-learning and for teaching classes
with. These resources are developed in a separate GitHub repository,
`numpy-tutorials <https://github.com/numpy/numpy-tutorials>`__, where you can
check out existing notebooks, open issues to suggest new topics or submit your
own tutorials as pull requests.

.. _contributing:

More on contributing
====================

Don't worry if English is not your first language, or if you can only come up
with a rough draft. Open source is a community effort. Do your best -- we'll
help fix issues.

Images and real-life data make text more engaging and powerful, but be sure
what you use is appropriately licensed and available. Here again, even a rough
idea for artwork can be polished by others.

For now, the only data formats accepted by NumPy are those also used by other
Python scientific libraries like pandas, SciPy, or Matplotlib. We're
developing a package to accept more formats; contact us for details.

NumPy documentation is kept in the source code tree. To get your document
into the docbase you must download the tree, :ref:`build it
<howto-build-docs>`, and submit a pull request. If GitHub and pull requests
are new to you, check our :ref:`Contributor Guide <devindex>`.

Our markup language is reStructuredText (rST), which is more elaborate than
Markdown. Sphinx, the tool many Python projects use to build and link project
documentation, converts the rST into HTML and other formats. For more on
rST, see the `Quick reStructuredText Guide
<https://docutils.sourceforge.io/docs/user/rst/quickref.html>`__ or the
`reStructuredText Primer
<http://www.sphinx-doc.org/en/stable/usage/restructuredtext/basics.html>`__


***********************
Contributing indirectly
***********************

If you run across outside material that would be a useful addition to the
NumPy docs, let us know by `opening an issue <https://github.com/numpy/numpy/issues>`__.

You don't have to contribute here to contribute to NumPy. You've contributed
if you write a tutorial on your blog, create a YouTube video, or answer questions
on Stack Overflow and other sites.


*********************
Documentation reading
*********************

- The leading organization of technical writers,
  `Write the Docs <https://www.writethedocs.org/>`__,
  holds conferences, hosts learning resources, and runs a Slack channel.

- "Every engineer is also a writer," says Google's
  `collection of technical writing resources <https://developers.google.com/tech-writing>`__,
  which includes free online courses for developers in planning and writing
  documents.

- `Software Carpentry's <https://software-carpentry.org/lessons>`__ mission is
  teaching software to researchers. In addition to hosting the curriculum, the
  website explains how to present ideas effectively.
