===================================================
NEP 44 — Restructuring the NumPy Documentation
===================================================

:Author: Ralf Gommers
:Author: Melissa Mendonça
:Author: Mars Lee
:Status: Accepted
:Type: Process
:Created: 2020-02-11
:Resolution: https://mail.python.org/pipermail/numpy-discussion/2020-March/080467.html

Abstract
========

This document proposes a restructuring of the NumPy Documentation, both in form
and content, with the goal of making it more organized and discoverable for
beginners and experienced users.

Motivation and Scope
====================

See `here <https://numpy.org/devdocs/>`_ for the front page of the latest docs.
The organization is quite confusing and illogical (e.g. user and developer docs
are mixed). We propose the following:

- Reorganizing the docs into the four categories mentioned in [1]_, namely *Tutorials*, *How Tos*, *Reference Guide* and *Explanations* (more about this below).
- Creating dedicated sections for Tutorials and How-Tos, including orientation
  on how to create new content;
- Adding an Explanations section for key concepts and techniques that require
  deeper descriptions, some of which will be rearranged from the Reference Guide.

Usage and Impact
================

The documentation is a fundamental part of any software project, especially
open source projects. In the case of NumPy, many beginners might feel demotivated
by the current structure of the documentation, since it is difficult to discover
what to learn (unless the user has a clear view of what to look for in the
Reference docs, which is not always the case).

Looking at the results of a "NumPy Tutorial" search on any search engine also
gives an idea of the demand for this kind of content. Having official high-level
documentation written using up-to-date content and techniques will certainly
mean more users (and developers/contributors) are involved in the NumPy
community.

Backward compatibility
======================

The restructuring will effectively demand a complete rewrite of links and some
of the current content. Input from the community will be useful for identifying
key links and pages that should not be broken.

Detailed description
====================

As discussed in the article [1]_, there are four categories of doc content:

- Tutorials
- How-to guides
- Explanations
- Reference guide

We propose to use those categories as the ones we use (for writing and
reviewing) whenever we add a new documentation section.

The reasoning for this is that it is clearer both for
developers/documentation writers and to users where each piece of
information should go, and the scope and tone of each document. For
example, if explanations are mixed with basic tutorials, beginners
might be overwhelmed and alienated. On the other hand, if the reference
guide contains basic how-tos, it might be difficult for experienced
users to find the information they need, quickly.

Currently, there are many blogs and tutorials on the internet about NumPy or
using NumPy. One of the issues with this is that if users search for this
information they may end up in an outdated (unofficial) tutorial before
they find the current official documentation. This can be especially
confusing, especially for beginners. Having a better infrastructure for the
documentation also aims to solve this problem by giving users high-level,
up-to-date official documentation that can be easily updated.

Status and ideas of each type of doc content
--------------------------------------------

**Reference guide**

NumPy has a quite complete reference guide. All functions are documented, most
have examples, and most are cross-linked well with *See Also* sections. Further
improving the reference guide is incremental work that can be done (and is being
done) by many people. There are, however, many explanations in the reference
guide. These can be moved to a more dedicated Explanations section on the docs.

**How-to guides**

NumPy does not have many how-to's. The subclassing and array ducktyping section
may be an example of a how-to. Others that could be added are:

- Parallelization (controlling BLAS multithreading with ``threadpoolctl``, using
  multiprocessing, random number generation, etc.)
- Storing and loading data (``.npy``/``.npz`` format, text formats, Zarr, HDF5,
  Bloscpack, etc.)
- Performance (memory layout, profiling, use with Numba, Cython, or Pythran)
- Writing generic code that works with NumPy, Dask, CuPy, pydata/sparse, etc.

**Explanations**

There is a reasonable amount of content on fundamental NumPy concepts such as
indexing, vectorization, broadcasting, (g)ufuncs, and dtypes. This could be
organized better and clarified to ensure it's really about explaining the concepts
and not mixed with tutorial or how-to like content.

There are few explanations about anything other than those fundamental NumPy
concepts. 

Some examples of concepts that could be expanded:

- Copies vs. Views;
- BLAS and other linear algebra libraries; 
- Fancy indexing.

In addition, there are many explanations in the Reference Guide, which should be
moved to this new dedicated Explanations section.

**Tutorials**

There's a lot of scope for writing better tutorials. We have a new *NumPy for
absolute beginners tutorial* [3]_ (GSoD project of Anne Bonner). In addition we
need a number of tutorials addressing different levels of experience with Python
and NumPy. This could be done using engaging data sets, ideas or stories. For
example, curve fitting with polynomials and functions in ``numpy.linalg`` could
be done with the Keeling curve (decades worth of CO2 concentration in air
measurements) rather than with synthetic random data.

Ideas for tutorials (these capture the types of things that make sense, they're
not necessarily the exact topics we propose to implement):

- Conway's game of life with only NumPy (note: already in `Nicolas Rougier's book
  <https://www.labri.fr/perso/nrougier/from-python-to-numpy/#the-game-of-life>`_)
- Using masked arrays to deal with missing data in time series measurements
- Using Fourier transforms to analyze the Keeling curve data, and extrapolate it.
- Geospatial data (e.g. lat/lon/time to create maps for every year via a stacked
  array, like `gridMet data <http://www.climatologylab.org/gridmet.html>`_)
- Using text data and dtypes (e.g. use speeches from different people, shape
  ``(n_speech, n_sentences, n_words)``)

The *Preparing to Teach* document [2]_ from the Software Carpentry Instructor
Training materials is a nice summary of how to write effective lesson plans (and
tutorials would be very similar). In addition to adding new tutorials, we also
propose a *How to write a tutorial* document, which would help users contribute
new high-quality content to the documentation.

Data sets
---------

Using interesting data in the NumPy docs requires giving all users access to
that data, either inside NumPy or in a separate package. The former is not the
best idea, since it's hard to do without increasing the size of NumPy
significantly. Even for SciPy there has so far been no consensus on this (see
`scipy PR 8707 <https://github.com/scipy/scipy/pull/8707>`_ on adding a new
``scipy.datasets`` subpackage).

So we'll aim for a new (pure Python) package, named ``numpy-datasets`` or
``scipy-datasets`` or something similar. That package can take some lessons from
how, e.g., scikit-learn ships data sets. Small data sets can be included in the
repo, large data sets can be accessed via a downloader class or function.

Related Work
============

Some examples of documentation organization in other projects:

- `Documentation for Jupyter <https://jupyter.org/documentation>`_
- `Documentation for Python <https://docs.python.org/3/>`_
- `Documentation for TensorFlow <https://www.tensorflow.org/learn>`_

These projects make the intended audience for each part of the documentation
more explicit, as well as previewing some of the content in each section. 

Implementation
==============

Currently, the `documentation for NumPy <https://numpy.org/devdocs/>`_ can be
confusing, especially for beginners. Our proposal is to reorganize the docs in
the following structure:

- For users:
    - Absolute Beginners Tutorial
    - main Tutorials section
    - How Tos for common tasks with NumPy
    - Reference Guide (API Reference)
    - Explanations
    - F2Py Guide
    - Glossary
- For developers/contributors:
    - Contributor's Guide
    - Under-the-hood docs
    - Building and extending the documentation
    - Benchmarking 
    - NumPy Enhancement Proposals
- Meta information
    - Reporting bugs
    - Release Notes
    - About NumPy
    - License

Ideas for follow-up
-------------------

Besides rewriting the current documentation to some extent, it would be ideal
to have a technical infrastructure that would allow more contributions from the
community. For example, if Jupyter Notebooks could be submitted as-is as
tutorials or How-Tos, this might create more contributors and broaden the NumPy
community.

Similarly, if people could download some of the documentation in Notebook
format, this would certainly mean people would use less outdated material for
learning NumPy.

It would also be interesting if the new structure for the documentation makes
translations easier.
      
Discussion
==========

Discussion around this NEP can be found on the NumPy mailing list:

- https://mail.python.org/pipermail/numpy-discussion/2020-February/080419.html

References and Footnotes
========================

.. [1] `What nobody tells you about documentation <https://www.divio.com/blog/documentation/>`_

.. [2] `Preparing to Teach <https://carpentries.github.io/instructor-training/15-lesson-study/index.html>`_ (from the `Software Carpentry <https://software-carpentry.org/>`_ Instructor Training materials)

.. [3] `NumPy for absolute beginners Tutorial <https://numpy.org/devdocs/user/absolute_beginners.html>`_ by Anne Bonner

Copyright
=========

This document has been placed in the public domain.
