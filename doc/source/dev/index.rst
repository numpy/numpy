.. _devindex:

#####################
Contributing to NumPy
#####################

Not a coder? Not a problem! NumPy is multi-faceted, and we can use a lot of help.
These are all activities we'd like to get help with (they're all important, so
we list them in alphabetical order):

- Code maintenance and development
- Community coordination
- DevOps
- Developing educational content & narrative documentation
- Fundraising
- Marketing
- Project management
- Translating content
- Website design and development
- Writing technical documentation

We understand that everyone has a different level of experience,
also NumPy is a pretty well-established project, so it's hard to
make assumptions about an ideal "first-time-contributor". 
So, that's why we don't mark issues with the "good-first-issue"
label. Instead, you'll find `issues labeled "Sprintable" <https://github.com/numpy/numpy/labels/sprintable>`__.
These issues can either be:

- **Easily fixed** when you have guidance from an experienced
  contributor (perfect for working in a sprint).
- **A learning opportunity** for those ready to dive deeper,
  even if you're not in a sprint. 

Additionally, depending on your prior experience, some "Sprintable"
issues might be easy, while others could be more challenging for you.

The rest of this document discusses working on the NumPy code base and documentation.
We're in the process of updating our descriptions of other activities and roles.
If you are interested in these other activities, please contact us!
You can do this via
the `numpy-discussion mailing list <https://mail.python.org/mailman/listinfo/numpy-discussion>`__,
or on `GitHub <https://github.com/numpy/numpy>`__ (open an issue or comment on a
relevant issue). These are our preferred communication channels (open source is open
by nature!), however if you prefer to discuss in a more private space first,
you can do so on Slack (see `numpy.org/contribute
<https://numpy.org/contribute/>`__ for details).

Development process - summary
=============================

Here's the short summary, complete TOC links are below:

1. If you are a first-time contributor:

   * Go to `https://github.com/numpy/numpy
     <https://github.com/numpy/numpy>`_ and click the
     "fork" button to create your own copy of the project.

   * Clone the project to your local computer::

      git clone --recurse-submodules https://github.com/your-username/numpy.git

   * Change the directory::

      cd numpy

   * Add the upstream repository::

      git remote add upstream https://github.com/numpy/numpy.git

   * Now, ``git remote -v`` will show two remote repositories named:

     - ``upstream``, which refers to the ``numpy`` repository
     - ``origin``, which refers to your personal fork

   * Pull the latest changes from upstream, including tags::

      git checkout main
      git pull upstream main --tags

   * Initialize numpy's submodules::

      git submodule update --init

2. Develop your contribution:

   * Create a branch for the feature you want to work on. Since the
     branch name will appear in the merge message, use a sensible name
     such as 'linspace-speedups'::

      git checkout -b linspace-speedups

   * Commit locally as you progress (``git add`` and ``git commit``)
     Use a :ref:`properly formatted<writing-the-commit-message>` commit message,
     write tests that fail before your change and pass afterward, run all the
     :ref:`tests locally<development-environment>`. Be sure to document any
     changed behavior in docstrings, keeping to the NumPy docstring
     :ref:`standard<howto-document>`.

3. To submit your contribution:

   * Push your changes back to your fork on GitHub::

      git push origin linspace-speedups

   * Go to GitHub. The new branch will show up with a green Pull Request
     button. Make sure the title and message are clear, concise, and self-
     explanatory. Then click the button to submit it.

   * If your commit introduces a new feature or changes functionality, post on
     the `mailing list`_ to explain your changes. For bug fixes, documentation
     updates, etc., this is generally not necessary, though if you do not get
     any reaction, do feel free to ask for review.

4. Review process:

   * Reviewers (the other developers and interested community members) will
     write inline and/or general comments on your Pull Request (PR) to help
     you improve its implementation, documentation and style.  Every single
     developer working on the project has their code reviewed, and we've come
     to see it as friendly conversation from which we all learn and the
     overall code quality benefits.  Therefore, please don't let the review
     discourage you from contributing: its only aim is to improve the quality
     of project, not to criticize (we are, after all, very grateful for the
     time you're donating!). See our :ref:`Reviewer Guidelines
     <reviewer-guidelines>` for more information.

   * To update your PR, make your changes on your local repository, commit,
     **run tests, and only if they succeed** push to your fork. As soon as
     those changes are pushed up (to the same branch as before) the PR will
     update automatically. If you have no idea how to fix the test failures,
     you may push your changes anyway and ask for help in a PR comment.

   * Various continuous integration (CI) services are triggered after each PR
     update to build the code, run unit tests, measure code coverage and check
     coding style of your branch. The CI tests must pass before your PR can be
     merged. If CI fails, you can find out why by clicking on the "failed"
     icon (red cross) and inspecting the build and test log. To avoid overuse
     and waste of this resource,
     :ref:`test your work<recommended-development-setup>` locally before
     committing.

   * A PR must be **approved** by at least one core team member before merging.
     Approval means the core team member has carefully reviewed the changes,
     and the PR is ready for merging.

5. Document changes

   Beyond changes to a functions docstring and possible description in the
   general documentation, if your change introduces any user-facing
   modifications they may need to be mentioned in the release notes.
   To add your change to the release notes, you need to create a short file
   with a summary and place it in ``doc/release/upcoming_changes``.
   The file ``doc/release/upcoming_changes/README.rst`` details the format and
   filename conventions.

   If your change introduces a deprecation, make sure to discuss this first on
   GitHub or the mailing list first. If agreement on the deprecation is
   reached, follow :ref:`NEP 23 deprecation policy <NEP23>`  to add the deprecation.

6. Cross referencing issues

   If the PR relates to any issues, you can add the text ``xref gh-xxxx`` where
   ``xxxx`` is the number of the issue to github comments. Likewise, if the PR
   solves an issue, replace the ``xref`` with ``closes``, ``fixes`` or any of
   the other flavors `github accepts <https://help.github.com/en/articles/
   closing-issues-using-keywords>`_.

   In the source code, be sure to preface any issue or PR reference with
   ``gh-xxxx``.

For a more detailed discussion, read on and follow the links at the bottom of
this page.

.. _guidelines:

Guidelines
----------

* All code should have tests (see `test coverage`_ below for more details).
* All code should be `documented <https://numpydoc.readthedocs.io/
  en/latest/format.html#docstring-standard>`_.
* No changes are ever committed without review and approval by a core
  team member. Please ask politely on the PR or on the `mailing list`_ if you
  get no response to your pull request within a week.

.. _stylistic-guidelines:

Stylistic guidelines
--------------------

* Set up your editor to follow `PEP 8 <https://www.python.org/dev/peps/
  pep-0008/>`_ (remove trailing white space, no tabs, etc.).  Check code
  with ruff.

* Use NumPy data types instead of strings (``np.uint8`` instead of
  ``"uint8"``).

* Use the following import conventions::

   import numpy as np

* For C code, see :ref:`NEP 45 <NEP45>`.


Test coverage
-------------

Pull requests (PRs) that modify code should either have new tests, or modify existing
tests to fail before the PR and pass afterwards. You should :ref:`run the tests
<development-environment>` before pushing a PR.

Running NumPy's test suite locally requires some additional packages, such as
``pytest`` and ``hypothesis``. The additional testing dependencies are listed
in ``requirements/test_requirements.txt`` in the top-level directory, and can
conveniently be installed with::

    $ python -m pip install -r requirements/test_requirements.txt

Tests for a module should ideally cover all code in that module,
i.e., statement coverage should be at 100%.

To measure the test coverage, run::

  $ spin test --coverage

This will create a report in ``html`` format at ``build/coverage``, which can be
viewed with your browser, e.g.::

  $ firefox build/coverage/index.html

.. _building-docs:

Building docs
-------------

To build the HTML documentation, use::

  spin docs

You can also run ``make`` from the ``doc`` directory. ``make help`` lists
all targets.

To get the appropriate dependencies and other requirements,
see :ref:`howto-build-docs`.

Development process - details
=============================

The rest of the story

.. toctree::
   :maxdepth: 2

   development_environment
   howto_build_docs
   development_workflow
   development_advanced_debugging
   development_ghcodespaces
   reviewer_guidelines
   ../benchmarking
   NumPy C style guide <https://numpy.org/neps/nep-0045-c_style_guide.html>
   depending_on_numpy
   releasing
   governance/index
   howto-docs

NumPy-specific workflow is in :ref:`numpy-development-workflow
<development-workflow>`.

.. _`mailing list`: https://mail.python.org/mailman/listinfo/numpy-discussion
