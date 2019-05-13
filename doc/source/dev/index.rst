#####################
Contributing to NumPy
#####################

Development process - summary
=============================

Here's the short summary, complete TOC links are below:

1. If you are a first-time contributor:

   * Go to `https://github.com/numpy/numpy
     <https://github.com/numpy/numpy>`_ and click the
     "fork" button to create your own copy of the project.

   * Clone the project to your local computer::

      git clone https://github.com/your-username/numpy.git

   * Change the directory::

      cd numpy

   * Add the upstream repository::

      git remote add upstream https://github.com/numpy/numpy.git

   * Now, `git remote -v` will show two remote repositories named:

     - ``upstream``, which refers to the ``numpy`` repository
     - ``origin``, which refers to your personal fork

2. Develop your contribution:

   * Pull the latest changes from upstream::

      git checkout master
      git pull upstream master

   * Create a branch for the feature you want to work on. Since the
     branch name will appear in the merge message, use a sensible name
     such as 'linspace-speedups'::

      git checkout -b linspace-speedups

   * Commit locally as you progress (``git add`` and ``git commit``)
     Use a `properly formatted <writing-the-commit-message>` commit message,
     write tests that fail before your change and pass afterward, run all the
     `tests locally <development-environment>`. Be sure to document any
     changed behavior in docstrings, keeping to the NumPy docstring
     `standard <howto-document>`.

3. To submit your contribution:

   * Push your changes back to your fork on GitHub::

      git push origin linspace-speedups

   * Enter your GitHub username and password (repeat contributors or advanced
     users can remove this step by connecting to GitHub with `SSH <set-up-and-
     configure-a-github-account>`.

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
     time you're donating!).

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
     and waste of this resource, `test your work <recommended-development-
     setup>` locally before committing.

   * A PR must be **approved** by at least one core team member before merging.
     Approval means the core team member has carefully reviewed the changes,
     and the PR is ready for merging.

5. Document changes

   Beyond changes to a functions docstring and possible description in the
   general documentation, if your change introduces any user-facing
   modifications, update the current release notes under
   ``doc/release/X.XX-notes.rst``

   If your change introduces a deprecation, make sure to discuss this first on
   GitHub or the mailing list first. If agreement on the deprecation is
   reached, follow `NEP 23 deprecation policy <http://www.numpy.org/neps/
   nep-0023-backwards-compatibility.html>`_  to add the deprecation.

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

Divergence between ``upstream/master`` and your feature branch
--------------------------------------------------------------

If GitHub indicates that the branch of your Pull Request can no longer
be merged automatically, you have to incorporate changes that have been made
since you started into your branch. Our recommended way to do this is to
`rebase on master <rebasing-on-master>`.

Guidelines
----------

* All code should have tests (see `test coverage`_ below for more details).
* All code should be `documented <docstring-standard>`.
* No changes are ever committed without review and approval by a core
  team member.Please ask politely on the PR or on the `mailing list`_ if you
  get no response to your pull request within a week.

Stylistic Guidelines
--------------------

* Set up your editor to follow `PEP 8 <https://www.python.org/dev/peps/
  pep-0008/>`_ (remove trailing white space, no tabs, etc.).  Check code with
  pyflakes / flake8.

* Use numpy data types instead of strings (``np.uint8`` instead of
  ``"uint8"``).

* Use the following import conventions::

   import numpy as np

* For C code, see the `numpy-c-style-guide`


Test coverage
-------------

Pull requests (PRs) that modify code should either have new tests, or modify existing
tests to fail before the PR and pass afterwards. You should `run the tests
<development-environment>` before pushing a PR.

Tests for a module should ideally cover all code in that module,
i.e., statement coverage should be at 100%.

To measure the test coverage, install
`pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`__
and then run::

  $ python runtests.py --coverage

This will create a report in `build/coverage`, which can be viewed with::

  $ firefox build/coverage/index.html

Building docs
-------------

To build docs, run ``make`` from the ``doc`` directory. ``make help`` lists
all targets. For example, to build the HTML documentation, you can run:

.. code:: sh

    make html

Then, all the HTML files will be generated in ``doc/build/html/``.
Since the documentation is based on docstrings, the appropriate version of
numpy must be installed in the host python used to run sphinx.

Requirements
~~~~~~~~~~~~

`Sphinx <http://www.sphinx-doc.org/en/stable/>`__ is needed to build
the documentation. Matplotlib and SciPy are also required.

Fixing Warnings
~~~~~~~~~~~~~~~

-  "citation not found: R###" There is probably an underscore after a
   reference in the first line of a docstring (e.g. [1]\_). Use this
   method to find the source file: $ cd doc/build; grep -rin R####

-  "Duplicate citation R###, other instance in..."" There is probably a
   [2] without a [1] in one of the docstrings

Development process - details
=============================

The rest of the story

.. toctree::
   :maxdepth: 2

   conduct/code_of_conduct
   Git Basics <gitwash/index>
   development_environment
   development_workflow
   ../benchmarking
   style_guide
   releasing
   governance/index

NumPy-specific workflow is in `numpy-development-workflow`.

.. _`mailing list`: https://mail.python.org/mailman/listinfo/numpy-devel
