#####################
Contributing to NumPy
#####################

Development process
===================

Here's the long and short of it:

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
     such as 'transform-speedups'::

      git checkout -b transform-speedups

   * Commit locally as you progress (``git add`` and ``git commit``)
     Use a `properly formatted <writing-the-commit-message>` commit message,
     write tests that fail before your change and pass afterward, run all the
     `tests locally <development-environment>` after running ``git clean -xfd``
     to be sure you have not broken anything, and be sure to document any
     changed behavior in `numpydoc-appropriate docstrings <howto-document>`.

3. To submit your contribution:

   * Push your changes back to your fork on GitHub::

      git push origin transform-speedups

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

   * A code-change PR must be **approved** by at least two core team members
     before merging. Approval means the core team member has carefully reviewed
     the changes, and the PR is ready for merging.

5. Document changes

   Beyond changes to a functions docstring and possible description in the
   general documentation, if your change introduces any user-facing
   modifications, update the current release notice under
   ``doc/release/X.XX-notes.rst``

   If your change introduces a deprecation, make sure to follow the
   `deprecation policy <http://www.numpy.org/neps/nep-0023-backwards-
   compatibility.html>`_.

6. Cross referencing issues

   If the PR relates to any issues, you can add the text ``xref #xxxx`` where
   ``xxxx`` is the number of the issue to github comments. Likewise, if the PR
   solves an issue, replace the ``xref`` with ``closes``, ``fixes`` or any of
   the other flavors `github accepts <https://help.github.com/en/articles/
   closing-issues-using-keywords>`_.

   In the source code, be sure to preface any issue or PR reference with
   ``gh-xxxx`` since the code may move to a different host.

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
  team member.  Ask on the `mailing list`_ if you get no response to your pull
  request.

Stylistic Guidelines
--------------------

* Set up your editor to follow `PEP08 <https://www.python.org/dev/peps/
  pep-0008/>`_ (remove trailing white space, no tabs, etc.).  Check code with
  pyflakes / flake8.

* Use numpy data types instead of strings (``np.uint8`` instead of
  ``"uint8"``).

* Use the following import conventions::

   import numpy as np

   cimport numpy as cnp  # in Cython code

* Use ``np_intp`` as data type for all indexing, shape and size variables
  in C/C++ and Cython code.

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

This will create a report in `build/coverage`, which can be viewed with

```
firefox build/coverage/index.html
```

Building docs
-------------

To build docs, run ``make`` from the ``doc`` directory. ``make help`` lists
all targets. For example, to build the HTML documentation, you can run:

.. code:: sh

    make html

Then, all the HTML files will be generated in ``doc/build/html/``.
To rebuild a full clean documentation, run:

.. code:: sh

    make dist

Requirements
~~~~~~~~~~~~

`Sphinx <http://www.sphinx-doc.org/en/stable/>`__ and LaTeX are needed to build
the documentation.

**Sphinx:**

Sphinx and other python packages needed to build the documentation
Python dependencies can be installed using:

.. code:: sh

    pip install -r sphinx matplotlib scipy

**LaTeX Ubuntu:**

.. code:: sh

    sudo apt-get install -qq texlive texlive-latex-extra dvipng

**LaTeX Mac:**

Install the full `MacTex <https://www.tug.org/mactex/>`__ installation or
install the smaller
`BasicTex <https://www.tug.org/mactex/morepackages.html>`__ and add *ucs*
and *dvipng* packages:

.. code:: sh

    sudo tlmgr install ucs dvipng

Fixing Warnings
~~~~~~~~~~~~~~~

-  "citation not found: R###" There is probably an underscore after a
   reference in the first line of a docstring (e.g. [1]\_). Use this
   method to find the source file: $ cd doc/build; grep -rin R####

-  "Duplicate citation R###, other instance in..."" There is probably a
   [2] without a [1] in one of the docstrings

-  Make sure to use pre-sphinxification paths to images (not the
   \_images directory)

Deprecation cycle
-----------------

If the behavior of the library has to be changed, a deprecation cycle must be
followed to warn users.

- a deprecation cycle is *not* necessary when:

    * adding a new function, or
    * adding a new keyword argument to the *end* of a function signature, or
    * fixing what was buggy behaviour

- a deprecation cycle is necessary for *any breaking API change*, meaning a
    change where the function, invoked with the same arguments, would return a
    different result after the change. This includes:

    * changing the order of arguments or keyword arguments, or
    * adding arguments or keyword arguments to a function, or
    * changing a function's name or submodule, or
    * changing the default value of a function's arguments.

Usually, our policy is to put in place a deprecation cycle over two releases.

For the sake of illustration, we consider the modification of a default value in
a function signature. In version N (therefore, next release will be N+1), we
have

.. code-block:: python

    def a_function(array, rescale=True):
        out = do_something(array, rescale=rescale)
        return out

that has to be changed to

.. code-block:: python

    def a_function(array, rescale=None):
        if rescale is None:
            # 2019-02-24
            warn('The default value of rescale will change to `False`
                 in version N+3', DeprecationWarning, stacklevel=2)
            rescale = True
        out = do_something(image, rescale=rescale)
        return out

and in version N+3

.. code-block:: python

    def a_function(array, rescale=False):
        out = do_something(array, rescale=rescale)
        return out

Here is the process for a 2-release deprecation cycle:

- In the signature, set default to `None`, and modify the docstring to specify
  that it's `True`.
- In the function, _if_ rescale is set to `None`, set to `True` and warn that the
  default will change to `False` in version N+3.
- In ``doc/release/X.XX-notes.rst``, under deprecations, add "In
  `a_function`, the `rescale` argument will default to `False` in N+3."

Note that the 2-release deprecation cycle is not a strict rule and in some
cases, the developers can agree on a different procedure upon justification
(like when we can't detect the change, or it involves moving or deleting an
entire function for example).

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

NumPy-specific workflow is in `development-workflow`.

.. _`mailing list`: https://mail.python.org/mailman/listinfo/numpy-devel
