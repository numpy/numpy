.. _reviewer-guidelines:

===================
Reviewer guidelines
===================

Reviewing open pull requests (PRs) helps move the project forward. We encourage
people outside the project to get involved as well; it's a great way to get
familiar with the codebase.

Who can be a reviewer?
======================

Reviews can come from outside the NumPy team -- we welcome contributions from
domain experts (for instance, `linalg` or `fft`) or maintainers of other
projects. You do not need to be a NumPy maintainer (a NumPy team member with
permission to merge a PR) to review.

If we do not know you yet, consider introducing yourself in `the mailing list or
Slack <https://numpy.org/community/>`_ before you start reviewing pull requests.

Communication Guidelines
========================

- Every PR, good or bad, is an act of generosity. Opening with a positive
  comment will help the author feel rewarded, and your subsequent remarks may be
  heard more clearly. You may feel good also.
- Begin if possible with the large issues, so the author knows they've been
  understood. Resist the temptation to immediately go line by line, or to open
  with small pervasive issues.
- You are the face of the project, and NumPy some time ago decided `the kind of
  project it will be <https://numpy.org/code-of-conduct/>`_: open, empathetic,
  welcoming, friendly and patient. Be `kind
  <https://youtu.be/tzFWz5fiVKU?t=49m30s>`_ to contributors.
- Do not let perfect be the enemy of the good, particularly for documentation.
  If you find yourself making many small suggestions, or being too nitpicky on
  style or grammar, consider merging the current PR when all important concerns
  are addressed. Then, either push a commit directly (if you are a maintainer)
  or open a follow-up PR yourself.
- If you need help writing replies in reviews, check out some
  :ref:`standard replies for reviewing<saved-replies>`.

Reviewer Checklist
==================

- Is the intended behavior clear under all conditions? Some things to watch:
   - What happens with unexpected inputs like empty arrays or nan/inf values?
   - Are axis or shape arguments tested to be `int` or `tuples`?
   - Are unusual `dtypes` tested if a function supports those?
- Should variable names be improved for clarity or consistency?
- Should comments be added, or rather removed as unhelpful or extraneous?
- Does the documentation follow the :ref:`NumPy guidelines<howto-document>`? Are
  the docstrings properly formatted?
- Does the code follow NumPy's :ref:`Stylistic Guidelines<stylistic-guidelines>`?
- If you are a maintainer, and it is not obvious from the PR description, add a
  short explanation of what a branch did to the merge message and, if closing an
  issue, also add "Closes gh-123" where 123 is the issue number.
- For code changes, at least one maintainer (i.e. someone with commit rights)
  should review and approve a pull request. If you are the first to review a
  PR and approve of the changes use the GitHub `approve review
  <https://help.github.com/articles/reviewing-changes-in-pull-requests/>`_ tool
  to mark it as such. If a PR is straightforward, for example it's a clearly
  correct bug fix, it can be merged straight away. If it's more complex or
  changes public API, please leave it open for at least a couple of days so
  other maintainers get a chance to review.
- If you are a subsequent reviewer on an already approved PR, please use the
  same review method as for a new PR (focus on the larger issues, resist the
  temptation to add only a few nitpicks).  If you have commit rights and think
  no more review is needed, merge the PR.

For maintainers
---------------
  
- Make sure all automated CI tests pass before merging a PR, and that the
  :ref:`documentation builds <building-docs>` without any errors.
- In case of merge conflicts, ask the PR submitter to :ref:`rebase on main
  <rebasing-on-main>`.
- For PRs that add new features or are in some way complex, wait at least a day
  or two before merging it. That way, others get a chance to comment before the
  code goes in. Consider adding it to the release notes.
- When merging contributions, a committer is responsible for ensuring that those
  meet the requirements outlined in the :ref:`Development process guidelines
  <guidelines>` for NumPy. Also, check that new features and backwards
  compatibility breaks were discussed on the `numpy-discussion mailing list
  <https://mail.python.org/mailman/listinfo/numpy-discussion>`_.
- Squashing commits or cleaning up commit messages of a PR that you consider too
  messy is OK. Remember to retain the original author's name when doing this.
  Make sure commit messages follow the :ref:`rules for NumPy
  <writing-the-commit-message>`.
- When you want to reject a PR: if it's very obvious, you can just close it and
  explain why. If it's not, then it's a good idea to first explain why you
  think the PR is not suitable for inclusion in NumPy and then let a second
  committer comment or close.
- If the PR submitter doesn't respond to your comments for 6 months, move the PR 
  in question to the inactive category with the “inactive” tag attached. 
  At this point, the PR can be closed by a maintainer. If there is any interest 
  in finalizing the PR under consideration, this can be indicated at any time, 
  without waiting 6 months, by a comment.  
- Maintainers are encouraged to finalize PRs when only small changes are
  necessary before merging (e.g., fixing code style or grammatical errors).
  If a PR becomes inactive, maintainers may make larger changes. 
  Remember, a PR is a collaboration between a contributor and a reviewer/s, 
  sometimes a direct push is the best way to finish it.

API Changes
-----------
As mentioned most public API changes should be discussed ahead of time and
often with a wider audience (on the mailing list, or even through a NEP).

For changes in the public C-API be aware that the NumPy C-API is backwards
compatible so that any addition must be ABI compatible with previous versions.
When it is not the case, you must add a guard.

For example ``PyUnicodeScalarObject`` struct contains the following::

    #if NPY_FEATURE_VERSION >= NPY_1_20_API_VERSION
        char *buffer_fmt;
    #endif

Because the ``buffer_fmt`` field was added to its end in NumPy 1.20 (all
previous fields remained ABI compatible).
Similarly, any function added to the API table in
``numpy/core/code_generators/numpy_api.py`` must use the ``MinVersion``
annotation.
For example::

    'PyDataMem_SetHandler':                 (304, MinVersion("1.22")),

Header only functionality (such as a new macro) typically does not need to be
guarded.

GitHub Workflow
---------------

When reviewing pull requests, please use workflow tracking features on GitHub as
appropriate:

- After you have finished reviewing, if you want to ask for the submitter to
  make changes, change your review status to "Changes requested." This can be
  done on GitHub, PR page, Files changed tab, Review changes (button on the top
  right).
- If you're happy about the current status, mark the pull request as Approved
  (same way as Changes requested). Alternatively (for maintainers): merge
  the pull request, if you think it is ready to be merged.

It may be helpful to have a copy of the pull request code checked out on your
own machine so that you can play with it locally. You can use the `GitHub CLI
<https://docs.github.com/en/github/getting-started-with-github/github-cli>`_ to
do this by clicking the ``Open with`` button in the upper right-hand corner of
the PR page. 

Assuming you have your :ref:`development environment<development-environment>`
set up, you can now build the code and test it.

.. _saved-replies:

Standard replies for reviewing
==============================

It may be helpful to store some of these in GitHub's `saved
replies <https://github.com/settings/replies/>`_ for reviewing:

**Usage question**
    .. code-block:: md

        You are asking a usage question. The issue tracker is for bugs and new features.
        I'm going to close this issue, feel free to ask for help via our [help channels](https://numpy.org/gethelp/).

**You’re welcome to update the docs**
    .. code-block:: md

        Please feel free to offer a pull request updating the documentation if you feel it could be improved.

**Self-contained example for bug**
    .. code-block:: md

        Please provide a [self-contained example code](https://stackoverflow.com/help/mcve), including imports and data (if possible), so that other contributors can just run it and reproduce your issue.
        Ideally your example code should be minimal.

**Software versions**
    .. code-block:: md

        To help diagnose your issue, please paste the output of:
        ```
        python -c 'import numpy; print(numpy.version.version)'
        ```
        Thanks.

**Code blocks**
    .. code-block:: md

        Readability can be greatly improved if you [format](https://help.github.com/articles/creating-and-highlighting-code-blocks/) your code snippets and complete error messages appropriately.
        You can edit your issue descriptions and comments at any time to improve readability.
        This helps maintainers a lot. Thanks!

**Linking to code**
    .. code-block:: md

        For clarity's sake, you can link to code like [this](https://help.github.com/articles/creating-a-permanent-link-to-a-code-snippet/).

**Better description and title**
    .. code-block:: md

        Please make the title of the PR more descriptive.
        The title will become the commit message when this is merged.
        You should state what issue (or PR) it fixes/resolves in the description using the syntax described [here](https://docs.github.com/en/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue#linking-a-pull-request-to-an-issue-using-a-keyword).

**Regression test needed**
    .. code-block:: md

        Please add a [non-regression test](https://en.wikipedia.org/wiki/Non-regression_testing) that would fail at main but pass in this PR.

**Don’t change unrelated**
    .. code-block:: md

        Please do not change unrelated lines. It makes your contribution harder to review and may introduce merge conflicts to other pull requests.

.. include:: gitwash/git_links.inc
