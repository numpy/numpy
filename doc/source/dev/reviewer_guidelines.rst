.. _reviewer-guidelines:

===================
Reviewer Guidelines
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
- If you need help writing replies in reviews, check out some `Standard replies
  for reviewing
  <https://scikit-learn.org/stable/developers/tips.html#saved-replies>`_.

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

.. include:: gitwash/git_links.inc
