.. _reviewer-guidelines:

===================
Reviewer Guidelines
===================

Reviewing open pull requests (PRs) is very welcome and a valuable way to help
increase the speed at which the project moves forward. We encourage everyone to
get involved in the review process; it's also a great way to get familiar with
the codebase.

Who can be a reviewer?
======================

You don't have to be a project member to contribute reviews. Consider being a
reviewer if 

- You have contributed to the NumPy project before, and feel like you can
  comment on technical or organizational aspects of the PR; 
- You haven't contributed before, but are an experienced NumPy user, and see a
  pull request that touches a funcionality that you use often;
- You can contribute comments about content, style, organization or good
  documentation practices.  

If we do not know you yet, consider introducing yourself before you start
reviewing pull requests.

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
  If you find yourself making many small suggestions, either open a PR against
  the original branch, push changes to the contributor branch, or merge the PR
  and then open a new PR against upstream.
- If you need help writing replies in reviews, check out some `Standard replies
  for reviewing
  <https://scikit-learn.org/stable/developers/tips.html#saved-replies>`_.

Reviewer Checklist
==================

- Is the intended behavior clear under all conditions (e.g. unexpected inputs
  like empty arrays or nan/inf values)?
- Is the code easy to read and low on redundancy? Should variable names be
  improved for clarity or consistency? Should comments be added? Should comments
  be removed as unhelpful or extraneous?
- Does the documentation follow the :ref:`NumPy guidelines<howto-document>`? Are
  the docstrings properly formatted?
- If it is not obvious from the PR description, add a short explanation of what
  a branch did to the merge message and, if closing an issue, also add "Closes
  #123" where 123 is the issue number.
- For code changes, at least one core team member (those with commit rights)
  should review and approve all pull requests. If you are the first to review a
  PR and approve of the changes use the GitHub `approve review
  <https://help.github.com/articles/reviewing-changes-in-pull-requests/>`_ tool
  to mark it as such. If you are a subsequent reviewer, please approve the
  review or, if you have commit rights and think no more review is needed, merge
  the PR.

Core team members
-----------------
  
- Make sure all automated CI tests pass before merging a PR, and that the
  :ref:`documentation builds<building-docs>` without any errors.
- In case of merge conflicts, ask the PR submitter to :ref:`rebase on master
  <rebasing-on-master>`.
- For PRs that add new features or are in some way complex, wait at least a day
  or two before merging it. That way, others get a chance to comment before the
  code goes in. Consider adding it to the release notes.
- When merging contributions, a committer is responsible for ensuring that those
  meet the requirements outlined in the :ref:`Development process guidelines
  <guidelines>` for NumPy. Also, check that new features and backwards
  compatibility breaks were discussed on the `numpy-discussion mailing list
  <https://mail.python.org/mailman/listinfo/numpy-discussion>`_.
- Backports and trivial additions to finish a PR (really trivial, like a typo or
  PEP8 fix) can be pushed directly.
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
  (same way as Changes requested). Alternatively (for core team members): merge
  the pull request, if you think it is ready to be merged.

It may be helpful to have a copy of the pull request code checked out on your
own machine so that you can play with it locally. One way to do this is to
navigate to the NumPy root directory in the terminal and enter::
  
  git fetch upstream pull/PULL_REQUEST_ID/head:NEW_BRANCH_NAME

where ``PULL_REQUEST_ID`` is the number corresponding with the pull request
(e.g. 10286 for PR #10286) and ``NEW_BRANCH_NAME`` is whatever name you'd like
to use to refer to the author's code (e.g. review_10286).

Now you can check out the branch::

  git checkout NEW_BRANCH_NAME

which converts the code in your local repository to match the author's modified
version of NumPy.

Alternatively, you can use the `GitHub CLI
<https://docs.github.com/en/github/getting-started-with-github/github-cli>`_ to
do this by clicking the ``Open with`` button in the upper right-hand corner of
the PR page. 

Assuming you have your :ref:`development environment<development-environment>`
set up, you can now build the code and test it.

.. include:: gitwash/git_links.inc
