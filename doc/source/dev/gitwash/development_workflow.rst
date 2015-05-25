.. _development-workflow:

====================
Development workflow
====================

You already have your own forked copy of the NumPy_ repository, by
following :ref:`forking`, :ref:`set-up-fork`, you have configured git_
by following :ref:`configure-git`, and have linked the upstream
repository as explained in :ref:`linking-to-upstream`.

What is described below is a recommended workflow with Git.

Basic workflow
##############

In short:

1. Start a new *feature branch* for each set of edits that you do.
   See :ref:`below <making-a-new-feature-branch>`.

2. Hack away! See :ref:`below <editing-workflow>`

3. When finished:

   - *Contributors*: push your feature branch to your own Github repo, and
     :ref:`create a pull request <asking-for-merging>`.

   - *Core developers* If you want to push changes without
     further review, see the notes :ref:`below <pushing-to-main>`.
     
This way of working helps to keep work well organized and the history
as clear as possible.

.. seealso::

   There are many online tutorials to help you `learn git`_. For discussions
   of specific git workflows, see these discussions on `linux git workflow`_,
   and `ipython git workflow`_.

.. _making-a-new-feature-branch:

Making a new feature branch
===========================

First, update your master branch with changes that have been made in the main
Numpy repository. In this case, the ``--ff-only`` flag ensures that a new
commit is not created when you merge the upstream and master branches. It is
very important to avoid merging adding new commits to ``master``.

::

   # go to the master branch
   git checkout master
   # download changes from github
   git fetch upstream
   # update the master branch
   git merge upstream/master --ff-only
   # Push new commits to your Github repo
   git push

.. note::

   You could also use ``pull``, which combines ``fetch`` and ``merge``, as
   follows::
   
        git pull --ff-only upstream master

   However, never use ``git pull`` without explicity indicating the source
   branch (as above); the inherent ambiguity can cause problems. This avoids a
   common mistake if you are new to Git. 

Finally create a new branch for your work and check it out::

   git checkout -b my-new-feature master


.. _editing-workflow:

The editing workflow
====================

Overview
--------

::

   # hack hack
   git status # Optional
   git diff # Optional
   git add modified_file
   git commit 
   # push the branch to your own Github repo
   git push

In more detail
--------------

#. Make some changes. When you feel that you've made a complete, working set
   of related changes, move on to the next steps.

#. Optional: Check which files have changed with ``git status`` (see `git
   status`_).  You'll see a listing like this one::

     # On branch my-new-feature
     # Changed but not updated:
     #   (use "git add <file>..." to update what will be committed)
     #   (use "git checkout -- <file>..." to discard changes in working directory)
     #
     #	modified:   README
     #
     # Untracked files:
     #   (use "git add <file>..." to include in what will be committed)
     #
     #	INSTALL
     no changes added to commit (use "git add" and/or "git commit -a")

#. Optional: Compare the changes with the previous version using with ``git
   diff`` (`git diff`_). This brings up a simple text browser interface that
   highlights the difference between your files and the previous verison.

#. Add any relevant modified or new files using  ``git add modified_file``
   (see `git add`_). This puts the files into a staging area, which is a queue
   of files that will be added to your next commit. Only add files that have
   related, complete changes. Leave files with unfinished changes for later
   commits.

#. To commit the staged files into the local copy of your repo, do ``git
   commit``. At this point, a text editor will open up to allow you to write a
   commit message. Read the :ref:`commit message
   section<writing-the-commit-message>` to be sure that you are writing a
   properly formatted and sufficiently detailed commit message. After saving
   your message and closing the editor, your commit will be saved. For trivial
   commits, a short commit message can be passed in through the command line
   using the ``-m`` flag. For example, ``git commit -am "ENH: Some message"``. 
   
   In some cases, you will see this form of the commit command: ``git commit
   -a``. The extra ``-a`` flag automatically commits all modified files and
   removes all deleted files. This can save you some typing of numerous ``git
   add`` commands; however, it can add unwanted changes to a commit if you're
   not careful. For more information, see `why the -a flag?`_ - and the
   helpful use-case description in the `tangled working copy problem`_.  

#. Push the changes to your forked repo on github_::

      git push origin my-new-feature

   For more information, see `git push`_.
    
.. note::
    
   Assuming you have followed the instructions in these pages, git will create
   a default link to your github_ repo called ``origin``.  In git >= 1.7 you
   can ensure that the link to origin is permanently set by using the
   ``--set-upstream`` option::
   
      git push --set-upstream origin my-new-feature
   
   From now on git_ will know that ``my-new-feature`` is related to the
   ``my-new-feature`` branch in your own github_ repo. Subsequent push calls
   are then simplified to the following::

      git push
   
   You have to use ``--set-upstream`` for each new branch that you create.
    

It may be the case that while you were working on your edits, new commits have
been added to ``upstream`` that affect your work. In this case, follow the
:ref:`rebasing-on-master` section of this document to apply those changes to
your branch.

.. _writing-the-commit-message:

Writing the commit message
--------------------------

Commit messages should be clear and follow a few basic rules.  Example::

   ENH: add functionality X to numpy.<submodule>.

   The first line of the commit message starts with a capitalized acronym
   (options listed below) indicating what type of commit this is.  Then a blank
   line, then more text if needed.  Lines shouldn't be longer than 72
   characters.  If the commit is related to a ticket, indicate that with
   "See #3456", "See ticket 3456", "Closes #3456" or similar.

Describing the motivation for a change, the nature of a bug for bug fixes or
some details on what an enhancement does are also good to include in a commit
message.  Messages should be understandable without looking at the code
changes.  A commit message like ``MAINT: fixed another one`` is an example of
what not to do; the reader has to go look for context elsewhere.

Standard acronyms to start the commit message with are::

   API: an (incompatible) API change
   BLD: change related to building numpy
   BUG: bug fix
   DEP: deprecate something, or remove a deprecated object
   DEV: development tool or utility
   DOC: documentation
   ENH: enhancement
   MAINT: maintenance commit (refactoring, typos, etc.)
   REV: revert an earlier commit
   STY: style fix (whitespace, PEP8)
   TST: addition or modification of tests
   REL: related to releasing numpy


.. _asking-for-merging:

Asking for your changes to be merged with the main repo
=======================================================

When you feel your work is finished, you can create a pull request (PR). Github
has a nice help page that outlines the process for `filing pull requests`_. 

If your changes involve modifications to the API or addition/modification of a
function, you should initiate a code review. This involves sending an email to
the `NumPy mailing list`_ with a link to your PR along with a description of
and a motivation for your changes.

.. _pushing-to-main:

Pushing changes to the main repo
================================

*This is only relevant if you have commit rights to the main Numpy repo.*

When you have a set of "ready" changes in a feature branch ready for
Numpy's ``master`` or ``maintenance`` branches, you can push
them to ``upstream`` as follows:

1. First, merge or rebase on the target branch.

   a) Only a few, unrelated commits then prefer rebasing::

        git fetch upstream
        git rebase upstream/master

      See :ref:`rebasing-on-master`.

   b) If all of the commits are related, create a merge commit::

        git fetch upstream
        git merge --no-ff upstream/master

2. Check that what you are going to push looks sensible::

        git log -p upstream/master..
        git log --oneline --graph

3. Push to upstream::

        git push upstream my-feature-branch:master

.. note:: 

    It's usually a good idea to use the ``-n`` flag to ``git push`` to check
    first that you're about to push the changes you want to the place you
    want.


.. _rebasing-on-master:

Rebasing on master
==================

This updates your feature branch with changes from the upstream `NumPy
github`_ repo. If you do not absolutely need to do this, try to avoid doing
it, except perhaps when you are finished. The first step will be to update
your master branch with new commits from upstream. This is done in the same
manner as described at the beginning of :ref:`making-a-new-feature-branch`.
Next, you need to update the feature branch::

   # go to the feature branch
   git checkout my-new-feature
   # make a backup in case you mess up
   git branch tmp my-new-feature
   # rebase on master
   git rebase master

If you have made changes to files that have changed also upstream,
this may generate merge conflicts that you need to resolve. See
:ref:`below<recovering-from-mess-up>` for help in this case.

Finally, remove the backup branch upon a successful rebase::

   git branch -D tmp

.. _recovering-from-mess-up:

Recovering from mess-ups
========================

Sometimes, you mess up merges or rebases. Luckily, in Git it is
relatively straightforward to recover from such mistakes.

If you mess up during a rebase::

   git rebase --abort

If you notice you messed up after the rebase::

   # reset branch back to the saved point
   git reset --hard tmp

If you forgot to make a backup branch::

   # look at the reflog of the branch
   git reflog show my-feature-branch

   8630830 my-feature-branch@{0}: commit: BUG: io: close file handles immediately
   278dd2a my-feature-branch@{1}: rebase finished: refs/heads/my-feature-branch onto 11ee694744f2552d
   26aa21a my-feature-branch@{2}: commit: BUG: lib: make seek_gzip_factory not leak gzip obj
   ...

   # reset the branch to where it was before the botched rebase
   git reset --hard my-feature-branch@{2}

If you didn't actually mess up but there are merge conflicts, you need to
resolve those.  This can be one of the trickier things to get right.  For a
good description of how to do this, see `this article on merging conflicts`_.


Additional things you might want to do
######################################

.. _rewriting-commit-history:

Rewriting commit history
========================

.. note::

   Do this only for your own feature branches.

There's an embarrassing typo in a commit you made? Or perhaps the you
made several false starts you would like the posterity not to see.

This can be done via *interactive rebasing*.

Suppose that the commit history looks like this::

    git log --oneline
    eadc391 Fix some remaining bugs
    a815645 Modify it so that it works
    2dec1ac Fix a few bugs + disable
    13d7934 First implementation
    6ad92e5 * masked is now an instance of a new object, MaskedConstant
    29001ed Add pre-nep for a copule of structured_array_extensions.
    ...

and ``6ad92e5`` is the last commit in the ``master`` branch. Suppose we
want to make the following changes:

* Rewrite the commit message for ``13d7934`` to something more sensible.
* Combine the commits ``2dec1ac``, ``a815645``, ``eadc391`` into a single one.

We do as follows::

    # make a backup of the current state
    git branch tmp HEAD
    # interactive rebase
    git rebase -i 6ad92e5

This will open an editor with the following text in it::

    pick 13d7934 First implementation
    pick 2dec1ac Fix a few bugs + disable
    pick a815645 Modify it so that it works
    pick eadc391 Fix some remaining bugs

    # Rebase 6ad92e5..eadc391 onto 6ad92e5
    #
    # Commands:
    #  p, pick = use commit
    #  r, reword = use commit, but edit the commit message
    #  e, edit = use commit, but stop for amending
    #  s, squash = use commit, but meld into previous commit
    #  f, fixup = like "squash", but discard this commit's log message
    #
    # If you remove a line here THAT COMMIT WILL BE LOST.
    # However, if you remove everything, the rebase will be aborted.
    #

To achieve what we want, we will make the following changes to it::

    r 13d7934 First implementation
    pick 2dec1ac Fix a few bugs + disable
    f a815645 Modify it so that it works
    f eadc391 Fix some remaining bugs

This means that (i) we want to edit the commit message for
``13d7934``, and (ii) collapse the last three commits into one. Now we
save and quit the editor.

Git will then immediately bring up an editor for editing the commit
message. After revising it, we get the output::

    [detached HEAD 721fc64] FOO: First implementation
     2 files changed, 199 insertions(+), 66 deletions(-)
    [detached HEAD 0f22701] Fix a few bugs + disable
     1 files changed, 79 insertions(+), 61 deletions(-)
    Successfully rebased and updated refs/heads/my-feature-branch.

and the history looks now like this::

     0f22701 Fix a few bugs + disable
     721fc64 ENH: Sophisticated feature
     6ad92e5 * masked is now an instance of a new object, MaskedConstant

If it went wrong, recovery is again possible as explained :ref:`above
<recovering-from-mess-up>`.

Deleting a branch on github_
============================

::

   git checkout master
   # delete branch locally
   git branch -D my-unwanted-branch
   # delete branch on github
   git push origin :my-unwanted-branch

(Note the colon ``:`` before ``test-branch``.  See also:
http://github.com/guides/remove-a-remote-branch


Several people sharing a single repository
==========================================

If you want to work on some stuff with other people, where you are all
committing into the same repository, or even the same branch, then just
share it via github_.

First fork NumPy into your account, as from :ref:`forking`.

Then, go to your forked repository github page, say
``http://github.com/your-user-name/numpy``

Click on the 'Admin' button, and add anyone else to the repo as a
collaborator:

   .. image:: pull_button.png

Now all those people can do::

    git clone git@githhub.com:your-user-name/numpy.git

Remember that links starting with ``git@`` use the ssh protocol and are
read-write; links starting with ``git://`` are read-only.

Your collaborators can then commit directly into that repo with the
usual::

     git commit -am 'ENH - much better code'
     git push origin master # pushes directly into your repo

Exploring your repository
=========================

To see a graphical representation of the repository branches and
commits::

   gitk --all

To see a linear list of commits for this branch::

   git log

You can also look at the `network graph visualizer`_ for your github_
repo.

Backporting
===========

Backporting is the process of copying new feature/fixes committed in
`numpy/master`_ back to stable release branches. To do this you make a branch
off the branch you are backporting to, cherry pick the commits you want from
``numpy/master``, and then submit a pull request for the branch containing the
backport.

1. Assuming you already have a fork of NumPy on Github. We need to
   update it from upstream::

    # Add upstream.
    git remote add upstream https://github.com/numpy/numpy.git

    # Get the latest updates.
    git fetch upstream

    # Make sure you are on master.
    git checkout master

    # Apply the updates locally.
    git rebase upstream/master

    # Push the updated code to your github repo.
    git push origin

2. Next you need to make the branch you will work on. This needs to be
   based on the older version of NumPy (not master)::

    # Make a new branch based on numpy/maintenance/1.8.x,
    # backport-3324 is our new name for the branch.
    git checkout -b backport-3324 upstream/maintenance/1.8.x

3. Now you need to apply the changes from master to this branch using
   `git cherry-pick`_::

    # This pull request included commits aa7a047 to c098283 (inclusive)
    # so you use the .. syntax (for a range of commits), the ^ makes the
    # range inclusive.
    git cherry-pick aa7a047^..c098283
    ...
    # Fix any conflicts, then if needed:
    git cherry-pick --continue

4. You might run into some conflicts cherry picking here. These are
   resolved the same way as merge/rebase conflicts. Except here you can
   use `git blame`_ to see the difference between master and the
   backported branch to make sure nothing gets screwed up.

5. Push the new branch to your Github repository::

    git push -u origin backport-3324

6. Finally make a pull request using Github. Make sure it is against the
   maintenance branch and not master, Github will usually suggest you
   make the pull request against master.

.. include:: git_links.inc
