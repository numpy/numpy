This guide contains a walkthrough of branching NumPy 2.3.x on Linux.  The
commands can be copied into the command line, but be sure to replace 2.3 and
2.4 by the correct versions. It is good practice to make ``.mailmap`` as
current as possible before making the branch, that may take several weeks.

This should be read together with the
:ref:`general release guide <prepare_release>`.

Branching
=========

Make the branch
---------------

This is only needed when starting a new maintenance branch. The start of a new
development cycle in the main branch should get an annotated tag. That is done
as follows::

    $ git checkout main
    $ git pull upstream main
    $ git commit --allow-empty -m'REL: Begin NumPy 2.4.0 development'
    $ git push upstream HEAD

If the push fails because new PRs have been merged, do::

    $ git pull --rebase upstream

and repeat the push. Once the push succeeds, tag it::

    $ git tag -a -s v2.4.0.dev0 -m'Begin NumPy 2.4.0 development'
    $ git push upstream v2.4.0.dev0

then make the new branch and push it::

    $ git branch maintenance/2.3.x HEAD^
    $ git push upstream maintenance/2.3.x

Prepare the main branch for further development
-----------------------------------------------

Make a PR branch to prepare ``main`` for further development::

    $ git checkout -b 'prepare-main-for-2.4.0-development' v2.4.0.dev0

Delete the release note fragments::

    $ git rm doc/release/upcoming_changes/[0-9]*.*.rst

Create the new release notes skeleton and add to index::

    $ cp doc/source/release/template.rst doc/source/release/2.4.0-notes.rst
    $ gvim doc/source/release/2.4.0-notes.rst  # put the correct version
    $ git add doc/source/release/2.4.0-notes.rst
    $ gvim doc/source/release.rst  # add new notes to notes index
    $ git add doc/source/release.rst

Update ``cversions.txt`` to add current release. There should be no new hash
to worry about at this early point, just add a comment following previous
practice::

    $ gvim numpy/_core/code_generators/cversions.txt
    $ git add numpy/_core/code_generators/cversions.txt

Check your work, commit it, and push::

    $ git status  # check work
    $ git commit -m'REL: Prepare main for NumPy 2.4.0 development'
    $ git push origin HEAD

Now make a pull request.

