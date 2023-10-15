This guide contains a walkthrough of branching NumPy 1.21.x on Linux.  The
commands can be copied into the command line, but be sure to replace 1.21 and
1.22 by the correct versions. It is good practice to make ``.mailmap`` as
current as possible before making the branch, that may take several weeks.

This should be read together with the
:ref:`general release guide <prepare_release>`.

Branching
=========

Make the branch
---------------

This is only needed when starting a new maintenance branch. Because
NumPy now depends on tags to determine the version, the start of a new
development cycle in the main branch needs an annotated tag. That is done
as follows::

    $ git checkout main
    $ git pull upstream main
    $ git commit --allow-empty -m'REL: Begin NumPy 1.22.0 development'
    $ git push upstream HEAD

If the push fails because new PRs have been merged, do::

    $ git pull --rebase upstream

and repeat the push. Once the push succeeds, tag it::

    $ git tag -a -s v1.22.0.dev0 -m'Begin NumPy 1.22.0 development'
    $ git push upstream v1.22.0.dev0

then make the new branch and push it::

    $ git branch maintenance/1.21.x HEAD^
    $ git push upstream maintenance/1.21.x

Prepare the main branch for further development
-----------------------------------------------

Make a PR branch to prepare main for further development::

    $ git checkout -b 'prepare-main-for-1.22.0-development' v1.22.0.dev0

Delete the release note fragments::

    $ git rm doc/release/upcoming_changes/[0-9]*.*.rst

Create the new release notes skeleton and add to index::

    $ cp doc/source/release/template.rst doc/source/release/1.22.0-notes.rst
    $ gvim doc/source/release/1.22.0-notes.rst  # put the correct version
    $ git add doc/source/release/1.22.0-notes.rst
    $ gvim doc/source/release.rst  # add new notes to notes index
    $ git add doc/source/release.rst

Update ``pavement.py`` and update the ``RELEASE_NOTES`` variable to point to
the new notes::

    $ gvim pavement.py
    $ git add pavement.py

Update ``cversions.txt`` to add current release. There should be no new hash
to worry about at this early point, just add a comment following previous
practice::

    $ gvim numpy/core/code_generators/cversions.txt
    $ git add numpy/core/code_generators/cversions.txt

Check your work, commit it, and push::

    $ git status  # check work
    $ git commit -m'REL: Prepare main for NumPy 1.22.0 development'
    $ git push origin HEAD

Now make a pull request.

