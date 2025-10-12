This is a walkthrough of the NumPy 2.4.0 release on Linux, which will be the
first feature release using the `numpy/numpy-release
<https://github.com/numpy/numpy-release>`__ repository.

The commands can be copied into the command line, but be sure to replace 2.4.0
with the correct version. This should be read together with the
:ref:`general release guide <prepare_release>`.

Facility preparation
====================

Before beginning to make a release, use the ``requirements/*_requirements.txt`` files to
ensure that you have the needed software. Most software can be installed with
pip, but some will require apt-get, dnf, or whatever your system uses for
software. You will also need a GitHub personal access token (PAT) to push the
documentation. There are a few ways to streamline things:

- Git can be set up to use a keyring to store your GitHub personal access token.
  Search online for the details.
- You can use the ``keyring`` app to store the PyPI password for twine. See the
  online twine documentation for details.


Prior to release
================

Add/drop Python versions
------------------------

When adding or dropping Python versions, multiple config and CI files need to
be edited in addition to changing the minimum version in ``pyproject.toml``.
Make these changes in an ordinary PR against main and backport if necessary.
We currently release wheels for new Python versions after the first Python RC
once manylinux and cibuildwheel support that new Python version.


Backport pull requests
----------------------

Changes that have been marked for this release must be backported to the
maintenance/2.4.x branch.


Update 2.4.0 milestones
-----------------------

Look at the issues/prs with 2.4.0 milestones and either push them off to a
later version, or maybe remove the milestone. You may need to add a milestone.


Make a release PR
=================

Four documents usually need to be updated or created for the release PR:

- The changelog
- The release notes
- The ``.mailmap`` file
- The ``pyproject.toml`` file

These changes should be made in an ordinary PR against the maintenance branch.
Other small, miscellaneous fixes may be part of this PR. The commit message
might be something like::

    REL: Prepare for the NumPy 2.4.0 release

    - Create 2.4.0-changelog.rst.
    - Update 2.4.0-notes.rst.
    - Update .mailmap.
    - Update pyproject.toml


Set the release version
-----------------------

Check the ``pyproject.toml`` file and set the release version and update the
classifier if needed::

    $ gvim pyproject.toml


Check the ``doc/source/release.rst`` file
-----------------------------------------

make sure that the release notes have an entry in the ``release.rst`` file::

    $ gvim doc/source/release.rst


Generate the changelog
----------------------

The changelog is generated using the changelog tool::

    $ spin changelog $GITHUB v2.3.0..maintenance/2.4.x > doc/changelog/2.4.0-changelog.rst

where ``GITHUB`` contains your GitHub access token. The text will need to be
checked for non-standard contributor names. It is also a good idea to remove
any links that may be present in the PR titles as they don't translate well to
Markdown, replace them with monospaced text. The non-standard contributor names
should be fixed by updating the ``.mailmap`` file, which is a lot of work. It
is best to make several trial runs before reaching this point and ping the
malefactors using a GitHub issue to get the needed information.


Finish the release notes
------------------------

If there are any release notes snippets in ``doc/release/upcoming_changes/``,
run ``spin notes``, which will incorporate the snippets into the
``doc/source/release/notes-towncrier.rst`` file and delete the snippets::

    $ spin notes
    $ gvim doc/source/release/notes-towncrier.rst doc/source/release/2.4.0-notes.rst
    
Once the ``notes-towncrier`` contents has been incorporated into release note
the ``.. include:: notes-towncrier.rst`` directive can be removed.  The notes
will always need some fixups, the introduction will need to be written, and
significant changes should be called out. For patch releases the changelog text
may also be appended, but not for the initial release as it is too long. Check
previous release notes to see how this is done.


Release walkthrough
===================

Note that in the code snippets below, ``upstream`` refers to the root repository on
GitHub and ``origin`` to its fork in your personal GitHub repositories. You may
need to make adjustments if you have not forked the repository but simply
cloned it locally. You can also edit ``.git/config`` and add ``upstream`` if it
isn't already present.


1. Prepare the release commit
-----------------------------

Checkout the branch for the release, make sure it is up to date, and clean the
repository::

    $ git checkout maintenance/2.4.x
    $ git pull upstream maintenance/2.4.x
    $ git submodule update
    $ git clean -xdfq

Sanity check::

    $ python3 -m spin test -m full

Tag the release and push the tag. This requires write permission for the numpy
repository::

    $ git tag -a -s v2.4.0 -m"NumPy 2.4.0 release"
    $ git push upstream v2.4.0

If you need to delete the tag due to error::

   $ git tag -d v2.4.0
   $ git push --delete upstream v2.4.0


2. Build wheels and sdist
-------------------------

Create a ``maintenance/2.4.x`` branch in the ``numpy-release`` repository,
and open a PR changing the ``SOURCE_REF_TO_BUILD`` identifier at the top of
``.github/workflows/wheels.yml`` to ``v2.4.0``. That will do a full set of
wheel builds on the PR, if everything looks good merge the PR.

All wheels are currently built in that repository on GitHub Actions, they take
about 1 hour to build. 

If you wish to manually trigger a wheel build, you can do so: in your browser,
go to `numpy-release/actions/workflows/wheels.yml <https://github.com/numpy/numpy-release/actions/workflows/wheels.yml>`__
and click on the "Run workflow" button, then choose the tag to build. If some
wheel builds fail for unrelated reasons, you can re-run them as normal
in the GitHub Actions UI with "re-run failed".

Once you are ready to publish a release to PyPI, use that same "Run workflow"
button and choose ``pypi`` in the *environment* dropdown. All wheels and the
sdist will build and be ready to release to PyPI after manual inspection that
everything passed. E.g., the number of artifacts is correct, and the wheel
filenames and sizes look as expected. If desired, you can also download an
artifact for local unzipping and inspection. You will get an email notification
as well with a "Review pending deployments" link. Once you're ready, press the
button to start the uploads to PyPI, which will complete the release.


3. Upload files to GitHub Releases
----------------------------------

Go to `<https://github.com/numpy/numpy/releases>`_, there should be a ``v2.4.0
tag``, click on it and hit the edit button for that tag and update the title to
"v2.4.0 (<date>)". There are two ways to add files, using an editable text
window and as binary uploads.

Start by running ``spin notes 2.4.0`` and then edit the ``release/README.md``
that is translated from the rst version using pandoc. Things that will need
fixing: PR lines from the changelog, if included, are wrapped and need
unwrapping, links should be changed to monospaced text. Then copy the contents
to the clipboard and paste them into the text window. It may take several tries
to get it look right. Then

- Download the sdist (``numpy-2.4.0.tar.gz``) from PyPI upload it to GitHub as
  a binary file.
- Upload ``release/README.rst`` as a binary file.
- Upload ``doc/changelog/2.4.0-changelog.rst`` as a binary file.
- Check the pre-release button if this is a pre-releases.
- Hit the ``Publish release`` button at the bottom.

.. note::
   Please ensure that all 3 files are uploaded are present and the
   release text is complete. Releases are configured to be immutable, so
   mistakes can't (easily) be fixed anymore.


4. Upload documents to numpy.org (skip for prereleases)
-------------------------------------------------------

.. note:: You will need a GitHub personal access token to push the update.

This step is only needed for final releases and can be skipped for pre-releases
and most patch releases. ``make merge-doc`` clones the ``numpy/doc`` repo into
``doc/build/merge`` and updates it with the new documentation::

    $ git clean -xdfq
    $ git co v2.4.0
    $ rm -rf doc/build  # want version to be current
    $ python -m spin docs merge-doc --build
    $ pushd doc/build/merge

If the release series is a new one, you will need to add a new section to the
``doc/build/merge/index.html`` front page just after the "insert here" comment::

    $ gvim index.html +/'insert here'

Further, update the version-switcher json file to add the new release and
update the version marked ``(stable)`` and ``preferred``::

    $ gvim _static/versions.json

Then run ``update.py`` to update the version in ``_static``::

    $ python3 update.py

You can "test run" the new documentation in a browser to make sure the links
work, although the version dropdown will not change, it pulls its information
from ``numpy.org``::

    $ firefox index.html  # or google-chrome, etc.

Update the stable link and update::

    $ ln -sfn 2.4 stable
    $ ls -l  # check the link

Once everything seems satisfactory, update, commit and upload the changes::

    $ git commit -a -m"Add documentation for v2.4.0"
    $ git push git@github.com:numpy/doc
    $ popd


5. Reset the maintenance branch into a development state (skip for prereleases)
-------------------------------------------------------------------------------

Create release notes for next release and edit them to set the version. These
notes will be a skeleton and have little content::

    $ git checkout -b begin-2.4.1 maintenance/2.4.x
    $ cp doc/source/release/template.rst doc/source/release/2.4.1-notes.rst
    $ gvim doc/source/release/2.4.1-notes.rst
    $ git add doc/source/release/2.4.1-notes.rst

Add new release notes to the documentation release list. Then update the
``version`` in ``pyproject.toml``::

    $ gvim pyproject.toml

Commit the result::

    $ git commit -a -m"MAINT: Prepare 2.4.x for further development"
    $ git push origin HEAD

Go to GitHub and make a PR. It should be merged quickly.


6. Announce the release on numpy.org (skip for prereleases)
-----------------------------------------------------------

This assumes that you have forked `<https://github.com/numpy/numpy.org>`_::

    $ cd ../numpy.org
    $ git checkout main
    $ git pull upstream main
    $ git checkout -b announce-numpy-2.4.0
    $ gvim content/en/news.md

- For all releases, go to the bottom of the page and add a one line link. Look
  to the previous links for example.
- For the ``*.0`` release in a cycle, add a new section at the top with a short
  description of the new features and point the news link to it.
- Edit the newsHeader and date fields at the top of news.md
- Also edit the butttonText on line 14 in content/en/config.yaml

commit and push::

    $ git commit -a -m"announce the NumPy 2.4.0 release"
    $ git push origin HEAD

Go to GitHub and make a PR.


7. Announce to mailing lists
----------------------------

The release should be announced on the numpy-discussion and
python-announce-list mailing lists. Look at previous announcements for the
basic template. The contributor and PR lists are the same as generated for the
release notes above. If you crosspost, make sure that python-announce-list is
BCC so that replies will not be sent to that list.


8. Post-release update main (skip for prereleases)
--------------------------------------------------

Checkout main and forward port the documentation changes. You may also want
to update these notes if procedures have changed or improved::

    $ git checkout -b post-2.4.0-release-update main
    $ git checkout maintenance/2.4.x doc/source/release/2.4.0-notes.rst
    $ git checkout maintenance/2.4.x doc/changelog/2.4.0-changelog.rst
    $ git checkout maintenance/2.4.x .mailmap  # only if updated for release.
    $ gvim doc/source/release.rst  # Add link to new notes
    $ git status  # check status before commit
    $ git commit -a -m"MAINT: Update main after 2.4.0 release."
    $ git push origin HEAD

Go to GitHub and make a PR.

