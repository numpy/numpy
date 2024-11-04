This is a walkthrough of the NumPy 2.1.0 release on Linux, modified for
building with GitHub Actions and cibuildwheels and uploading to the
`anaconda.org staging repository for NumPy <https://anaconda.org/multibuild-wheels-staging/numpy>`_.
The commands can be copied into the command line, but be sure to replace 2.1.0
by the correct version. This should be read together with the
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

When adding or dropping Python versions, three files need to be edited:

- .github/workflows/wheels.yml  # for github cibuildwheel
- tools/ci/cirrus_wheels.yml  # for cibuildwheel aarch64/arm64 builds
- pyproject.toml  # for classifier and minimum version check.

Make these changes in an ordinary PR against main and backport if necessary.
Add ``[wheel build]`` at the end of the title line of the commit summary so
that wheel builds will be run to test the changes. We currently release wheels
for new Python versions after the first Python rc once manylinux and
cibuildwheel support it. For Python 3.11 we were able to release within a week
of the rc1 announcement.


Backport pull requests
----------------------

Changes that have been marked for this release must be backported to the
maintenance/2.1.x branch.

Update 2.1.0 milestones
-----------------------

Look at the issues/prs with 2.1.0 milestones and either push them off to a
later version, or maybe remove the milestone. You may need to add a milestone.


Make a release PR
=================

Four documents usually need to be updated or created for the release PR:

- The changelog
- The release notes
- The ``.mailmap`` file
- The ``pyproject.toml`` file

These changes should be made in an ordinary PR against the maintenance branch.
The commit heading should contain a ``[wheel build]`` directive to test if the
wheels build. Other small, miscellaneous fixes may be part of this PR. The
commit message might be something like::

    REL: Prepare for the NumPy 2.1.0 release [wheel build]

    - Create 2.1.0-changelog.rst.
    - Update 2.1.0-notes.rst.
    - Update .mailmap.
    - Update pyproject.toml


Set the release version
-----------------------

Check the ``pyproject.toml`` file and set the release version if needed::

    $ gvim pyproject.toml


Check the ``pavement.py`` and ``doc/source/release.rst`` files
--------------------------------------------------------------

Check that the ``pavement.py`` file points to the correct release notes. It should
have been updated after the last release, but if not, fix it now. Also make
sure that the notes have an entry in the ``release.rst`` file::

    $ gvim pavement.py doc/source/release.rst


Generate the changelog
----------------------

The changelog is generated using the changelog tool::

    $ spin changelog $GITHUB v2.0.0..maintenance/2.1.x > doc/changelog/2.1.0-changelog.rst

where ``GITHUB`` contains your GitHub access token. The text will need to be
checked for non-standard contributor names and dependabot entries removed. It
is also a good idea to remove any links that may be present in the PR titles
as they don't translate well to markdown, replace them with monospaced text. The
non-standard contributor names should be fixed by updating the ``.mailmap``
file, which is a lot of work. It is best to make several trial runs before
reaching this point and ping the malefactors using a GitHub issue to get the
needed information.


Finish the release notes
------------------------

If there are any release notes snippets in ``doc/release/upcoming_changes/``,
run ``spin notes``, which will incorporate the snippets into the
``doc/source/release/notes-towncrier.rst`` file and delete the snippets::

    $ spin notes
    $ gvim doc/source/release/notes-towncrier.rst doc/source/release/2.1.0-notes.rst
    
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

    $ git checkout maintenance/2.1.x
    $ git pull upstream maintenance/2.1.x
    $ git submodule update
    $ git clean -xdfq

Sanity check::

    $ python3 -m spin test -m full

Tag the release and push the tag. This requires write permission for the numpy
repository::

    $ git tag -a -s v2.1.0 -m"NumPy 2.1.0 release"
    $ git push upstream v2.1.0

If you need to delete the tag due to error::

   $ git tag -d v2.1.0
   $ git push --delete upstream v2.1.0


2. Build wheels
---------------

Tagging the build at the beginning of this process will trigger a wheel build
via cibuildwheel and upload wheels and an sdist to the staging repo. The CI run
on github actions (for all x86-based and macOS arm64 wheels) takes about 1 1/4
hours. The CI runs on cirrus (for aarch64 and M1) take less time. You can check
for uploaded files at the `staging repository`_, but note that it is not
closely synched with what you see of the running jobs.

If you wish to manually trigger a wheel build, you can do so:

- On github actions -> `Wheel builder`_ there is a "Run workflow" button, click
  on it and choose the tag to build
- On Cirrus we don't currently have an easy way to manually trigger builds and
  uploads.

If a wheel build fails for unrelated reasons, you can rerun it individually:

- On github actions select `Wheel builder`_ click on the commit that contains
  the build you want to rerun. On the left there is a list of wheel builds,
  select the one you want to rerun and on the resulting page hit the
  counterclockwise arrows button.
- On cirrus, log into cirrusci, look for the v2.1.0 tag and rerun the failed jobs.

.. _`staging repository`: https://anaconda.org/multibuild-wheels-staging/numpy/files
.. _`Wheel builder`: https://github.com/numpy/numpy/actions/workflows/wheels.yml


3. Download wheels
------------------

When the wheels have all been successfully built and staged, download them from the
Anaconda staging directory using the ``tools/download-wheels.py`` script::

    $ cd ../numpy
    $ mkdir -p release/installers
    $ python3 tools/download-wheels.py 2.1.0


4. Generate the README files
----------------------------

This needs to be done after all installers are downloaded, but before the pavement
file is updated for continued development::

    $ paver write_release


5. Upload to PyPI
-----------------

Upload to PyPI using ``twine``. A recent version of ``twine`` of is needed
after recent PyPI changes, version ``3.4.1`` was used here::

    $ cd ../numpy
    $ twine upload release/installers/*.whl
    $ twine upload release/installers/*.gz  # Upload last.

If one of the commands breaks in the middle, you may need to selectively upload
the remaining files because PyPI does not allow the same file to be uploaded
twice. The source file should be uploaded last to avoid synchronization
problems that might occur if pip users access the files while this is in
process, causing pip to build from source rather than downloading a binary
wheel. PyPI only allows a single source distribution, here we have
chosen the zip archive.


6. Upload files to GitHub
-------------------------

Go to `<https://github.com/numpy/numpy/releases>`_, there should be a ``v2.1.0
tag``, click on it and hit the edit button for that tag and update the title to
'v2.1.0 (<date>). There are two ways to add files, using an editable text
window and as binary uploads. Start by editing the ``release/README.md`` that
is translated from the rst version using pandoc. Things that will need fixing:
PR lines from the changelog, if included, are wrapped and need unwrapping,
links should be changed to monospaced text.  Then copy the contents to the
clipboard and paste them into the text window. It may take several tries to get
it look right. Then

- Upload ``release/installers/numpy-2.1.0.tar.gz`` as a binary file.
- Upload ``release/README.rst`` as a binary file.
- Upload ``doc/changelog/2.1.0-changelog.rst`` as a binary file.
- Check the pre-release button if this is a pre-releases.
- Hit the ``{Publish,Update} release`` button at the bottom.


7. Upload documents to numpy.org (skip for prereleases)
-------------------------------------------------------

.. note:: You will need a GitHub personal access token to push the update.

This step is only needed for final releases and can be skipped for pre-releases
and most patch releases. ``make merge-doc`` clones the ``numpy/doc`` repo into
``doc/build/merge`` and updates it with the new documentation::

    $ git clean -xdfq
    $ git co v2.1.0
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

    $ ln -sfn 2.1 stable
    $ ls -l  # check the link

Once everything seems satisfactory, update, commit and upload the changes::

    $ git commit -a -m"Add documentation for v2.1.0"
    $ git push git@github.com:numpy/doc
    $ popd


8. Reset the maintenance branch into a development state (skip for prereleases)
-------------------------------------------------------------------------------

Create release notes for next release and edit them to set the version. These
notes will be a skeleton and have little content::

    $ git checkout -b begin-2.1.1 maintenance/2.1.x
    $ cp doc/source/release/template.rst doc/source/release/2.1.1-notes.rst
    $ gvim doc/source/release/2.1.1-notes.rst
    $ git add doc/source/release/2.1.1-notes.rst

Add new release notes to the documentation release list and update the
``RELEASE_NOTES`` variable in ``pavement.py``::

    $ gvim doc/source/release.rst pavement.py

Update the ``version`` in ``pyproject.toml``::

    $ gvim pyproject.toml

Commit the result::

    $ git commit -a -m"MAINT: Prepare 2.1.x for further development"
    $ git push origin HEAD

Go to GitHub and make a PR. It should be merged quickly.


9. Announce the release on numpy.org (skip for prereleases)
-----------------------------------------------------------

This assumes that you have forked `<https://github.com/numpy/numpy.org>`_::

    $ cd ../numpy.org
    $ git checkout main
    $ git pull upstream main
    $ git checkout -b announce-numpy-2.1.0
    $ gvim content/en/news.md

- For all releases, go to the bottom of the page and add a one line link. Look
  to the previous links for example.
- For the ``*.0`` release in a cycle, add a new section at the top with a short
  description of the new features and point the news link to it.

commit and push::

    $ git commit -a -m"announce the NumPy 2.1.0 release"
    $ git push origin HEAD

Go to GitHub and make a PR.


10. Announce to mailing lists
-----------------------------

The release should be announced on the numpy-discussion, scipy-devel, and
python-announce-list mailing lists. Look at previous announcements for the
basic template. The contributor and PR lists are the same as generated for the
release notes above. If you crosspost, make sure that python-announce-list is
BCC so that replies will not be sent to that list.


11. Post-release update main (skip for prereleases)
---------------------------------------------------

Checkout main and forward port the documentation changes. You may also want
to update these notes if procedures have changed or improved::

    $ git checkout -b post-2.1.0-release-update main
    $ git checkout maintenance/2.1.x doc/source/release/2.1.0-notes.rst
    $ git checkout maintenance/2.1.x doc/changelog/2.1.0-changelog.rst
    $ git checkout maintenance/2.1.x .mailmap  # only if updated for release.
    $ gvim doc/source/release.rst  # Add link to new notes
    $ git status  # check status before commit
    $ git commit -a -m"MAINT: Update main after 2.1.0 release."
    $ git push origin HEAD

Go to GitHub and make a PR.

