====================================
Getting started with Git development
====================================

This section and the next describe in detail how to set up git for working
with the NumPy source code.  If you have git already set up, skip to
:ref:`development-workflow`.

Basic Git setup
###############

* :ref:`install-git`.
* Introduce yourself to Git::

      git config --global user.email you@yourdomain.example.com
      git config --global user.name "Your Name Comes Here"

.. _forking:

Making your own copy (fork) of NumPy
####################################

You need to do this only once.  The instructions here are very similar
to the instructions at http://help.github.com/forking/ - please see that
page for more detail.  We're repeating some of it here just to give the
specifics for the NumPy_ project, and to suggest some default names.

.. _set-up-and-configure-a-github-account:

Set up and configure a github_ account
======================================

If you don't have a github_ account, go to the github_ page, and make one.

You then need to configure your account to allow write access - see the
``Generating SSH keys`` help on `github help`_.

Create your own forked copy of NumPy_
=========================================

#. Log into your github_ account.
#. Go to the NumPy_ github home at `NumPy github`_.
#. Click on the *fork* button:

   .. image:: forking_button.png

   After a short pause, you should find yourself at the home page for
   your own forked copy of NumPy_.

.. include:: git_links.inc


.. _set-up-fork:

Set up your fork
################

First you follow the instructions for :ref:`forking`.

Overview
========

::

   git clone https://github.com/your-user-name/numpy.git
   cd numpy
   git remote add upstream https://github.com/numpy/numpy.git

In detail
=========

Clone your fork
---------------

#. Clone your fork to the local computer with ``git clone
   https://github.com/your-user-name/numpy.git``
#. Investigate.  Change directory to your new repo: ``cd numpy``. Then
   ``git branch -a`` to show you all branches.  You'll get something
   like::

      * master
      remotes/origin/master

   This tells you that you are currently on the ``master`` branch, and
   that you also have a ``remote`` connection to ``origin/master``.
   What remote repository is ``remote/origin``? Try ``git remote -v`` to
   see the URLs for the remote.  They will point to your github_ fork.

   Now you want to connect to the upstream `NumPy github`_ repository, so
   you can merge in changes from trunk.

.. _linking-to-upstream:

Linking your repository to the upstream repo
--------------------------------------------

::

   cd numpy
   git remote add upstream https://github.com/numpy/numpy.git

``upstream`` here is just the arbitrary name we're using to refer to the
main NumPy_ repository at `NumPy github`_.

Just for your own satisfaction, show yourself that you now have a new
'remote', with ``git remote -v show``, giving you something like::

   upstream	https://github.com/numpy/numpy.git (fetch)
   upstream	https://github.com/numpy/numpy.git (push)
   origin	https://github.com/your-user-name/numpy.git (fetch)
   origin	https://github.com/your-user-name/numpy.git (push)

To keep in sync with changes in NumPy, you want to set up your repository
so it pulls from ``upstream`` by default.  This can be done with::

   git config branch.master.remote upstream
   git config branch.master.merge refs/heads/master

You may also want to have easy access to all pull requests sent to the
NumPy repository::

   git config --add remote.upstream.fetch '+refs/pull/*/head:refs/remotes/upstream/pr/*'

Your config file should now look something like (from
``$ cat .git/config``)::

   [core]
           repositoryformatversion = 0
           filemode = true
           bare = false
           logallrefupdates = true
           ignorecase = true
           precomposeunicode = false
   [remote "origin"]
           url = https://github.com/your-user-name/numpy.git
           fetch = +refs/heads/*:refs/remotes/origin/*
   [remote "upstream"]
           url = https://github.com/numpy/numpy.git
           fetch = +refs/heads/*:refs/remotes/upstream/*
           fetch = +refs/pull/*/head:refs/remotes/upstream/pr/*
   [branch "master"]
           remote = upstream
           merge = refs/heads/master

.. include:: git_links.inc
