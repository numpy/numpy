.. _set-up-fork:

==================
 Set up your fork
==================

First you follow the instructions for :ref:`forking`. 

Overview
========

::

   git clone git@github.com:your-user-name/numpy.git
   cd numpy
   git remote add upstream git://github.com/numpy/numpy.git

In detail
=========

Clone your fork
---------------

#. Clone your fork to the local computer with ``git clone
   git@github.com:your-user-name/numpy.git``
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
   git remote add upstream git://github.com/numpy/numpy.git

``upstream`` here is just the arbitrary name we're using to refer to the
main NumPy_ repository at `NumPy github`_.

Note that we've used ``git://`` for the URL rather than ``git@``.  The
``git://`` URL is read only.  This means we that we can't accidentally
(or deliberately) write to the upstream repo, and we are only going to
use it to merge into our own code.

Just for your own satisfaction, show yourself that you now have a new
'remote', with ``git remote -v show``, giving you something like::

   upstream	git://github.com/numpy/numpy.git (fetch)
   upstream	git://github.com/numpy/numpy.git (push)
   origin	git@github.com:your-user-name/numpy.git (fetch)
   origin	git@github.com:your-user-name/numpy.git (push)

.. include:: git_links.inc

