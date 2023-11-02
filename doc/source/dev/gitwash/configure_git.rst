.. _configure-git:

=================
Git configuration
=================

.. _git-config-basic:

Overview
========

Your personal git_ configurations are saved in the ``.gitconfig`` file in
your home directory.
Here is an example ``.gitconfig`` file::

  [user]
          name = Your Name
          email = you@yourdomain.example.com

  [alias]
          ci = commit -a
          co = checkout
          st = status -a
          stat = status -a
          br = branch
          wdiff = diff --color-words

  [core]
          editor = vim

  [merge]
          summary = true

You can edit this file directly or you can use the ``git config --global``
command::

  git config --global user.name "Your Name"
  git config --global user.email you@yourdomain.example.com
  git config --global alias.ci "commit -a"
  git config --global alias.co checkout
  git config --global alias.st "status -a"
  git config --global alias.stat "status -a"
  git config --global alias.br branch
  git config --global alias.wdiff "diff --color-words"
  git config --global core.editor vim
  git config --global merge.summary true

To set up on another computer, you can copy your ``~/.gitconfig`` file,
or run the commands above.

In detail
=========

user.name and user.email
------------------------

It is good practice to tell git_ who you are, for labeling any changes
you make to the code.  The simplest way to do this is from the command
line::

  git config --global user.name "Your Name"
  git config --global user.email you@yourdomain.example.com

This will write the settings into your git configuration file,  which
should now contain a user section with your name and email::

  [user]
        name = Your Name
        email = you@yourdomain.example.com

Of course you'll need to replace ``Your Name`` and ``you@yourdomain.example.com``
with your actual name and email address.

Aliases
-------

You might well benefit from some aliases to common commands.

For example, you might well want to be able to shorten ``git checkout``
to ``git co``.  Or you may want to alias ``git diff --color-words``
(which gives a nicely formatted output of the diff) to ``git wdiff``

The following ``git config --global`` commands::

  git config --global alias.ci "commit -a"
  git config --global alias.co checkout
  git config --global alias.st "status -a"
  git config --global alias.stat "status -a"
  git config --global alias.br branch
  git config --global alias.wdiff "diff --color-words"

will create an ``alias`` section in your ``.gitconfig`` file with contents
like this::

  [alias]
          ci = commit -a
          co = checkout
          st = status -a
          stat = status -a
          br = branch
          wdiff = diff --color-words

Editor
------

You may also want to make sure that your editor of choice is used ::

  git config --global core.editor vim

Merging
-------

To enforce summaries when doing merges (``~/.gitconfig`` file again)::

   [merge]
      log = true

Or from the command line::

  git config --global merge.log true


.. include:: git_links.inc
