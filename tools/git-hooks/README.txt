Introduction
------------
The 'git' distributed source code management system has the capability to run
user defined scripts called 'hooks' when particular 'git' commands are used.
These hooks can be used for anything from sending an e-mail to validating the
commit to make sure it follows particular standards.

Contents
--------
hooks/              This directory will hold the hooks for the Numpy project
install-wrappers    Shell script to copy wrappers to ../../.git/hooks
README.txt          This file
wrappers/           The wrapper scripts

Notes
-----
The 'git' system will look in ${GIT_DIR}/hooks for scripts, which is a
location that 'git' will not be able to track files.  In order to have 'git'
tracked files, you either have to link to them or use wrapper scripts.
Wrapper scripts have a better chance of running on Windows.

It is not a requirement to use 'git' hooks, but as they improve they will
become increasingly helpful.

Installation
------------
The wrapper scripts have to be installed by copying them to ${GIT_DIR}/hooks
(relative to this directory ../../.git/hooks).  This is all that the
convenience script 'install-wrappers' does for you.

Documentation
-------------
http://book.git-scm.com/5_git_hooks.html
http://progit.org/book/ch7-3.html
http://stackoverflow.com/questions/3148863/how-can-i-customize-gits-merge-commit-message

