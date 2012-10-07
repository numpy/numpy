.. _following-latest:

=============================
 Following the latest source
=============================

These are the instructions if you just want to follow the latest
*NumPy* source, but you don't need to do any development for now.
If you do want to contribute a patch (excellent!) or do more extensive
NumPy development, see :ref:`development-workflow`.

The steps are:

* :ref:`install-git`
* get local copy of the git repository from Github_
* update local copy from time to time

Get the local copy of the code
==============================

From the command line::

   git clone git://github.com/numpy/numpy.git

You now have a copy of the code tree in the new ``numpy`` directory.
If this doesn't work you can try the alternative read-only url::

   git clone https://github.com/numpy/numpy.git

Updating the code
=================

From time to time you may want to pull down the latest code.  Do this with::

   cd numpy
   git fetch
   git merge --ff-only

The tree in ``numpy`` will now have the latest changes from the initial
repository.

.. _Github: https://github.com/numpy
