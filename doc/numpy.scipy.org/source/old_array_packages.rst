====================
Older Array Packages
====================

It may take months for the large code base that uses Numeric and/or Numarray 
to transition to the new NumPy system.  Links to the older packages are 
provided here.  New users should start out with NumPy.

Much of the documentation for Numeric and Numarray is applicable to the NumPy package.  However, there are `significant feature improvements <http://numpy.scipy.org/new_features.html>`_.  A complete guide to the new system has been written by the primary developer, Travis Oliphant. It is now in the public domain.  Other Documentation is available at `the scipy website <http://www.scipy.org/>`_ and in the docstrings (which can be extracted using pydoc). Free Documentation for Numeric (most of which is still valid) is `here <http://numpy.scipy.org/numpydoc/numdoc.htm>`_ or as a `pdf <http://numpy.scipy.org/numpy.pdf>`_ file.   Obviously you should replace references to Numeric in that document with numpy (i.e. instead of "import Numeric", use "import numpy").  

Upgrading from historical implementations
=========================================

NumPy derives from the old Numeric code base and can be used as a replacement for Numeric.   It also adds the features introduced by Numarray and can also be used to replace Numarray.  

Numeric users should find the transition relatively easy (although not without some effort).  There is a module (numpy.oldnumeric.alter_code1) that can makemost of the necessary changes to your Python code that used Numeric to work with NumPy's Numeric compatibility module. 

Users of numarray can also transition their code using a similar module (numpy.numarray.alter_code1) and the numpy.numarray compatibility layer. 

C-code written to either package can be easily ported to NumPy using "numpy/oldnumeric.h" and "numpy/libnumarray.h" for the Numeric C-API and the Numarray C-API respectively. `Sourceforge download site <http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103>`_

For about 6 months at the end of 2005, the new package was called SciPy Core (not to be confused with the full SciPy package which remains a `separate <http://www.scipy.org/>`_ package), and so you will occasionally see references to SciPy Core floating around.  It was decided in January 2006 to go with the historical name of NumPy for the new package.  Realize that NumPy (module name numpy) is the new name.   Because of the name-change, there were a lot of dicussions that took place on scipy-dev@scipy.org and scipy-user@scipy.org.  If you have a question about the new system, you may wish to run a search on those mailing lists as well as the main NumPy list (numpy-discussion@lists.sourceforge.net)

Numeric (version 24.2)
======================

Numeric was the first array object built for Python.  It has been quite successful and is used in a wide variety of settings and applications.   Maintenance has ceased for Numeric, and users should transisition to NumPy as quickly as possible.   There is a module called numpy.oldnumeric.alter_code1 in NumPy that can make the transition to NumPy easier (it will automatically perform the search-and-replace style changes that need to be made to python code that uses Numeric to make it work with NumPy). 

Documentation for Numeric is at http://numpy.scipy.org/numpydoc/numdoc.htm> or as a `pdf <http://numpy.scipy.org/numpy.pdf>`_ file `Sourceforge Numeric Download Page <http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=1351>`_

Numarray
========

Numarray is another implementation of an array object for Python written after 
Numeric and before NumPy. Sponsors of numarray have indicated they will be 
moving to NumPy as soon as is feasible for them so that eventually numarray 
will be phased out (probably sometime in 2007). This project shares some of 
the resources with the Numeric sourceforge site but maintains its own web page
at http://www.stsci.edu/resources/software_hardware/numarray
`Sourceforge Numarray Download Page <http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=32367>`_