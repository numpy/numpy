NumPy Reference Guide
=====================

Instructions
------------
1. Optionally download an XML dump of the newest docstrings from the doc wiki
   at ``/pydocweb/dump`` and save it as ``dump.xml``.
2. Run ``make html`` or ``make dist``

You can also run ``summarize.py`` to see which parts of the Numpy
namespace are documented.


TODO
----

* Numberless [*] footnotes cause LaTeX errors.

* ``See also`` sections are still somehow broken even if some work.
  The problem is that Sphinx searches like this::

      'name'
      'active_module.name'
      'active_module.active_class.name'.
  
  Whereas, we would like to have this:

      'name'
      'active_module.name'
      'parent_of_active_module.name'
      'parent_of_parent_of_active_module.name'
      ...
      'numpy.name'

  We can get one step upwards by always using 'numpy' as the active module.
  It seems difficult to beat Sphinx to do what we want.
  Do we need to change our docstring standard slightly, ie. allow only
  leaving the 'numpy.' prefix away?

* Link resolution doesn't work as intended... eg. `doc.ufunc`_
