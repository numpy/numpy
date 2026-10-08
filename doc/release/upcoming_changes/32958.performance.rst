Faster ``import numpy`` on Python 3.15+
---------------------------------------
On Python 3.15+, NumPy uses lazy imports (PEP 810): much of NumPy is only
imported when it is first used, which makes ``import numpy`` about 40% faster.
