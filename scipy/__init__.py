"""SciPy Core

You can support the development of SciPy by purchasing documentation at

http://www.trelgol.com

It is being distributed for a fee for a limited time to try and raise money for
development.
"""


import os as _os
_install_init = _os.path.join(_os.path.dirname(__file__),'install__init__.py')
if _os.path.isfile(_install_init):
    execfile(_install_init)
else:
    print 'Running from source directory.'
del _os
