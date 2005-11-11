"""Simple example to show how to use weave.inline on SWIG2 wrapped
objects.  SWIG2 refers to SWIG versions >= 1.3.

To run this example you must build the trivial SWIG2 extension called
swig2_ext.  To do this you need to do something like this::

 $ swig -c++ -python -I. -o swig2_ext_wrap.cxx swig2_ext.i

 $ g++ -Wall -O2 -I/usr/include/python2.3 -fPIC -I. -c \
   -o swig2_ext_wrap.os swig2_ext_wrap.cxx

 $ g++ -shared -o _swig2_ext.so swig2_ext_wrap.os \
   -L/usr/lib/python2.3/config

The files swig2_ext.i and swig2_ext.h are included in the same
directory that contains this file.

Note that weave's SWIG2 support works fine whether SWIG_COBJECT_TYPES
are used or not.

Author: Prabhu Ramachandran
Copyright (c) 2004, Prabhu Ramachandran
License: BSD Style.

"""

# Import our SWIG2 wrapped library
import swig2_ext

import weave
from weave import swig2_spec, converters

# SWIG2 support is not enabled by default.  We do this by adding the
# swig2 converter to the default list of converters.
converters.default.insert(0, swig2_spec.swig2_converter())

def test():
    """Instantiate the SWIG wrapped object and then call its method
    from C++ using weave.inline
    
    """
    a = swig2_ext.A()
    b = swig2_ext.foo()  # This will be an APtr instance.
    b.thisown = 1 # Prevent memory leaks.
    code = """a->f();
              b->f();
              """
    weave.inline(code, ['a', 'b'], include_dirs=['.'], 
                 headers=['"swig2_ext.h"'], verbose=1)

    
if __name__ == "__main__":
    test()
