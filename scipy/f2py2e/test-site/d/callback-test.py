#!/usr/bin/env python
"""

Copyright 2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2000/01/28 13:17:19 $
Pearu Peterson
"""

__version__ = "$Revision: 1.2 $[10:-1]"

import cbfoo
from Numeric import *
def fun(x):
    return x
i=0
while i<15000:
    i=i+1
    cbfoo.foo(fun,i,(2,))
print "ok"



