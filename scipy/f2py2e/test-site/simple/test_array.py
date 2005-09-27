#!/usr/bin/env python
"""

Copyright 2001 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Revision: 1.1 $
$Date: 2001/12/13 16:56:10 $
Pearu Peterson
"""

__version__ = "$Id: test_array.py,v 1.1 2001/12/13 16:56:10 pearu Exp $"


import sys,os
from distutils import util
p = os.path.abspath(os.path.join('.',
                                 'build',"lib.%s-%s" % \
                                 (util.get_platform(),
                                  sys.version[0:3])))
sys.path.insert(0,p)
from array import *
from Numeric import array

if 1:
    def fun(a):
        return a+2
    a = bar(fun,3)
    assert a==5,`a`

    def fun(a):
        return [2,3]
    a = bar(fun,3)
    assert a==2

    def fun(a):
        raise 'fun_error'
    try:
        a = bar(fun,3)
    except 'fun_error':
        pass

    def fun(a):
        if a>2:
            return bar(fun,a-1)
        return a
    a = bar(fun,5)
    assert a==2

    def fun(a):
        if a>2:
            return bar(fun,a-1)
        raise ValueError,'fun_mess'
    try:
        a = bar(fun,5)
    except ValueError,mess:
        assert str(mess)=='fun_mess'

print 'ok'

sys.exit(0)
