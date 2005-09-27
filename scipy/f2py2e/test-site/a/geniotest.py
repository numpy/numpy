#!/usr/bin/env python
"""
Usage:
   Run runme.py instead.

Copyright 1999 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2002/01/02 22:43:51 $
Pearu Peterson
"""

__version__ = "$Revision: 1.10 $[10:-1]"

ffname='iotestrout.f'
hfname='iotest.pyf'
pyname='runiotest.py'
from Numeric import *
import pprint,string
show=pprint.pprint
mmap={
    'real*8':{'pref':'d','in':4.7,'inout':5.3},
    'real':{'pref':'f','in':4,'inout':5.3},
    'integer*8':{'pref':'l','in':4,'inout':-7L},
    'integer':{'pref':'i','in':4.1,'inout':-7.2},
    'integer*2':{'pref':'s','in':4,'inout':-7},
    'integer*1':{'pref':'b','in':4,'inout':-7},
    'logical':{'pref':'i','in':0,'inout':1},
    'logical*1':{'pref':'1','in':0,'inout':1},
    'logical*2':{'pref':'s','in':0,'inout':1},
    'double precision':{'pref':'d','in':4.7,'inout':5.3},
    'complex':{'pref':'F','in':4.7-2j,'inout':5.3+4j},
    'double complex':{'pref':'D','in':4.7-2j,'inout':5.3+4j},
    'complex*16':{'pref':'D','in':4.7-2j,'inout':5.3+4j},
    }
#mmap={'double complex':{'pref':'D','in':4.7-2j,'inout':5.3+4j}}
for k in mmap.keys():
    if string.find(k,'logical')>=0:
        mmap[k]['inout_res']=not mmap[k]['inout']
        mmap[k]['out_res']=mmap[k]['in']
        mmap[k]['ret']=1
    else:
        mmap[k]['inout_res']=mmap[k]['inout']+mmap[k]['inout']
        mmap[k]['out_res']=mmap[k]['in']
        mmap[k]['ret']=mmap[k]['inout_res']+mmap[k]['in']
        if string.find(k,'integer')>=0:
            mmap[k]['inout_res'] = int(mmap[k]['inout_res'])
            mmap[k]['out_res'] = int(mmap[k]['out_res'])
            mmap[k]['ret'] = int(mmap[k]['ret'])
# Begin
ff=open(ffname,'w')
hf=open(hfname,'w')
py=open(pyname,'w')
hf.write('!%f90\npythonmodule iotest\n')
py.write("""\
import sys
from Numeric import array
put=sys.stdout.write
put('import iotest:')
import iotest
#except: put('unsuccesful\\n');sys.exit()
#put('successful\\n')
""")
# Body
i=0
for k in mmap.keys():
    i=i+1
    ft=k
    p=`i`
    if string.find(k,'logical')>=0:
        ff.write("""
      function f%sio(f%sin,f%sout,f%sinout)
        %s f%sin,f%sout,f%sinout,f%sio
        f%sout = f%sin
        f%sinout = .not.f%sinout
        f%sio = .TRUE.
      end"""%(p,p,p,p,ft,p,p,p,p,p,p,p,p,p))
    else:
        ff.write("""
      function f%sio(f%sin,f%sout,f%sinout)
        %s f%sin,f%sout,f%sinout,f%sio
        f%sout = f%sin
        f%sinout = f%sinout + f%sinout
        f%sio = f%sinout + f%sin
      end"""%(p,p,p,p,ft,p,p,p,p,p,p,p,p,p,p,p,p))
    hf.write("""\
    interface
        function f%sio(f%sin,f%sout,f%sinout)
             %s f%sio
             %s intent(in) :: f%sin
             %s intent(in,out):: f%sout
             %s intent(inout):: f%sinout
        end
    end
"""%(p,p,p,p,ft,p,ft,p,ft,p,ft,p))
    py.write("""
i = %s
o = 0.0
io = array(%s,'%s')
print '\\n(%s)',iotest.f%sio.__doc__
if 1:
\tr,o=iotest.f%sio(i,o,io)
\tif r != %s:
\t\tprint 'FAILURE',
\telse:
\t\tprint 'SUCCESS',
\tprint '(%s:out)',`r`,'==',`%s`,'(expected)'
\tif o != %s:
\t\tprint 'FAILURE',
\telse:
\t\tprint 'SUCCESS',
\tprint '(%s:in,out)',`o`,'==',`%s`,'(expected)'
\tif io != %s:
\t\tprint 'FAILURE',
\telse:
\t\tprint 'SUCCESS',
\tprint '(%s:inout)',`io`,'==',`%s`,'(expected)'
print 'ok'
""" % (mmap[k]['in'],mmap[k]['inout'],mmap[k]['pref'],k,p,p,
       mmap[k]['ret'],k,mmap[k]['ret'],
       mmap[k]['out_res'],k,mmap[k]['out_res'],
       mmap[k]['inout_res'],k,mmap[k]['inout_res'],
       ))
# Close up
hf.write('end pythonmodule iotest')
ff.close()
hf.close()
py.close()

















