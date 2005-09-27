#!/usr/bin/env python
"""

Test arguments against all combinations of intent attributes.

Copyright 2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2000/09/17 16:10:27 $
Pearu Peterson
"""

__version__ = "$Revision: 1.3 $"[10:-1]

import commands,os,sys,string

testmod = 0
if 'f90mod' in sys.argv:
    testmod = 1 # test F90 module support
    print 'Testing F90 module support:'
kinds={}
kinds['real']=['','*8']
kinds['integer']=['','*1','*2','*8']
kinds['complex']=['','*16']
kinds['logical']=['','*1','*2','*8']
kinds['character']=['','*2','*5','*(*)']
finitexpr={}
pyretexpr={}
pyinitexpr={}
py0initexpr={}

finitexpr['real'] = '1.23'
pyretexpr['real'] = '1.23'
pyinitexpr['real'] = '32.1'
py0initexpr['real'] = 'array(32.1,"f")'
py0initexpr['real*8'] = 'array(32.1,"d")'
py0initexpr['real*16'] = 'array(32.1,"d")'

finitexpr['integer'] = '123'
pyretexpr['integer'] = '123'
pyinitexpr['integer'] = '321'
py0initexpr['integer'] = 'array(321,"i")'

py0initexpr['integer*2'] = 'array(321,"s")'
py0initexpr['integer*4'] = 'array(321,"l")'

finitexpr['integer*1'] = '65'
pyretexpr['integer*1'] = '65'
pyinitexpr['integer*1'] = '56'
py0initexpr['integer*1'] = 'array(56,"1")'

finitexpr['integer*8'] = '20\n      a = a*222222222'
pyretexpr['integer*8'] = '20L*222222222L'
pyinitexpr['integer*8'] = '20L*222222223L'
py0initexpr['integer*8'] = 'array(20L*222222223L)'

finitexpr['complex'] = '(1.0,23.0)'
pyretexpr['complex'] = '1+23j'
pyinitexpr['complex'] = '23+1j'
py0initexpr['complex'] = 'array(23+1j,"F")'
py0initexpr['complex*16'] = 'array(23+1j,"D")'
py0initexpr['complex*32'] = 'array(23+1j,"D")'

finitexpr['logical'] = '.TRUE.'
pyretexpr['logical'] = '1'
pyinitexpr['logical'] = '0'
py0initexpr['logical'] = 'array(0,"1")'

finitexpr['character'] = '"AbC"'
pyretexpr['character'] = '"AbC"'
pyinitexpr['character'] = '"cBa"'
py0initexpr['character'] = 'array("cBa")'

for k in kinds.keys():
    for t in finitexpr[k]:
        for kt in kinds[k]:
            if not finitexpr.has_key(k+kt):
                finitexpr[k+kt] = finitexpr[k]
            if not pyretexpr.has_key(k+kt):
                pyretexpr[k+kt] = pyretexpr[k]
            if not pyinitexpr.has_key(k+kt):
                pyinitexpr[k+kt] = pyinitexpr[k]
            if not py0initexpr.has_key(k+kt):
                py0initexpr[k+kt] = py0initexpr[k]

def run(com):
    global mess
    s,o=commands.getstatusoutput(com)
    if s:
        mess('\n'+o+'\n'+80*'='+'\n')
        sys.stdout.write('failed\n')
        sys.stdout.flush()
    return s,o

def gen(typespec):
    global mess
    suf={0:'',1:'2'}[testmod]
    t = '      '
    tp= 'Cf2py '
    fcode = ['']
    pycode=['']
    def addpy(line,pycode=pycode):
        pycode[0] = '%s%s\n'%(pycode[0],line)
    def add(line,fcode=fcode):
        fcode[0] = '%s%s\n'%(fcode[0],line)

    sys.stdout.write('Testing %s foo() ...'%(typespec))
    mess('Testing %s foo() ...'%(typespec))

    if testmod:
        add(t+'module genfun')
        add(t+'contains')
    add(t+'function foo()')
    add(t+'%s a,foo'%(typespec))
    add(t+'a = '+finitexpr[typespec])
    add(t+'write(*,*) "Fa=>",a,"<="')
    add(t+'foo = a')
    if testmod:
        add(t+'end function foo')
        add(t+'end module genfun')
    else:
        add(t+'end')

    addpy('print "genfuntest: %s : foo"'%(typespec))
    addpy('from Numeric import array,ArrayType')
    addpy('from types import *')
    addpy('import string')
    addpy("""\
def eq(a,b,t=''):
    if '*' in t: typ = t[:string.index(t,'*')]
    else: typ = t
    if typ=='logical':
        ret = (not a)==(not b)
    elif typ in ['real','complex']:
        ret = abs(a-b)<1e-5
    elif typ == 'character':
        if type(a)==ArrayType:
            a=a.tostring()
        i=string.find(a,'\\000')
        if i>=0: a=a[:i]
        l=min(len(a),len(b))
        ret = l and (('%s'%a)[:l]==('%s'%b)[:l]) # in case 'character*(*) out' works elsewhere
    else:
        ret = (a==b)
    if not ret:
        print 'Comparison between %s %s and %s failed'%(`t`,`a`,`b`)
    return ret
""")

    addpy('import genfun%s'%suf)
    if testmod:
        addpy('genfun=genfun%s.genfun'%(suf))
    addpy('print genfun.foo.__doc__')

    addpy('a = '+py0initexpr[typespec])
    addpy('r = genfun.foo()')
    addpy('print "Pyr=>%s<="%(r)')
    addpy('assert eq(r,%s,%s)'%(pyretexpr[typespec],`typespec`))

    fcode = fcode[0]
    pycode = pycode[0]

    f = open('genfuntest.f','w')
    f.write(fcode)
    f.close()
    f = open('genfunruntest.py','w')
    f.write(pycode)
    f.close()
    s,o=run('../../f2py2e.py genfuntest.f -m genfun%s -h genfun%s.pyf --overwrite-makefile --overwrite-signature'%(suf,suf))
    if s: return 1
    if testmod:
        s,o=run('gmake -f Makefile-genfun%s clean test'%suf)
    else:
        s,o=run('gmake -f Makefile-genfun%s distclean test'%suf)
    if s: return 1
    s,o=run('python genfunruntest.py')
    if s: return 1
    s,o=run('gmake -f Makefile-genfun%s distclean'%suf)
    s,o=run('rm -f genfuntest.f genfunruntest.py genfun%s.pyf'%(suf))
    sys.stdout.write('ok\n')
    mess('ok\n')
    sys.stdout.flush()
    return 0

if __name__ == "__main__":
    f=open('genfuntests.log','w')
    mess = f.write
    failed=[]
    for t in kinds.keys():
        #if not t=='integer': continue
        for k in kinds[t]:
            #if not t+k=='character*(*)': continue
            if gen(t+k):
                failed.append('%s foo()'%(t+k))
    if failed:
        print 'Failures:\n\t'+string.join(failed,'\n\t')
    else:
        print 'No failures.'
    print 'See genruntests.log for error details'
    f.close()
