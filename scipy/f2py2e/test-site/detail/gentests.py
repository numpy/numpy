#!/usr/bin/env python
"""

Test arguments against all combinations of intent attributes.

Copyright 2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2001/12/17 18:11:12 $
Pearu Peterson
"""

__version__ = "$Revision: 1.8 $"[10:-1]

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
    if os.name=='posix':
        s,o=commands.getstatusoutput(com)
    else:
        s,o = os.system(com),'???'
    if s:
        mess('\n'+o+'\n'+80*'='+'\n')
        sys.stdout.write('failed\n')
        sys.stdout.flush()
    return s,o
saveintents = {}
def isintent_in(intent):
    if not saveintents.has_key(intent): 
        saveintents[intent] = map(string.strip,string.split(intent,','))
    return 'in' in saveintents[intent] and not isintent_inout(intent)
def isintent_inout(intent):
    if not saveintents.has_key(intent): 
        saveintents[intent] = map(string.strip,string.split(intent,','))
    return ('inout' in saveintents[intent] or 'outin' in saveintents[intent]) and not isintent_hide(intent)
isintent_outin = isintent_inout
def isintent_out(intent):
    if not saveintents.has_key(intent): 
        saveintents[intent] = map(string.strip,string.split(intent,','))
    return 'out' in saveintents[intent]
def isintent_out_only(intent):
    return isintent_out(intent) and (not (isintent_in(intent) or isintent_inout(intent)))
def isintent_hide(intent):
    if not saveintents.has_key(intent): 
        saveintents[intent] = map(string.strip,string.split(intent,','))
    return 'hide' in saveintents[intent]

def gen(typespec,intent='',dims=''):
    suf={0:'',1:'2'}[testmod]
    global mess,saveintents
    saveintents = {}
    intents=map(string.strip,string.split(intent,','))
    t = '      '
    tp= 'Cf2py '
    fcode = ['']
    pycode=['']
    def addpy(line,pycode=pycode):
        pycode[0] = '%s%s\n'%(pycode[0],line)
    def add(line,fcode=fcode):
        fcode[0] = '%s%s\n'%(fcode[0],line)
    attr = ''
    if intent:
        attr = '%s, intent(%s)'%(attr,intent)
    sys.stdout.write('Testing %s%s ...'%(typespec,attr))
    mess('Testing %s%s ...'%(typespec,attr))

    if testmod:
        add(t+'module gen')
        add(t+'contains')
    add(t+'subroutine foo(a)')
    add(t+'%s a'%(typespec))
    
    addpy('print "gentest: %s%s : a"'%(typespec,attr))
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
    elif typ == 'integer':
        if type(a)==ArrayType:
            a=a[0]
        ret = (a==b)
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
    addpy('import gen%s'%suf)
    if testmod:
        addpy('gen=gen%s.gen'%suf)
    addpy('print gen.foo.__doc__')

    if intent:
        add(tp+'intent(%s) a'%(intent))
    if isintent_hide(intent) and not isintent_out(intent):
        addpy('r = gen.foo()')
        add(t+'a = '+finitexpr[typespec])
    elif isintent_out_only(intent):
        addpy('r = gen.foo()')
        add(t+'a = '+finitexpr[typespec])
    else:
        if isintent_in(intent) or not string.strip(intent):
            addpy('a = '+pyinitexpr[typespec])
        elif isintent_inout(intent):
            addpy('a = '+py0initexpr[typespec])
            add(t+'write(*,*) "Fa=>",a,"<="')
            add(t+'a = '+finitexpr[typespec])
        else:
            raise 'Unexpexted intent(%s)'%intent
        addpy('print "Pya=>%s<="%(a)')
        addpy('r = gen.foo(a)')
    add(t+'write(*,*) "Fa=>",a,"<="')
    if testmod:
        add(t+'end subroutine foo')
        add(t+'end module gen')
    else: add(t+'end')
    if isintent_out(intent):
        addpy('print "Pyr=>%s<="%(r)')
        if isintent_out_only(intent):
            addpy('assert eq(r,%s,%s)'%(pyretexpr[typespec],`typespec`))
        else:
            if isintent_in(intent) or not string.strip(intent):
                addpy('assert eq(r,%s,%s)'%(pyinitexpr[typespec],`typespec`))
            else:
                addpy('assert eq(r,%s,%s)'%(pyretexpr[typespec],`typespec`))
    if isintent_inout(intent):
        addpy('print "Pya=>%s<="%(a)')
        addpy('assert eq(a,%s,%s)'%(pyretexpr[typespec],`typespec`))
    fcode = fcode[0]
    pycode = pycode[0]

    f = open('gentest.f','w')
    f.write(fcode)
    f.close()
    f = open('genruntest.py','w')
    f.write(pycode)
    f.close()
    po=''
    s,o=run('../../f2py2e.py gentest.f -m gen%s -h gen%s.pyf --overwrite-makefile --overwrite-signature'%(suf,suf))
    if s: return 1
    po=po+o
    if testmod:
        s,o=run('gmake -f Makefile-gen%s clean test'%(suf))
    else:
        s,o=run('gmake -f Makefile-gen%s clean test'%(suf))
    if s: mess(po);return 1
    po=po+o
    s,o=run('python genruntest.py')
    if s: mess(po);return 1
    po=po+o
    s,o=run('gmake -f Makefile-gen%s distclean'%(suf))
    s,o=run('rm -f gentest.f genruntest.py gen%s.pyf'%suf)
    sys.stdout.write('ok\n')
    mess('ok\n')
    sys.stdout.flush()
    return 0

if __name__ == "__main__":
    f=open('gentests.log','w')
    mess = f.write
    #mess = sys.stderr.write
    failed=[]
    for t in kinds.keys():
        #if not t=='integer': continue
        for k in kinds[t]:
            #if not k=='*2': continue
            for intent in ['in','in,out','out','inout','inout,out','hide','hide,out']:
                if gen(t+k,intent):
                    failed.append('%s, intent(%s)'%(t+k,intent))
    if failed:
        print 'Failures:\n\t'+string.join(failed,'\n\t')
    else:
        print 'No failures.'
    print 'See gentests.log for error details'
    f.close()
