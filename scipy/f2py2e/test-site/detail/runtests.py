#!/usr/bin/env python
"""

Copyright 2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2001/12/17 18:11:12 $
Pearu Peterson
"""

__version__ = "$Revision: 1.12 $"[10:-1]

import string,commands,sys,os

sys.stderr=open('runtests.log','w')
f2py=sys.executable+' '+os.path.abspath(os.path.join('..','..','f2py2e.py'))
devnull='> /dev/null'
devnull=''

tests = []

import simpletest, argpasstest, returntest, cbreturntest
import argpasstest
tests=tests+simpletest.tests
tests=tests+returntest.tests
tests=tests+argpasstest.tests
tests=tests+cbreturntest.tests

def cleandir(path):
    """Remove directory and its contents recursively."""
    path = os.path.abspath(path)
    cdir = os.getcwd()
    os.chdir(path)
    for p in map(os.path.abspath,os.listdir(path)):
        if os.path.isdir(p):
            cleandir(p)
        else:
            os.remove(p)
    os.chdir(cdir)
    os.rmdir(path)

testbasename='tmptest'
id=0
needcflags=[]
unsuccessfultests=[]
curdir = os.path.abspath(os.getcwd())
for test in tests:
    id=id+1

    os.chdir(curdir)
    wd = 'tmp_test%s'%id
    if os.path.isdir(wd):
        cleandir(wd)
    os.mkdir(wd)
    wd = os.path.abspath(wd)
    os.chdir(wd)

    if test.has_key('depends'):
        fl=0
        for d in test['depends']:
            if d in unsuccessfultests:
                fl=1;break
        if fl: continue
    sys.stdout.write('%s (id=%s) ... '%(test['name'],id))
    sys.stdout.flush()
    ffiles=[]
    pyfiles=[]
    f2pyflags=[]
    trydefine=['']
    testid = ''
    if test.has_key('trydefine'):
        trydefine=trydefine+test['trydefine']
    if test.has_key('f2pyflags'):
        f2pyflags=f2pyflags+test['f2pyflags']
    if test.has_key('f'):
        ffiles.append(testbasename+`id`+'.f')
        f=open(ffiles[-1],'w')
        f.write(string.replace(test['f'],'f2pytest','f2pytest%s'%id))
        f.close()
    if test.has_key('py'):
        pyfiles.append(testbasename+`id`+'.py')
        f=open(pyfiles[-1],'w')
        f.write(string.replace(test['py'],'f2pytest','f2pytest%s'%id))
        f.close()
    if test.has_key('id'): testid=test['id']
    #trydefine.reverse()
    for d in trydefine:
        if d:
            sys.stdout.write('\tTrying with %s ... '%(d))
            sys.stdout.flush()
        coms=[]

        cflags=string.join(needcflags+[d],' ')

        coms.append('%s -m f2pytest%s %s --setup --overwrite-setup -makefile CFLAGS="%s" %s %s'%(f2py,id,string.join(ffiles,' '),cflags,string.join(f2pyflags,' '),devnull))
        coms.append('%s setup_f2pytest%s.py build --build-platlib . '%(\
            sys.executable,
            id))
        coms.append(sys.executable+' '+string.join(pyfiles,' '))
        success=1
        fl = 1

        for c in coms:
            if os.name=='posix':
                status,result=commands.getstatusoutput(c)
            else:
                status = os.system(c)
                result = '???'
            if fl:
                #print result
                fl=0
            if not status==0:
                sys.stderr.write('%s ... '%test['name'])
                sys.stderr.write('Command %s failed:\n%s\n'%(`c`,result))
                sys.stderr.flush()
                success=0
                break
            elif result[:2]=='ok':
                success=2
        if success:
            sys.stdout.write(result)
            if os.path.exists('core'):
                sys.stdout.write('(core dump)')
            sys.stdout.write('\n')
            sys.stdout.flush()
            if success==2:
                cleandir(wd)
                #os.system('gmake -f Makefile-f2pytest%s distclean'%(id))
                #os.system('rm -f tmptest%s.{f,py} Makefile-f2pytest%s*'%(id,id))
                if d:
                    needcflags.append(d)
                if testid and testid in unsuccessfultests:
                    del unsuccessfultests[unsuccessfultests.index(testid)]
                break
        else:
            if testid and testid not in unsuccessfultests:
                unsuccessfultests.append(testid)
            sys.stdout.write('failed')
            if os.path.exists('core'):
                sys.stdout.write('(core dump)')
                sys.stderr.write('(core dump)\n')
            sys.stdout.write('\n')
            sys.stdout.flush()
            sys.stderr.write(79*'='+'\n')
#os.system('%s --show-compilers'%(f2py))
if needcflags:
    sys.stdout.write('You must use CFLAGS="%s" when compiling Python/CAPI modules.\n'%(string.join(needcflags,' ')))
sys.stdout.write('See test-site/detail/runtests.log for error messages.\n')
sys.stderr.close()
print unsuccessfultests

