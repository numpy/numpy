#!/usr/bin/env python
"""

Rules for building C/API module with f2py2e.

Copyright 1999,2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2004/11/26 11:13:06 $
Pearu Peterson
"""

__version__ = "$Revision: 1.16 $"[10:-1]

f2py_version='See `f2py -v`'

import pprint,copy
import sys,string,time,types,copy
errmess=sys.stderr.write
outmess=sys.stdout.write
show=pprint.pprint

from auxfuncs import *
def var2fixfortran(vars,a,fa=None,f90mode=None):
    if fa is None:
        fa = a
    if not vars.has_key(a):
        show(vars)
        outmess('var2fixfortran: No definition for argument "%s".\n'%a)
        return ''
    if not vars[a].has_key('typespec'):
        show(vars[a])
        outmess('var2fixfortran: No typespec for argument "%s".\n'%a)
        return ''
    vardef=vars[a]['typespec']
    if vardef=='type' and vars[a].has_key('typename'):
        vardef='%s(%s)'%(vardef,vars[a]['typename'])
    selector={}
    lk = ''
    if vars[a].has_key('kindselector'):
        selector=vars[a]['kindselector']
        lk = 'kind'
    elif vars[a].has_key('charselector'):
        selector=vars[a]['charselector']
        lk = 'len'
    if selector.has_key('*'):
        if f90mode:
            if selector['*'] in ['*',':','(*)']:
                vardef='%s(len=*)'%(vardef)
            else:
                vardef='%s(%s=%s)'%(vardef,lk,selector['*'])
        else:
            if selector['*'] in ['*',':']:
                vardef='%s*(%s)'%(vardef,selector['*'])
            else:
                vardef='%s*%s'%(vardef,selector['*'])
    else:
        if selector.has_key('len'):
            vardef='%s(len=%s'%(vardef,selector['len'])
            if selector.has_key('kind'):
                vardef='%s,kind=%s)'%(vardef,selector['kind'])
            else:
                vardef='%s)'%(vardef)
        elif selector.has_key('kind'):
            vardef='%s(kind=%s)'%(vardef,selector['kind'])

    vardef='%s %s'%(vardef,fa)
    if vars[a].has_key('dimension'):
        vardef='%s(%s)'%(vardef,string.join(vars[a]['dimension'],','))
    return vardef

def createfuncwrapper(rout,signature=0):
    assert isfunction(rout)
    ret = ['']
    def add(line,ret=ret):
        ret[0] = '%s\n      %s'%(ret[0],line)
    name = rout['name']
    fortranname = getfortranname(rout)
    f90mode = ismoduleroutine(rout)
    newname = '%sf2pywrap'%(name)
    vars = rout['vars']
    if not vars.has_key(newname):
        vars[newname] = vars[name]
        args = [newname]+rout['args'][1:]
    else:
        args = [newname]+rout['args']

    l = var2fixfortran(vars,name,newname,f90mode)
    return_char_star = 0
    if l[:13]=='character*(*)':
        return_char_star = 1
        if f90mode: l = 'character(len=10)'+l[13:]
        else: l = 'character*10'+l[13:]
        charselect = vars[name]['charselector']
        if charselect.get('*','')=='(*)':
            charselect['*'] = '10'
    if f90mode:
        sargs = string.join(args,', ')
        add('subroutine f2pywrap_%s_%s (%s)'%(rout['modulename'],name,sargs))
        if not signature:
            add('use %s, only : %s'%(rout['modulename'],fortranname))
    else:
        add('subroutine f2pywrap%s (%s)'%(name,string.join(args,', ')))
        add('external %s'%(fortranname))
        #if not return_char_star:
        l = l + ', '+fortranname
    args = args[1:]
    dumped_args = []
    for a in args:
        if isexternal(vars[a]):
            add('external %s'%(a))
            dumped_args.append(a)
    for a in args:
        if a in dumped_args: continue
        if isscalar(vars[a]):
            add(var2fixfortran(vars,a,f90mode=f90mode))
            dumped_args.append(a)
    for a in args:
        if a in dumped_args: continue
        add(var2fixfortran(vars,a,f90mode=f90mode))

    add(l)

    if not signature:
        if islogicalfunction(rout):
            add('%s = .not.(.not.%s(%s))'%(newname,fortranname,string.join(args,', ')))
        else:
            add('%s = %s(%s)'%(newname,fortranname,string.join(args,', ')))
    if f90mode:
        add('end subroutine f2pywrap_%s_%s'%(rout['modulename'],name))
    else:
        add('end')
    #print '**'*10
    #print ret[0]
    #print '**'*10
    return ret[0]

def assubr(rout):
    if not isfunction_wrap(rout): return rout,''
    fortranname = getfortranname(rout)
    name = rout['name']
    outmess('\t\tCreating wrapper for Fortran function "%s"("%s")...\n'%(name,fortranname))
    rout = copy.copy(rout)
    fname = name
    rname = fname
    if rout.has_key('result'):
        rname = rout['result']
        rout['vars'][fname]=rout['vars'][rname]
    fvar = rout['vars'][fname]
    if not isintent_out(fvar):
        if not fvar.has_key('intent'): fvar['intent']=[]
        fvar['intent'].append('out')
        flag=1
        for i in fvar['intent']:
            if i.startswith('out='):
                flag = 0
                break
        if flag:
            fvar['intent'].append('out=%s' % (rname))

    rout['args'] = [fname] + rout['args']
    return rout,createfuncwrapper(rout)
