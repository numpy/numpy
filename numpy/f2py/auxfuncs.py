#!/usr/bin/env python
"""

Auxiliary functions for f2py2e.

Copyright 1999,2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy (BSD style) LICENSE.


NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2005/07/24 19:01:55 $
Pearu Peterson
"""
__version__ = "$Revision: 1.65 $"[10:-1]

import __version__
f2py_version = __version__.version

import pprint
import sys,string,time,types,os
import cfuncs


errmess=sys.stderr.write
#outmess=sys.stdout.write
show=pprint.pprint

options={}
debugoptions=[]
wrapfuncs = 1

def outmess(t):
    if options.get('verbose',1):
        sys.stdout.write(t)

def debugcapi(var): return 'capi' in debugoptions
def _isstring(var):
    return var.has_key('typespec') and var['typespec']=='character' and (not isexternal(var))
def isstring(var):
    return _isstring(var) and not isarray(var)
def ischaracter(var):
    return isstring(var) and not (var.has_key('charselector'))
def isstringarray(var):
    return isarray(var) and _isstring(var)
def isarrayofstrings(var):
    # leaving out '*' for now so that
    # `character*(*) a(m)` and `character a(m,*)`
    # are treated differently. Luckily `character**` is illegal.
    return isstringarray(var) and var['dimension'][-1]=='(*)'
def isarray(var): return var.has_key('dimension') and (not isexternal(var))
def isscalar(var): return not (isarray(var) or isstring(var) or isexternal(var))
def iscomplex(var):
    return isscalar(var) and var.get('typespec') in ['complex','double complex']
def islogical(var):
    return isscalar(var) and var.get('typespec')=='logical'
def isinteger(var):
    return isscalar(var) and var.get('typespec')=='integer'
def isreal(var):
    return isscalar(var) and var.get('typespec')=='real'
def get_kind(var):
    try: return var['kindselector']['*']
    except KeyError:
        try: return var['kindselector']['kind']
        except KeyError: pass
def islong_long(var):
    if not isscalar(var): return 0
    if var.get('typespec') not in ['integer','logical']: return 0
    return get_kind(var)=='8'
def isunsigned_char(var):
    if not isscalar(var): return 0
    if var.get('typespec') != 'integer': return 0
    return get_kind(var)=='-1'
def isunsigned_short(var):
    if not isscalar(var): return 0
    if var.get('typespec') != 'integer': return 0
    return get_kind(var)=='-2'
def isunsigned(var):
    if not isscalar(var): return 0
    if var.get('typespec') != 'integer': return 0
    return get_kind(var)=='-4'
def isunsigned_long_long(var):
    if not isscalar(var): return 0
    if var.get('typespec') != 'integer': return 0
    return get_kind(var)=='-8'
def isdouble(var):
    if not isscalar(var): return 0
    if not var.get('typespec')=='real': return 0
    return get_kind(var)=='8'
def islong_double(var):
    if not isscalar(var): return 0
    if not var.get('typespec')=='real': return 0
    return get_kind(var)=='16'
def islong_complex(var):
    if not iscomplex(var): return 0
    return get_kind(var)=='32'

def iscomplexarray(var): return isarray(var) and var.get('typespec') in ['complex','double complex']
def isint1array(var): return isarray(var) and var.get('typespec')=='integer' \
    and get_kind(var)=='1'
def isunsigned_chararray(var): return isarray(var) and var.get('typespec')=='integer' and get_kind(var)=='-1'
def isunsigned_shortarray(var): return isarray(var) and var.get('typespec')=='integer' and get_kind(var)=='-2'
def isunsignedarray(var): return isarray(var) and var.get('typespec')=='integer' and get_kind(var)=='-4'
def isunsigned_long_longarray(var): return isarray(var) and var.get('typespec')=='integer' and get_kind(var)=='-8'
def isallocatable(var):
    return var.has_key('attrspec') and 'allocatable' in var['attrspec']
def ismutable(var): return not (not var.has_key('dimension') or isstring(var))
def ismoduleroutine(rout): return rout.has_key('modulename')
def ismodule(rout): return (rout.has_key('block') and 'module'==rout['block'])
def isfunction(rout): return (rout.has_key('block') and 'function'==rout['block'])
#def isfunction_wrap(rout):
#    return wrapfuncs and (iscomplexfunction(rout) or isstringfunction(rout)) and (not isexternal(rout))
def isfunction_wrap(rout):
    if isintent_c(rout): return 0
    return wrapfuncs and isfunction(rout) and (not isexternal(rout))
def issubroutine(rout): return (rout.has_key('block') and 'subroutine'==rout['block'])
def isroutine(rout): return isfunction(rout) or issubroutine(rout)
def islogicalfunction(rout):
    if not isfunction(rout): return 0
    if rout.has_key('result'): a=rout['result']
    else: a=rout['name']
    if rout['vars'].has_key(a): return islogical(rout['vars'][a])
    return 0
def islong_longfunction(rout):
    if not isfunction(rout): return 0
    if rout.has_key('result'): a=rout['result']
    else: a=rout['name']
    if rout['vars'].has_key(a): return islong_long(rout['vars'][a])
    return 0
def islong_doublefunction(rout):
    if not isfunction(rout): return 0
    if rout.has_key('result'): a=rout['result']
    else: a=rout['name']
    if rout['vars'].has_key(a): return islong_double(rout['vars'][a])
    return 0
def iscomplexfunction(rout):
    if not isfunction(rout): return 0
    if rout.has_key('result'): a=rout['result']
    else: a=rout['name']
    if rout['vars'].has_key(a): return iscomplex(rout['vars'][a])
    return 0
def iscomplexfunction_warn(rout):
    if iscomplexfunction(rout):
        outmess("""\
    **************************************************************
        Warning: code with a function returning complex value
        may not work correctly with your Fortran compiler.
        Run the following test before using it in your applications:
        $(f2py install dir)/test-site/{b/runme_scalar,e/runme}
        When using GNU gcc/g77 compilers, codes should work correctly.
    **************************************************************\n""")
        return 1
    return 0
def isstringfunction(rout):
    if not isfunction(rout): return 0
    if rout.has_key('result'): a=rout['result']
    else: a=rout['name']
    if rout['vars'].has_key(a): return isstring(rout['vars'][a])
    return 0
def hasexternals(rout): return rout.has_key('externals') and rout['externals']
def isthreadsafe(rout): return rout.has_key('f2pyenhancements') and rout['f2pyenhancements'].has_key('threadsafe')
def hasvariables(rout): return rout.has_key('vars') and rout['vars']
def isoptional(var): return (var.has_key('attrspec') and 'optional' in var['attrspec'] and 'required' not in var['attrspec']) and isintent_nothide(var)
def isexternal(var): return (var.has_key('attrspec') and 'external' in var['attrspec'])
def isrequired(var): return not isoptional(var) and isintent_nothide(var)
def isintent_in(var):
    if not var.has_key('intent'): return 1
    if 'hide' in var['intent']: return 0
    if 'inplace' in var['intent']: return 0
    if 'in' in var['intent']: return 1
    if 'out' in var['intent']: return 0
    if 'inout' in var['intent']: return 0
    if 'outin' in var['intent']: return 0
    return 1
def isintent_inout(var): return var.has_key('intent') and ('inout' in var['intent'] or 'outin' in var['intent']) and 'in' not in var['intent'] and 'hide' not in var['intent'] and 'inplace' not in var['intent']
def isintent_out(var):
    return 'out' in var.get('intent',[])
def isintent_hide(var): return (var.has_key('intent') and ('hide' in var['intent'] or ('out' in var['intent'] and 'in' not in var['intent'] and (not l_or(isintent_inout,isintent_inplace)(var)))))
def isintent_nothide(var): return not isintent_hide(var)
def isintent_c(var):
    return 'c' in var.get('intent',[])
# def isintent_f(var):
#     return not isintent_c(var)
def isintent_cache(var):
    return 'cache' in var.get('intent',[])
def isintent_copy(var):
    return 'copy' in var.get('intent',[])
def isintent_overwrite(var):
    return 'overwrite' in var.get('intent',[])
def isintent_callback(var):
    return 'callback' in var.get('intent',[])
def isintent_inplace(var):
    return 'inplace' in var.get('intent',[])
def isintent_aux(var):
    return 'aux' in var.get('intent',[])

isintent_dict = {isintent_in:'INTENT_IN',isintent_inout:'INTENT_INOUT',
                 isintent_out:'INTENT_OUT',isintent_hide:'INTENT_HIDE',
                 isintent_cache:'INTENT_CACHE',
                 isintent_c:'INTENT_C',isoptional:'OPTIONAL',
                 isintent_inplace:'INTENT_INPLACE'
                 }

def isprivate(var):
    return var.has_key('attrspec') and 'private' in var['attrspec']

def hasinitvalue(var): return var.has_key('=')
def hasinitvalueasstring(var):
    if not hasinitvalue(var): return 0
    return var['='][0] in ['"',"'"]
def hasnote(var):
    return var.has_key('note')
def hasresultnote(rout):
    if not isfunction(rout): return 0
    if rout.has_key('result'): a=rout['result']
    else: a=rout['name']
    if rout['vars'].has_key(a): return hasnote(rout['vars'][a])
    return 0
def hascommon(rout):
    return rout.has_key('common')
def containscommon(rout):
    if hascommon(rout): return 1
    if hasbody(rout):
        for b in rout['body']:
            if containscommon(b): return 1
    return 0
def containsmodule(block):
    if ismodule(block): return 1
    if not hasbody(block): return 0
    for b in block['body']:
        if containsmodule(b): return 1
    return 0
def hasbody(rout):
    return rout.has_key('body')
def hascallstatement(rout):
    return getcallstatement(rout) is not None

def istrue(var): return 1
def isfalse(var): return 0

class F2PYError(Exception):
    pass

class throw_error:
    def __init__(self,mess):
        self.mess = mess
    def __call__(self,var):
        mess = '\n\n  var = %s\n  Message: %s\n' % (var,self.mess)
        raise F2PYError,mess

def l_and(*f):
    l,l2='lambda v',[]
    for i in range(len(f)):
        l='%s,f%d=f[%d]'%(l,i,i)
        l2.append('f%d(v)'%(i))
    return eval('%s:%s'%(l,string.join(l2,' and ')))
def l_or(*f):
    l,l2='lambda v',[]
    for i in range(len(f)):
        l='%s,f%d=f[%d]'%(l,i,i)
        l2.append('f%d(v)'%(i))
    return eval('%s:%s'%(l,string.join(l2,' or ')))
def l_not(f):
    return eval('lambda v,f=f:not f(v)')

def isdummyroutine(rout):
    try:
        return rout['f2pyenhancements']['fortranname']==''
    except KeyError:
        return 0

def getfortranname(rout):
    try:
        name = rout['f2pyenhancements']['fortranname']
        if name=='':
            raise KeyError
        if not name:
            errmess('Failed to use fortranname from %s\n'%(rout['f2pyenhancements']))
            raise KeyError
    except KeyError:
        name = rout['name']
    return name

def getmultilineblock(rout,blockname,comment=1,counter=0):
    try:
        r = rout['f2pyenhancements'].get(blockname)
    except KeyError:
        return
    if not r: return
    if counter>0 and type(r) is type(''):
        return
    if type(r) is type([]):
        if counter>=len(r): return
        r = r[counter]
    if r[:3]=="'''":
        if comment:
            r = '\t/* start ' + blockname + ' multiline ('+`counter`+') */\n' + r[3:]
        else:
            r = r[3:]
        if r[-3:]=="'''":
            if comment:
                r = r[:-3] + '\n\t/* end multiline ('+`counter`+')*/'
            else:
                r = r[:-3]
        else:
            errmess("%s multiline block should end with `'''`: %s\n" \
                    % (blockname,repr(r)))
    return r

def getcallstatement(rout):
    return getmultilineblock(rout,'callstatement')

def getcallprotoargument(rout,cb_map={}):
    r = getmultilineblock(rout,'callprotoargument',comment=0)
    if r: return r
    if hascallstatement(rout):
        outmess('warning: callstatement is defined without callprotoargument\n')
        return
    from capi_maps import getctype
    arg_types,arg_types2 = [],[]
    if l_and(isstringfunction,l_not(isfunction_wrap))(rout):
        arg_types.extend(['char*','size_t'])
    for n in rout['args']:
        var = rout['vars'][n]
        if isintent_callback(var):
            continue
        if cb_map.has_key(n):
            ctype = cb_map[n]+'_typedef'
        else:
            ctype = getctype(var)
            if l_and(isintent_c,l_or(isscalar,iscomplex))(var):
                pass
            elif isstring(var):
                pass
                #ctype = 'void*'
            else:
                ctype = ctype+'*'
            if isstring(var) or isarrayofstrings(var):
                arg_types2.append('size_t')
        arg_types.append(ctype)

    proto_args = string.join(arg_types+arg_types2,',')
    if not proto_args:
        proto_args = 'void'
    #print proto_args
    return proto_args

def getusercode(rout):
    return getmultilineblock(rout,'usercode')
def getusercode1(rout):
    return getmultilineblock(rout,'usercode',counter=1)

def getpymethoddef(rout):
    return getmultilineblock(rout,'pymethoddef')

def getargs(rout):
    sortargs,args=[],[]
    if rout.has_key('args'):
        args=rout['args']
        if rout.has_key('sortvars'):
            for a in rout['sortvars']:
                if a in args: sortargs.append(a)
            for a in args:
                if a not in sortargs:
                    sortargs.append(a)
        else: sortargs=rout['args']
    return args,sortargs

def getargs2(rout):
    sortargs,args=[],rout.get('args',[])
    auxvars = [a for a in rout['vars'].keys() if isintent_aux(rout['vars'][a])\
               and a not in args]
    args = auxvars + args
    if rout.has_key('sortvars'):
        for a in rout['sortvars']:
            if a in args: sortargs.append(a)
        for a in args:
            if a not in sortargs:
                sortargs.append(a)
    else: sortargs=auxvars + rout['args']
    return args,sortargs

def getrestdoc(rout):
    if not rout.has_key('f2pymultilines'):
        return None
    k = None
    if rout['block']=='python module':
        k = rout['block'],rout['name']
    return rout['f2pymultilines'].get(k,None)

def gentitle(name):
    l=(80-len(name)-6)/2
    return '/*%s %s %s*/'%(l*'*',name,l*'*')
def flatlist(l):
    if type(l)==types.ListType:
        return reduce(lambda x,y,f=flatlist:x+f(y),l,[])
    return [l]
def stripcomma(s):
    if s and s[-1]==',': return s[:-1]
    return s
def replace(str,dict,defaultsep=''):
    if type(dict)==types.ListType:
        return map(lambda d,f=replace,sep=defaultsep,s=str:f(s,d,sep),dict)
    if type(str)==types.ListType:
        return map(lambda s,f=replace,sep=defaultsep,d=dict:f(s,d,sep),str)
    for k in 2*dict.keys():
        if k=='separatorsfor': continue
        if dict.has_key('separatorsfor') and dict['separatorsfor'].has_key(k):
            sep=dict['separatorsfor'][k]
        else:
            sep=defaultsep
        if type(dict[k])==types.ListType:
            str=string.replace(str,'#%s#'%(k),string.join(flatlist(dict[k]),sep))
        else:
            str=string.replace(str,'#%s#'%(k),dict[k])
    return str

def dictappend(rd,ar):
    if type(ar)==types.ListType:
        for a in ar: rd=dictappend(rd,a)
        return rd
    for k in ar.keys():
        if k[0]=='_': continue
        if rd.has_key(k):
            if type(rd[k])==types.StringType: rd[k]=[rd[k]]
            if type(rd[k])==types.ListType:
                if type(ar[k])==types.ListType: rd[k]=rd[k]+ar[k]
                else: rd[k].append(ar[k])
            elif type(rd[k])==types.DictType:
                if type(ar[k])==types.DictType:
                    if k=='separatorsfor':
                        for k1 in ar[k].keys():
                            if not rd[k].has_key(k1): rd[k][k1]=ar[k][k1]
                    else: rd[k]=dictappend(rd[k],ar[k])
        else: rd[k]=ar[k]
    return rd

def applyrules(rules,dict,var={}):
    ret={}
    if type(rules)==types.ListType:
        for r in rules:
            rr=applyrules(r,dict,var)
            ret=dictappend(ret,rr)
            if rr.has_key('_break'): break
        return ret
    if rules.has_key('_check') and (not rules['_check'](var)): return ret
    if rules.has_key('need'):
        res = applyrules({'needs':rules['need']},dict,var)
        if res.has_key('needs'):
            cfuncs.append_needs(res['needs'])

    for k in rules.keys():
        if k=='separatorsfor': ret[k]=rules[k]; continue
        if type(rules[k])==types.StringType:
            ret[k]=replace(rules[k],dict)
        elif type(rules[k])==types.ListType:
            ret[k]=[]
            for i in rules[k]:
                ar=applyrules({k:i},dict,var)
                if ar.has_key(k): ret[k].append(ar[k])
        elif k[0]=='_':
            continue
        elif type(rules[k])==types.DictType:
            ret[k]=[]
            for k1 in rules[k].keys():
                if type(k1)==types.FunctionType and k1(var):
                    if type(rules[k][k1])==types.ListType:
                        for i in rules[k][k1]:
                            if type(i)==types.DictType:
                                res=applyrules({'supertext':i},dict,var)
                                if res.has_key('supertext'): i=res['supertext']
                                else: i=''
                            ret[k].append(replace(i,dict))
                    else:
                        i=rules[k][k1]
                        if type(i)==types.DictType:
                            res=applyrules({'supertext':i},dict)
                            if res.has_key('supertext'): i=res['supertext']
                            else: i=''
                        ret[k].append(replace(i,dict))
        else:
            errmess('applyrules: ignoring rule %s.\n'%`rules[k]`)
        if type(ret[k])==types.ListType:
            if len(ret[k])==1: ret[k]=ret[k][0]
            if ret[k]==[]: del ret[k]
    return ret
