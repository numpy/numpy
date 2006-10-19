#!/usr/bin/env python
"""

Copyright 1999,2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2005/05/06 10:57:33 $
Pearu Peterson
"""

__version__ = "$Revision: 1.60 $"[10:-1]

import __version__
f2py_version = __version__.version

import string,copy,re,os
from auxfuncs import *
from crackfortran import markoutercomma
import cb_rules

# Numarray and Numeric users should set this False
using_newcore = True

depargs=[]
lcb_map={}
lcb2_map={}
# forced casting: mainly caused by the fact that Python or Numeric
#                 C/APIs do not support the corresponding C types.
c2py_map={'double':'float',
          'float':'float',                          # forced casting
          'long_double':'float',                    # forced casting
          'char':'int',                             # forced casting
          'signed_char':'int',                      # forced casting
          'unsigned_char':'int',                    # forced casting
          'short':'int',                            # forced casting
          'unsigned_short':'int',                   # forced casting
          'int':'int',                              # (forced casting)
          'long':'int',
          'long_long':'long',
          'unsigned':'int',                         # forced casting
          'complex_float':'complex',                # forced casting
          'complex_double':'complex',
          'complex_long_double':'complex',          # forced casting
          'string':'string',
          }
c2capi_map={'double':'PyArray_DOUBLE',
            'float':'PyArray_FLOAT',
            'long_double':'PyArray_DOUBLE',           # forced casting
            'char':'PyArray_CHAR',
            'unsigned_char':'PyArray_UBYTE',
            'signed_char':'PyArray_SBYTE',
            'short':'PyArray_SHORT',
            'unsigned_short':'PyArray_USHORT',
            'int':'PyArray_INT',
            'unsigned':'PyArray_UINT',
            'long':'PyArray_LONG',
            'long_long':'PyArray_LONG',                # forced casting
            'complex_float':'PyArray_CFLOAT',
            'complex_double':'PyArray_CDOUBLE',
            'complex_long_double':'PyArray_CDOUBLE',   # forced casting
            'string':'PyArray_CHAR'}

#These new maps aren't used anyhere yet, but should be by default
#  unless building numeric or numarray extensions.
if using_newcore:
    c2capi_map={'double':'PyArray_DOUBLE',
            'float':'PyArray_FLOAT',
            'long_double':'PyArray_LONGDOUBLE',
            'char':'PyArray_BYTE',
            'unsigned_char':'PyArray_UBYTE',
            'signed_char':'PyArray_BYTE',
            'short':'PyArray_SHORT',
            'unsigned_short':'PyArray_USHORT',
            'int':'PyArray_INT',
            'unsigned':'PyArray_UINT',
            'long':'PyArray_LONG',
            'unsigned_long':'PyArray_ULONG',
            'long_long':'PyArray_LONGLONG',
            'unsigned_long_long':'Pyarray_ULONGLONG',
            'complex_float':'PyArray_CFLOAT',
            'complex_double':'PyArray_CDOUBLE',
            'complex_long_double':'PyArray_CDOUBLE',
            'string':'PyArray_CHAR', # f2py 2e is not ready for PyArray_STRING (must set itemisize etc)
            #'string':'PyArray_STRING'
                
                }
c2pycode_map={'double':'d',
              'float':'f',
              'long_double':'d',                       # forced casting
              'char':'1',
              'signed_char':'1',
              'unsigned_char':'b',
              'short':'s',
              'unsigned_short':'w',
              'int':'i',
              'unsigned':'u',
              'long':'l',
              'long_long':'L',
              'complex_float':'F',
              'complex_double':'D',
              'complex_long_double':'D',               # forced casting
              'string':'c'
              }
if using_newcore:
    c2pycode_map={'double':'d',
                 'float':'f',
                 'long_double':'g',
                 'char':'b',
                 'unsigned_char':'B',
                 'signed_char':'b',
                 'short':'h',
                 'unsigned_short':'H',
                 'int':'i',
                 'unsigned':'I',
                 'long':'l',
                 'unsigned_long':'L',
                 'long_long':'q',
                 'unsigned_long_long':'Q',
                 'complex_float':'F',
                 'complex_double':'D',
                 'complex_long_double':'G',
                 'string':'S'}
c2buildvalue_map={'double':'d',
                  'float':'f',
                  'char':'b',
                  'signed_char':'b',
                  'short':'h',
                  'int':'i',
                  'long':'l',
                  'long_long':'L',
                  'complex_float':'N',
                  'complex_double':'N',
                  'complex_long_double':'N',
                  'string':'z'}
if using_newcore:
    #c2buildvalue_map=???
    pass

f2cmap_all={'real':{'':'float','4':'float','8':'double','12':'long_double','16':'long_double'},
            'integer':{'':'int','1':'signed_char','2':'short','4':'int','8':'long_long',
                       '-1':'unsigned_char','-2':'unsigned_short','-4':'unsigned',
                       '-8':'unsigned_long_long'},
            'complex':{'':'complex_float','8':'complex_float',
                       '16':'complex_double','24':'complex_long_double',
                       '32':'complex_long_double'},
            'complexkind':{'':'complex_float','4':'complex_float',
                           '8':'complex_double','12':'complex_long_double',
                           '16':'complex_long_double'},
            'logical':{'':'int','1':'char','2':'short','4':'int','8':'long_long'},
            'double complex':{'':'complex_double'},
            'double precision':{'':'double'},
            'byte':{'':'char'},
            'character':{'':'string'}
            }

if os.path.isfile('.f2py_f2cmap'):
    # User defined additions to f2cmap_all.
    # .f2py_f2cmap must contain a dictionary of dictionaries, only.
    # For example, {'real':{'low':'float'}} means that Fortran 'real(low)' is
    # interpreted as C 'float'.
    # This feature is useful for F90/95 users if they use PARAMETERSs
    # in type specifications.
    try:
        outmess('Reading .f2py_f2cmap ...\n')
        f = open('.f2py_f2cmap','r')
        d = eval(f.read(),{},{})
        f.close()
        for k,d1 in d.items():
            for k1 in d1.keys():
                d1[string.lower(k1)] = d1[k1]
            d[string.lower(k)] = d[k]
        for k in d.keys():
            if not f2cmap_all.has_key(k): f2cmap_all[k]={}
            for k1 in d[k].keys():
                if c2py_map.has_key(d[k][k1]):
                    if f2cmap_all[k].has_key(k1):
                        outmess("\tWarning: redefinition of {'%s':{'%s':'%s'->'%s'}}\n"%(k,k1,f2cmap_all[k][k1],d[k][k1]))
                    f2cmap_all[k][k1] = d[k][k1]
                    outmess('\tMapping "%s(kind=%s)" to "%s"\n' % (k,k1,d[k][k1]))
                else:
                    errmess("\tIgnoring map {'%s':{'%s':'%s'}}: '%s' must be in %s\n"%(k,k1,d[k][k1],d[k][k1],c2py_map.keys()))
        outmess('Succesfully applied user defined changes from .f2py_f2cmap\n')
    except:
        errmess('Failed to apply user defined changes from .f2py_f2cmap. Skipping.\n')
cformat_map={'double':'%g',
             'float':'%g',
             'long_double':'%Lg',
             'char':'%d',
             'signed_char':'%d',
             'unsigned_char':'%hhu',
             'short':'%hd',
             'unsigned_short':'%hu',
             'int':'%d',
             'unsigned':'%u',
             'long':'%ld',
             'unsigned_long':'%lu',
             'long_long':'%ld',
             'complex_float':'(%g,%g)',
             'complex_double':'(%g,%g)',
             'complex_long_double':'(%Lg,%Lg)',
             'string':'%s',
             }

############### Auxiliary functions
def getctype(var):
    """
    Determines C type
    """
    ctype='void'
    if isfunction(var):
        if var.has_key('result'): a=var['result']
        else: a=var['name']
        if var['vars'].has_key(a): return getctype(var['vars'][a])
        else: errmess('getctype: function %s has no return value?!\n'%a)
    elif issubroutine(var):
        return ctype
    elif var.has_key('typespec') and f2cmap_all.has_key(string.lower(var['typespec'])):
        typespec = string.lower(var['typespec'])
        f2cmap=f2cmap_all[typespec]
        ctype=f2cmap[''] # default type
        if var.has_key('kindselector'):
            if var['kindselector'].has_key('*'):
                try:
                    ctype=f2cmap[var['kindselector']['*']]
                except KeyError:
                    errmess('getctype: "%s %s %s" not supported.\n'%(var['typespec'],'*',var['kindselector']['*']))
            elif var['kindselector'].has_key('kind'):
                if f2cmap_all.has_key(typespec+'kind'):
                    f2cmap=f2cmap_all[typespec+'kind']
                try:
                    ctype=f2cmap[var['kindselector']['kind']]
                except KeyError:
                    if f2cmap_all.has_key(typespec):
                        f2cmap=f2cmap_all[typespec]
                    try:
                        ctype=f2cmap[str(var['kindselector']['kind'])]
                    except KeyError:
                        errmess('getctype: "%s(kind=%s)" not supported (use .f2py_f2cmap).\n'\
                                %(typespec,var['kindselector']['kind']))

    else:
        if not isexternal(var):
            errmess('getctype: No C-type found in "%s", assuming void.\n'%var)
    return ctype
def getstrlength(var):
    if isstringfunction(var):
        if var.has_key('result'): a=var['result']
        else: a=var['name']
        if var['vars'].has_key(a): return getstrlength(var['vars'][a])
        else: errmess('getstrlength: function %s has no return value?!\n'%a)
    if not isstring(var):
        errmess('getstrlength: expected a signature of a string but got: %s\n'%(`var`))
    len='1'
    if var.has_key('charselector'):
        a=var['charselector']
        if a.has_key('*'): len=a['*']
        elif a.has_key('len'): len=a['len']
    if re.match(r'\(\s*([*]|[:])\s*\)',len) or re.match(r'([*]|[:])',len):
    #if len in ['(*)','*','(:)',':']:
        if isintent_hide(var):
            errmess('getstrlength:intent(hide): expected a string with defined length but got: %s\n'%(`var`))
        len='-1'
    return len
def getarrdims(a,var,verbose=0):
    global depargs
    ret={}
    if isstring(var) and not isarray(var):
        ret['dims']=getstrlength(var)
        ret['size']=ret['dims']
        ret['rank']='1'
    elif isscalar(var):
        ret['size']='1'
        ret['rank']='0'
        ret['dims']=''
    elif isarray(var):
#         if not isintent_c(var):
#             var['dimension'].reverse()
        dim=copy.copy(var['dimension'])
        ret['size']=string.join(dim,'*')
        try: ret['size']=`eval(ret['size'])`
        except: pass
        ret['dims']=string.join(dim,',')
        ret['rank']=`len(dim)`
        ret['rank*[-1]']=`len(dim)*[-1]`[1:-1]
        for i in range(len(dim)): # solve dim for dependecies
            v=[]
            if dim[i] in depargs: v=[dim[i]]
            else:
                for va in depargs:
                    if re.match(r'.*?\b%s\b.*'%va,dim[i]):
                        v.append(va)
            for va in v:
                if depargs.index(va)>depargs.index(a):
                    dim[i]='*'
                    break
        ret['setdims'],i='',-1
        for d in dim:
            i=i+1
            if d not in ['*',':','(*)','(:)']:
                ret['setdims']='%s#varname#_Dims[%d]=%s,'%(ret['setdims'],i,d)
        if ret['setdims']: ret['setdims']=ret['setdims'][:-1]
        ret['cbsetdims'],i='',-1
        for d in var['dimension']:
            i=i+1
            if d not in ['*',':','(*)','(:)']:
                ret['cbsetdims']='%s#varname#_Dims[%d]=%s,'%(ret['cbsetdims'],i,d)
            elif verbose :
                errmess('getarrdims: If in call-back function: array argument %s must have bounded dimensions: got %s\n'%(`a`,`d`))
        if ret['cbsetdims']: ret['cbsetdims']=ret['cbsetdims'][:-1]
#         if not isintent_c(var):
#             var['dimension'].reverse()
    return ret
def getpydocsign(a,var):
    global lcb_map
    if isfunction(var):
        if var.has_key('result'): af=var['result']
        else: af=var['name']
        if var['vars'].has_key(af): return getpydocsign(af,var['vars'][af])
        else: errmess('getctype: function %s has no return value?!\n'%af)
        return '',''
    sig,sigout=a,a
    opt=''
    if isintent_in(var): opt='input'
    elif isintent_inout(var): opt='in/output'
    out_a = a
    if isintent_out(var):
        for k in var['intent']:
            if k[:4]=='out=':
                out_a = k[4:]
                break
    init=''
    ctype=getctype(var)

    if hasinitvalue(var):
        init,showinit=getinit(a,var)
        init='= %s'%(showinit)
    if isscalar(var):
        if isintent_inout(var):
            sig='%s :%s %s rank-0 array(%s,\'%s\')'%(a,init,opt,c2py_map[ctype],
                              c2pycode_map[ctype],)
        else:
            sig='%s :%s %s %s'%(a,init,opt,c2py_map[ctype])
        sigout='%s : %s'%(out_a,c2py_map[ctype])
    elif isstring(var):
        if isintent_inout(var):
            sig='%s :%s %s rank-0 array(string(len=%s),\'c\')'%(a,init,opt,getstrlength(var))
        else:
            sig='%s :%s %s string(len=%s)'%(a,init,opt,getstrlength(var))
        sigout='%s : string(len=%s)'%(out_a,getstrlength(var))
    elif isarray(var):
        dim=var['dimension']
        rank=`len(dim)`
        sig='%s :%s %s rank-%s array(\'%s\') with bounds (%s)'%(a,init,opt,rank,
                                             c2pycode_map[ctype],
                                             string.join(dim,','))
        if a==out_a:
            sigout='%s : rank-%s array(\'%s\') with bounds (%s)'\
                    %(a,rank,c2pycode_map[ctype],string.join(dim,','))
        else:
            sigout='%s : rank-%s array(\'%s\') with bounds (%s) and %s storage'\
                    %(out_a,rank,c2pycode_map[ctype],string.join(dim,','),a)
    elif isexternal(var):
        ua=''
        if lcb_map.has_key(a) and lcb2_map.has_key(lcb_map[a]) and lcb2_map[lcb_map[a]].has_key('argname'):
            ua=lcb2_map[lcb_map[a]]['argname']
            if not ua==a: ua=' => %s'%ua
            else: ua=''
        sig='%s : call-back function%s'%(a,ua)
        sigout=sig
    else:
        errmess('getpydocsign: Could not resolve docsignature for "%s".\\n'%a)
    return sig,sigout
def getarrdocsign(a,var):
    ctype=getctype(var)
    if isstring(var) and (not isarray(var)):
        sig='%s : rank-0 array(string(len=%s),\'c\')'%(a,getstrlength(var))
    elif isscalar(var):
        sig='%s : rank-0 array(%s,\'%s\')'%(a,c2py_map[ctype],
                                            c2pycode_map[ctype],)
    elif isarray(var):
        dim=var['dimension']
        rank=`len(dim)`
        sig='%s : rank-%s array(\'%s\') with bounds (%s)'%(a,rank,
                                                           c2pycode_map[ctype],
                                                           string.join(dim,','))
    return sig

def getinit(a,var):
    if isstring(var): init,showinit='""',"''"
    else: init,showinit='',''
    if hasinitvalue(var):
        init=var['=']
        showinit=init
        if iscomplex(var) or iscomplexarray(var):
            ret={}

            try:
                v = var["="]
                if ',' in v:
                    ret['init.r'],ret['init.i']=string.split(markoutercomma(v[1:-1]),'@,@')
                else:
                    v = eval(v,{},{})
                    ret['init.r'],ret['init.i']=str(v.real),str(v.imag)
            except: raise 'sign2map: expected complex number `(r,i)\' but got `%s\' as initial value of %s.'%(init,`a`)
            if isarray(var):
                init='(capi_c.r=%s,capi_c.i=%s,capi_c)'%(ret['init.r'],ret['init.i'])
        elif isstring(var):
            if not init: init,showinit='""',"''"
            if init[0]=="'":
                init='"%s"'%(string.replace(init[1:-1],'"','\\"'))
            if init[0]=='"': showinit="'%s'"%(init[1:-1])
    return init,showinit

def sign2map(a,var):
    """
    varname,ctype,atype
    init,init.r,init.i,pytype
    vardebuginfo,vardebugshowvalue,varshowvalue
    varrfromat
    intent
    """
    global lcb_map,cb_map
    out_a = a
    if isintent_out(var):
        for k in var['intent']:
            if k[:4]=='out=':
                out_a = k[4:]
                break
    ret={'varname':a,'outvarname':out_a}
    ret['ctype']=getctype(var)
    intent_flags = []
    for f,s in isintent_dict.items():
        if f(var): intent_flags.append('F2PY_%s'%s)
    if intent_flags:
        #XXX: Evaluate intent_flags here.
        ret['intent'] = string.join(intent_flags,'|')
    else:
        ret['intent'] = 'F2PY_INTENT_IN'
    if isarray(var): ret['varrformat']='N'
    elif c2buildvalue_map.has_key(ret['ctype']):
        ret['varrformat']=c2buildvalue_map[ret['ctype']]
    else: ret['varrformat']='O'
    ret['init'],ret['showinit']=getinit(a,var)
    if hasinitvalue(var) and iscomplex(var) and not isarray(var):
        ret['init.r'],ret['init.i'] = string.split(markoutercomma(ret['init'][1:-1]),'@,@')
    if isexternal(var):
        ret['cbnamekey']=a
        if lcb_map.has_key(a):
            ret['cbname']=lcb_map[a]
            ret['maxnofargs']=lcb2_map[lcb_map[a]]['maxnofargs']
            ret['nofoptargs']=lcb2_map[lcb_map[a]]['nofoptargs']
            ret['cbdocstr']=lcb2_map[lcb_map[a]]['docstr']
            ret['cblatexdocstr']=lcb2_map[lcb_map[a]]['latexdocstr']
        else:
            ret['cbname']=a
            errmess('sign2map: Confused: external %s is not in lcb_map%s.\n'%(a,lcb_map.keys()))
    if isstring(var):
        ret['length']=getstrlength(var)
    if isarray(var):
        ret=dictappend(ret,getarrdims(a,var))
        dim=copy.copy(var['dimension'])
    if c2capi_map.has_key(ret['ctype']): ret['atype']=c2capi_map[ret['ctype']]
    # Debug info
    if debugcapi(var):
        il=[isintent_in,'input',isintent_out,'output',
            isintent_inout,'inoutput',isrequired,'required',
            isoptional,'optional',isintent_hide,'hidden',
            iscomplex,'complex scalar',
            l_and(isscalar,l_not(iscomplex)),'scalar',
            isstring,'string',isarray,'array',
            iscomplexarray,'complex array',isstringarray,'string array',
            iscomplexfunction,'complex function',
            l_and(isfunction,l_not(iscomplexfunction)),'function',
            isexternal,'callback',
            isintent_callback,'callback',
            isintent_aux,'auxiliary',
            #ismutable,'mutable',l_not(ismutable),'immutable',
            ]
        rl=[]
        for i in range(0,len(il),2):
            if il[i](var): rl.append(il[i+1])
        if isstring(var):
            rl.append('slen(%s)=%s'%(a,ret['length']))
        if isarray(var):
#             if not isintent_c(var):
#                 var['dimension'].reverse()
            ddim=string.join(map(lambda x,y:'%s|%s'%(x,y),var['dimension'],dim),',')
            rl.append('dims(%s)'%ddim)
#             if not isintent_c(var):
#                 var['dimension'].reverse()
        if isexternal(var):
            ret['vardebuginfo']='debug-capi:%s=>%s:%s'%(a,ret['cbname'],string.join(rl,','))
        else:
            ret['vardebuginfo']='debug-capi:%s %s=%s:%s'%(ret['ctype'],a,ret['showinit'],string.join(rl,','))
        if isscalar(var):
            if cformat_map.has_key(ret['ctype']):
                ret['vardebugshowvalue']='debug-capi:%s=%s'%(a,cformat_map[ret['ctype']])
        if isstring(var):
            ret['vardebugshowvalue']='debug-capi:slen(%s)=%%d %s=\\"%%s\\"'%(a,a)
        if isexternal(var):
            ret['vardebugshowvalue']='debug-capi:%s=%%p'%(a)
    if cformat_map.has_key(ret['ctype']):
        ret['varshowvalue']='#name#:%s=%s'%(a,cformat_map[ret['ctype']])
        ret['showvalueformat']='%s'%(cformat_map[ret['ctype']])
    if isstring(var):
        ret['varshowvalue']='#name#:slen(%s)=%%d %s=\\"%%s\\"'%(a,a)
    ret['pydocsign'],ret['pydocsignout']=getpydocsign(a,var)
    if hasnote(var):
        ret['note']=var['note']
    return ret

def routsign2map(rout):
    """
    name,NAME,begintitle,endtitle
    rname,ctype,rformat
    routdebugshowvalue
    """
    global lcb_map
    name = rout['name']
    fname = getfortranname(rout)
    ret={'name':name,
         'texname':string.replace(name,'_','\\_'),
         'name_lower':string.lower(name),
         'NAME':string.upper(name),
         'begintitle':gentitle(name),
         'endtitle':gentitle('end of %s'%name),
         'fortranname':fname,
         'FORTRANNAME':string.upper(fname),
         'callstatement':getcallstatement(rout) or '',
         'usercode':getusercode(rout) or '',
         'usercode1':getusercode1(rout) or '',
         }
    if '_' in fname:
        ret['F_FUNC'] = 'F_FUNC_US'
    else:
        ret['F_FUNC'] = 'F_FUNC'
    if '_' in name:
        ret['F_WRAPPEDFUNC'] = 'F_WRAPPEDFUNC_US'
    else:
        ret['F_WRAPPEDFUNC'] = 'F_WRAPPEDFUNC'
    lcb_map={}
    if rout.has_key('use'):
        for u in rout['use'].keys():
            if cb_rules.cb_map.has_key(u):
                for un in cb_rules.cb_map[u]:
                    ln=un[0]
                    if rout['use'][u].has_key('map'):
                        for k in rout['use'][u]['map'].keys():
                            if rout['use'][u]['map'][k]==un[0]: ln=k;break
                    lcb_map[ln]=un[1]
            #else:
            #    errmess('routsign2map: cb_map does not contain module "%s" used in "use" statement.\n'%(u))
    elif rout.has_key('externals') and rout['externals']:
        errmess('routsign2map: Confused: function %s has externals %s but no "use" statement.\n'%(ret['name'],`rout['externals']`))
    ret['callprotoargument'] = getcallprotoargument(rout,lcb_map) or ''
    if isfunction(rout):
        if rout.has_key('result'): a=rout['result']
        else: a=rout['name']
        ret['rname']=a
        ret['pydocsign'],ret['pydocsignout']=getpydocsign(a,rout)
        ret['ctype']=getctype(rout['vars'][a])
        if hasresultnote(rout):
            ret['resultnote']=rout['vars'][a]['note']
            rout['vars'][a]['note']=['See elsewhere.']
        if c2buildvalue_map.has_key(ret['ctype']):
            ret['rformat']=c2buildvalue_map[ret['ctype']]
        else:
            ret['rformat']='O'
            errmess('routsign2map: no c2buildvalue key for type %s\n'%(`ret['ctype']`))
        if debugcapi(rout):
            if cformat_map.has_key(ret['ctype']):
                ret['routdebugshowvalue']='debug-capi:%s=%s'%(a,cformat_map[ret['ctype']])
            if isstringfunction(rout):
                ret['routdebugshowvalue']='debug-capi:slen(%s)=%%d %s=\\"%%s\\"'%(a,a)
        if isstringfunction(rout):
            ret['rlength']=getstrlength(rout['vars'][a])
            if ret['rlength']=='-1':
                errmess('routsign2map: expected explicit specification of the length of the string returned by the fortran function %s; taking 10.\n'%(`rout['name']`))
                ret['rlength']='10'
    if hasnote(rout):
        ret['note']=rout['note']
        rout['note']=['See elsewhere.']
    return ret

def modsign2map(m):
    """
    modulename
    """
    if ismodule(m):
        ret={'f90modulename':m['name'],
             'F90MODULENAME':string.upper(m['name']),
             'texf90modulename':string.replace(m['name'],'_','\\_')}
    else:
        ret={'modulename':m['name'],
             'MODULENAME':string.upper(m['name']),
             'texmodulename':string.replace(m['name'],'_','\\_')}
    ret['restdoc'] = getrestdoc(m) or []
    if hasnote(m):
        ret['note']=m['note']
        #m['note']=['See elsewhere.']
    ret['usercode'] = getusercode(m) or ''
    ret['usercode1'] = getusercode1(m) or ''
    if m['body']:
        ret['interface_usercode'] = getusercode(m['body'][0]) or ''
    else:
        ret['interface_usercode'] = ''
    ret['pymethoddef'] = getpymethoddef(m) or ''
    return ret

def cb_sign2map(a,var):
    ret={'varname':a}
    ret['ctype']=getctype(var)
    if c2capi_map.has_key(ret['ctype']):
        ret['atype']=c2capi_map[ret['ctype']]
    if cformat_map.has_key(ret['ctype']):
        ret['showvalueformat']='%s'%(cformat_map[ret['ctype']])
    if isarray(var):
        ret=dictappend(ret,getarrdims(a,var))
    ret['pydocsign'],ret['pydocsignout']=getpydocsign(a,var)
    if hasnote(var):
        ret['note']=var['note']
        var['note']=['See elsewhere.']
    return ret

def cb_routsign2map(rout,um):
    """
    name,begintitle,endtitle,argname
    ctype,rctype,maxnofargs,nofoptargs,returncptr
    """
    ret={'name':'cb_%s_in_%s'%(rout['name'],um),
         'returncptr':''}
    if isintent_callback(rout):
        if '_' in rout['name']:
            F_FUNC='F_FUNC_US'
        else:
            F_FUNC='F_FUNC'
        ret['callbackname'] = '%s(%s,%s)' \
                              % (F_FUNC,
                                 rout['name'].lower(),
                                 rout['name'].upper(),
                                 )
        ret['static'] = 'extern'
    else:
        ret['callbackname'] = ret['name']
        ret['static'] = 'static'
    ret['argname']=rout['name']
    ret['begintitle']=gentitle(ret['name'])
    ret['endtitle']=gentitle('end of %s'%ret['name'])
    ret['ctype']=getctype(rout)
    ret['rctype']='void'
    if ret['ctype']=='string': ret['rctype']='void'
    else:
        ret['rctype']=ret['ctype']
    if ret['rctype']!='void':
        if iscomplexfunction(rout):
            ret['returncptr'] = """
#ifdef F2PY_CB_RETURNCOMPLEX
return_value=
#endif
"""
        else:
            ret['returncptr'] = 'return_value='
    if cformat_map.has_key(ret['ctype']):
        ret['showvalueformat']='%s'%(cformat_map[ret['ctype']])
    if isstringfunction(rout):
        ret['strlength']=getstrlength(rout)
    if isfunction(rout):
        if rout.has_key('result'): a=rout['result']
        else: a=rout['name']
        if hasnote(rout['vars'][a]):
            ret['note']=rout['vars'][a]['note']
            rout['vars'][a]['note']=['See elsewhere.']
        ret['rname']=a
        ret['pydocsign'],ret['pydocsignout']=getpydocsign(a,rout)
        if iscomplexfunction(rout):
            ret['rctype']="""
#ifdef F2PY_CB_RETURNCOMPLEX
#ctype#
#else
void
#endif
"""
    else:
        if hasnote(rout):
            ret['note']=rout['note']
            rout['note']=['See elsewhere.']
    nofargs=0
    nofoptargs=0
    if rout.has_key('args') and rout.has_key('vars'):
        for a in rout['args']:
            var=rout['vars'][a]
            if l_or(isintent_in,isintent_inout)(var):
                nofargs=nofargs+1
                if isoptional(var):
                    nofoptargs=nofoptargs+1
    ret['maxnofargs']=`nofargs`
    ret['nofoptargs']=`nofoptargs`
    if hasnote(rout) and isfunction(rout) and rout.has_key('result'):
        ret['routnote']=rout['note']
        rout['note']=['See elsewhere.']
    return ret

def common_sign2map(a,var): # obsolute
    ret={'varname':a}
    ret['ctype']=getctype(var)
    if isstringarray(var): ret['ctype']='char'
    if c2capi_map.has_key(ret['ctype']):
        ret['atype']=c2capi_map[ret['ctype']]
    if cformat_map.has_key(ret['ctype']):
        ret['showvalueformat']='%s'%(cformat_map[ret['ctype']])
    if isarray(var):
        ret=dictappend(ret,getarrdims(a,var))
    elif isstring(var):
        ret['size']=getstrlength(var)
        ret['rank']='1'
    ret['pydocsign'],ret['pydocsignout']=getpydocsign(a,var)
    if hasnote(var):
        ret['note']=var['note']
        var['note']=['See elsewhere.']
    ret['arrdocstr']=getarrdocsign(a,var) # for strings this returns 0-rank but actually is 1-rank
    return ret
