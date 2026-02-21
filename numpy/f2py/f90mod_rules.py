"""
Build F90 module support for f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
__version__ = "$Revision: 1.27 $"[10:-1]

f2py_version = 'See `f2py -v`'

import numpy as np

from . import capi_maps, func2subr

# The environment provided by auxfuncs.py is needed for some calls to eval.
# As the needed functions cannot be determined by static inspection of the
# code, it is safest to use import * pending a major refactoring of f2py.
from .auxfuncs import *
from .crackfortran import undo_rmbadname, undo_rmbadname1

options = {}


def findf90modules(m):
    if ismodule(m):
        return [m]
    if not hasbody(m):
        return []
    ret = []
    for b in m['body']:
        if ismodule(b):
            ret.append(b)
        else:
            ret = ret + findf90modules(b)
    return ret


fgetdims1 = f"""\
      external f2pysetdata
      logical ns
      integer r,i
      integer({np.intp().itemsize}) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then"""

fgetdims2 = """\
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))"""

fgetdims2_sa = """\
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
         !s(r) must be equal to len(d(1))
      end if
      flag = 2
      call f2pysetdata(d,allocated(d))"""


def buildhooks(pymod):
    from . import rules
    ret = {'f90modhooks': [], 'initf90modhooks': [], 'body': [],
           'need': ['F_FUNC', 'arrayobject.h'],
           'separatorsfor': {'includes0': '\n', 'includes': '\n'},
           'docs': ['"Fortran 90/95 modules:\\n"'],
           'latexdoc': []}
    fhooks = ['']

    def fadd(line, s=fhooks):
        s[0] = f'{s[0]}\n      {line}'
    doc = ['']

    def dadd(line, s=doc):
        s[0] = f'{s[0]}\n{line}'

    usenames = getuseblocks(pymod)
    for m in findf90modules(pymod):
        sargs, fargs, efargs, modobjs, notvars, onlyvars = [], [], [], [], [
            m['name']], []
        sargsp = []
        ifargs = []
        mfargs = []
        if hasbody(m):
            for b in m['body']:
                notvars.append(b['name'])
        for n in m['vars'].keys():
            var = m['vars'][n]

            if (n not in notvars and isvariable(var)) and (not l_or(isintent_hide, isprivate)(var)):
                onlyvars.append(n)
                mfargs.append(n)
        outmess(f"\t\tConstructing F90 module support for \"{m['name']}\"...\n")
        if len(onlyvars) == 0 and len(notvars) == 1 and m['name'] in notvars:
            outmess(f"\t\t\tSkipping {m['name']} since there are no public vars/func in this module...\n")
            continue

        # gh-25186
        if m['name'] in usenames and containscommon(m):
            outmess(f"\t\t\tSkipping {m['name']} since it is in 'use' and contains a common block...\n")
            continue
        # skip modules with derived types
        if m['name'] in usenames and containsderivedtypes(m):
            outmess(f"\t\t\tSkipping {m['name']} since it is in 'use' and contains a derived type...\n")
            continue
        if onlyvars:
            outmess(f"\t\t  Variables: {' '.join(onlyvars)}\n")
        chooks = ['']

        def cadd(line, s=chooks):
            s[0] = f'{s[0]}\n{line}'
        ihooks = ['']

        def iadd(line, s=ihooks):
            s[0] = f'{s[0]}\n{line}'

        vrd = capi_maps.modsign2map(m)
        cadd(f"static FortranDataDef f2py_{m['name']}_def[] = {{")
        dadd(f"\\subsection{{Fortran 90/95 module \\texttt{{{m['name']}}}}}\n")
        if hasnote(m):
            note = m['note']
            if isinstance(note, list):
                note = '\n'.join(note)
            dadd(note)
        if onlyvars:
            dadd('\\begin{description}')
        for n in onlyvars:
            var = m['vars'][n]
            modobjs.append(n)
            ct = capi_maps.getctype(var)
            at = capi_maps.c2capi_map[ct]
            dm = capi_maps.getarrdims(n, var)
            dms = dm['dims'].replace('*', '-1').strip()
            dms = dms.replace(':', '-1').strip()
            if not dms:
                dms = '-1'
            use_fgetdims2 = fgetdims2
            rank = dm['rank']
            elsize = capi_maps.get_elsize(var)
            cadd(f'\t{{"{undo_rmbadname1(n)}",{rank},{{{{{dms}}}}},{at}, '
                 f'{elsize}}},')
            docsign = capi_maps.getarrdocsign(n, var)
            dadd(f'\\item[]{{{{}}\\verb@{docsign}@{{}}}}')
            if hasnote(var):
                note = var['note']
                if isinstance(note, list):
                    note = '\n'.join(note)
                dadd(f'--- {note}')
            if isallocatable(var):
                fargs.append(f"f2py_{m['name']}_getdims_{n}")
                efargs.append(fargs[-1])
                sargs.append(
                    f'void (*{n})(int*,npy_intp*,void(*)(char*,npy_intp*),int*)')
                sargsp.append('void (*)(int*,npy_intp*,void(*)(char*,npy_intp*),int*)')
                iadd(f"\tf2py_{m['name']}_def[i_f2py++].func = {n};")
                fadd(f'subroutine {fargs[-1]}(r,s,f2pysetdata,flag)')
                fadd(f"use {m['name']}, only: d => {undo_rmbadname1(n)}\n")
                fadd('integer flag\n')
                fhooks[0] = fhooks[0] + fgetdims1
                dms = range(1, int(dm['rank']) + 1)
                alloc_args = ','.join(f's({i})' for i in dms)
                fadd(f' allocate(d({alloc_args}))\n')
                fhooks[0] = fhooks[0] + use_fgetdims2
                fadd(f'end subroutine {fargs[-1]}')
            else:
                fargs.append(n)
                sargs.append(f'char *{n}')
                sargsp.append('char*')
                iadd(f"\tf2py_{m['name']}_def[i_f2py++].data = {n};")
        if onlyvars:
            dadd('\\end{description}')
        if hasbody(m):
            m_name = m['name']
            for b in m['body']:
                b_name = b['name']
                if not isroutine(b):
                    outmess("f90mod_rules.buildhooks:"
                            f" skipping {b['block']} {b_name}\n")
                    continue
                modobjs.append(f"{b_name}()")
                b['modulename'] = m_name
                api, wrap = rules.buildapi(b)
                if isfunction(b):
                    fhooks[0] = fhooks[0] + wrap
                    fargs.append(f"f2pywrap_{m_name}_{b_name}")
                    ifargs.append(func2subr.createfuncwrapper(b, signature=1))
                elif wrap:
                    fhooks[0] = fhooks[0] + wrap
                    fargs.append(f"f2pywrap_{m_name}_{b_name}")
                    ifargs.append(
                        func2subr.createsubrwrapper(b, signature=1))
                else:
                    fargs.append(b_name)
                    mfargs.append(fargs[-1])
                api['externroutines'] = []
                ar = applyrules(api, vrd)
                ar['docs'] = []
                ar['docshort'] = []
                ret = dictappend(ret, ar)
                cadd(f'\t{{"{b_name}",-1,{{{{-1}}}},0,0,NULL,(void *)'
                      f'f2py_rout_#modulename#_{m_name}_{b_name},'
                      f'doc_f2py_rout_#modulename#_{m_name}_{b_name}}},')
                sargs.append(f"char *{b_name}")
                sargsp.append('char *')
                iadd(f"\tf2py_{m_name}_def[i_f2py++].data = {b_name};")
        cadd('\t{NULL}\n};\n')
        iadd('}')
        m_name = m['name']
        ihooks[0] = (f'static void f2py_setup_{m_name}({",".join(sargs)}) '
                     f'{{\n\tint i_f2py=0;{ihooks[0]}')
        if '_' in m_name:
            F_FUNC = 'F_FUNC_US'
        else:
            F_FUNC = 'F_FUNC'
        iadd(f'extern void {F_FUNC}(f2pyinit{m_name},'
             f'F2PYINIT{m_name.upper()})(void (*)({','.join(sargsp)}));')
        iadd(f'static void f2py_init_{m_name}(void) {{')
        iadd(f'\t{F_FUNC}(f2pyinit{m_name},'
             f'F2PYINIT{m_name.upper()})(f2py_setup_{m_name});')
        iadd('}\n')
        ret['f90modhooks'] = ret['f90modhooks'] + chooks + ihooks
        ret['initf90modhooks'] = [
            '\t{',
            ('\t\tPyObject *tmp = '
            f'PyFortranObject_New(f2py_{m_name}_def,f2py_init_{m_name});'),
            f'\t\tPyDict_SetItemString(d, "{m_name}", tmp);',
            '\t\tPy_XDECREF(tmp);',
            '\t}',
        ] + ret["initf90modhooks"]
        fadd('')
        fadd(f"subroutine f2pyinit{m_name}(f2pysetupfunc)")
        if mfargs:
            for a in undo_rmbadname(mfargs):
                fadd(f"use {m_name}, only : {a}")
        if ifargs:
            fadd(' '.join(['interface'] + ifargs))
            fadd('end interface')
        fadd('external f2pysetupfunc')
        if efargs:
            for a in undo_rmbadname(efargs):
                fadd(f'external {a}')
        fadd(f"call f2pysetupfunc({','.join(undo_rmbadname(fargs))})")
        fadd(f"end subroutine f2pyinit{m_name}\n")

        dadd('\n'.join(ret['latexdoc']).replace(
            r'\subsection{', r'\subsubsection{'))

        ret['latexdoc'] = []
        ret['docs'].append(f"\"\t{m_name} --- {','.join(undo_rmbadname(modobjs))}\"")

    ret['routine_defs'] = ''
    ret['doc'] = []
    ret['docshort'] = []
    ret['latexdoc'] = doc[0]
    if len(ret['docs']) <= 1:
        ret['docs'] = ''
    return ret, fhooks[0]
