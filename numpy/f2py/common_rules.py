"""
Build common block mechanism for f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
from . import __version__

f2py_version = __version__.version

from . import capi_maps, func2subr
from .auxfuncs import getuseblocks, hasbody, hascommon, hasnote, isintent_hide, outmess
from .crackfortran import rmbadname


def findcommonblocks(block, top=1):
    ret = []
    if hascommon(block):
        for key, value in block['common'].items():
            vars_ = {v: block['vars'][v] for v in value}
            ret.append((key, value, vars_))
    elif hasbody(block):
        for b in block['body']:
            ret = ret + findcommonblocks(b, 0)
    if top:
        tret = []
        names = []
        for t in ret:
            if t[0] not in names:
                names.append(t[0])
                tret.append(t)
        return tret
    return ret


def buildhooks(m):
    ret = {'commonhooks': [], 'initcommonhooks': [],
           'docs': ['"COMMON blocks:\\n"']}
    fwrap = ['']

    def fadd(line, s=fwrap):
        s[0] = f'{s[0]}\n      {line}'
    chooks = ['']

    def cadd(line, s=chooks):
        s[0] = f'{s[0]}\n{line}'
    ihooks = ['']

    def iadd(line, s=ihooks):
        s[0] = f'{s[0]}\n{line}'
    doc = ['']

    def dadd(line, s=doc):
        s[0] = f'{s[0]}\n{line}'
    for (name, vnames, vars) in findcommonblocks(m):
        lower_name = name.lower()
        hnames, inames = [], []
        for n in vnames:
            if isintent_hide(vars[n]):
                hnames.append(n)
            else:
                inames.append(n)
        if hnames:
            outmess(f'\t\tConstructing COMMON block support for "{name}"...\n\t\t  '
                    f'{",".join(inames)}\n\t\t  Hidden: {",".join(hnames)}\n')
        else:
            outmess(f'\t\tConstructing COMMON block support for "{name}"...\n\t\t  '
                    f'{",".join(inames)}\n')
        fadd(f'subroutine f2pyinit{name}(setupfunc)')
        for usename in getuseblocks(m):
            fadd(f'use {usename}')
        fadd('external setupfunc')
        for n in vnames:
            fadd(func2subr.var2fixfortran(vars, n))
        if name == '_BLNK_':
            fadd(f"common {','.join(vnames)}")
        else:
            fadd(f"common /{name}/ {','.join(vnames)}")
        fadd(f"call setupfunc({','.join(inames)})")
        fadd('end\n')
        cadd(f'static FortranDataDef f2py_{name}_def[] = {{')
        idims = []
        for n in inames:
            ct = capi_maps.getctype(vars[n])
            elsize = capi_maps.get_elsize(vars[n])
            at = capi_maps.c2capi_map[ct]
            dm = capi_maps.getarrdims(n, vars[n])
            if dm['dims']:
                idims.append(f"({dm['dims']})")
            else:
                idims.append('')
            dms = dm['dims'].strip()
            if not dms:
                dms = '-1'
            cadd(f'\t{{\"{n}\",{dm["rank"]},{{{{{dms}}}}},{at}, {elsize}}},')
        cadd('\t{NULL}\n};')
        inames1 = rmbadname(inames)
        inames1_tps = ','.join(['char *' + s for s in inames1])
        cadd(f'static void f2py_setup_{name}({inames1_tps}) {{')
        cadd('\tint i_f2py=0;')
        for n in inames1:
            cadd(f'\tf2py_{name}_def[i_f2py++].data = {n};')
        cadd('}')
        if '_' in lower_name:
            F_FUNC = 'F_FUNC_US'
        else:
            F_FUNC = 'F_FUNC'
        cadd(f"extern void {F_FUNC}(f2pyinit{lower_name},F2PYINIT{name.upper()})"
             f"(void(*)({','.join(['char*'] * len(inames1))}));")
        cadd(f'static void f2py_init_{name}(void) {{')
        cadd(f'\t{F_FUNC}(f2pyinit{lower_name},F2PYINIT{name.upper()})'
             f'(f2py_setup_{name});')
        cadd('}\n')
        iadd(f'\ttmp = PyFortranObject_New(f2py_{name}_def,f2py_init_{name});')
        iadd('\tif (tmp == NULL) return NULL;')
        iadd(f'\tif (F2PyDict_SetItemString(d, "{name}", tmp) == -1) return NULL;')
        iadd('\tPy_DECREF(tmp);')
        tname = name.replace('_', '\\_')
        dadd(f'\\subsection{{Common block \\texttt{{{tname}}}}}\n')
        dadd('\\begin{description}')
        for n in inames:
            docsign = capi_maps.getarrdocsign(n, vars[n])
            dadd(f'\\item[]{{{{}}\\verb@{docsign}@{{}}}}')
            if hasnote(vars[n]):
                note = vars[n]['note']
                if isinstance(note, list):
                    note = '\n'.join(note)
                dadd(f'--- {note}')
        dadd('\\end{description}')
        ret['docs'].append(
            f"\"\t/{name}/ {','.join(map(lambda v, d: v + d, inames, idims))}\\n\"")
    ret['commonhooks'] = chooks
    ret['initcommonhooks'] = ihooks
    ret['latexdoc'] = doc[0]
    if len(ret['docs']) <= 1:
        ret['docs'] = ''
    return ret, fwrap[0]
