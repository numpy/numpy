"""

Rules for building C/API module with f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
import copy
import re

from ._isocbind import isoc_kindmap
from .auxfuncs import (
    getfortranname,
    isexternal,
    isfunction,
    isfunction_wrap,
    isintent_in,
    isintent_out,
    islogicalfunction,
    ismoduleroutine,
    isscalar,
    issubroutine,
    issubroutine_wrap,
    outmess,
    show,
)


def var2fixfortran(vars, a, fa=None, f90mode=None):
    if fa is None:
        fa = a
    if a not in vars:
        show(vars)
        outmess(f'var2fixfortran: No definition for argument "{a}".\n')
        return ''
    if 'typespec' not in vars[a]:
        show(vars[a])
        outmess(f'var2fixfortran: No typespec for argument "{a}".\n')
        return ''
    vardef = vars[a]['typespec']
    if vardef == 'type' and 'typename' in vars[a]:
        vardef = f"{vardef}({vars[a]['typename']})"
    selector = {}
    lk = ''
    if 'kindselector' in vars[a]:
        selector = vars[a]['kindselector']
        lk = 'kind'
    elif 'charselector' in vars[a]:
        selector = vars[a]['charselector']
        lk = 'len'
    if '*' in selector:
        if f90mode:
            if selector['*'] in ['*', ':', '(*)']:
                vardef = f'{vardef}(len=*)'
            else:
                vardef = f"{vardef}({lk}={selector['*']})"
        elif selector['*'] in ['*', ':']:
            vardef = f"{vardef}*({selector['*']})"
        else:
            vardef = f"{vardef}*{selector['*']}"
    elif 'len' in selector:
        vardef = f"{vardef}(len={selector['len']}"
        if 'kind' in selector:
            vardef = f"{vardef},kind={selector['kind']})"
        else:
            vardef = f'{vardef})'
    elif 'kind' in selector:
        vardef = f"{vardef}(kind={selector['kind']})"

    vardef = f'{vardef} {fa}'
    if 'dimension' in vars[a]:
        vardef = f"{vardef}({','.join(vars[a]['dimension'])})"
    return vardef

def useiso_c_binding(rout):
    useisoc = False
    for value in rout['vars'].values():
        kind_value = value.get('kindselector', {}).get('kind')
        if kind_value in isoc_kindmap:
            return True
    return useisoc


def _extract_cb_interface(saved_interface, cbname):
    """Extract the interface block for callback ``cbname`` from *saved_interface*.

    Returns the interface block as a string (including the surrounding
    ``interface`` / ``end interface`` lines), or ``None`` when no such block
    is found.
    """
    lines = saved_interface.split('\n')
    collecting = False
    depth = 0
    collected = []
    # Match nested "interface" block that contains "function <cbname>"
    # or "subroutine <cbname>" immediately after "interface".
    for line in lines:
        stripped = line.strip().lower()
        if not collecting:
            if stripped == 'interface':
                # Start of a candidate interface block -- peek ahead is
                # hard, so just start collecting and verify afterwards.
                collecting = True
                depth = 1
                collected = [line]
                continue
        else:
            collected.append(line)
            if re.match(r'\s*interface\b', stripped):
                depth += 1
            elif re.match(r'\s*end\s+interface\b', stripped):
                depth -= 1
                if depth == 0:
                    # Check whether this block defines cbname
                    block_text = '\n'.join(collected)
                    pattern = (
                        r'(?:function|subroutine)\s+' + re.escape(cbname)
                        + r'\b'
                    )
                    if re.search(pattern, block_text, re.IGNORECASE):
                        return block_text
                    collecting = False
                    collected = []
    return None


def _clean_saved_interface(saved_interface, cb_names):
    """Remove ``external`` declarations from *saved_interface* for names that
    have an interface block in the same scope.

    When f2py expands ``use __user__routines`` into inline interface blocks,
    the original ``external`` declaration becomes redundant and violates the
    Fortran standard (diagnosed by Intel compilers).
    """
    if not cb_names:
        return saved_interface
    lines = saved_interface.split('\n')
    result = []
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith('external'):
            # Parse the names after 'external'
            decl_part = line.split('external', 1)[1].strip()
            # Could be "external f" or "external :: f" or "external f, g"
            decl_part = decl_part.lstrip(':').strip()
            names = [n.strip().lower() for n in decl_part.split(',')]
            # Remove names that have interface blocks
            remaining = [n for n in names if n not in cb_names]
            if remaining:
                result.append(
                    line.replace(decl_part, ', '.join(remaining))
                )
            # else: skip the entire external line
        else:
            result.append(line)
    return '\n'.join(result)


def createfuncwrapper(rout, signature=0):
    assert isfunction(rout)

    extra_args = []
    vars = rout['vars']
    for a in rout['args']:
        v = rout['vars'][a]
        for i, d in enumerate(v.get('dimension', [])):
            if d == ':':
                dn = f'f2py_{a}_d{i}'
                dv = {'typespec': 'integer', 'intent': ['hide']}
                dv['='] = f'shape({a}, {i})'
                extra_args.append(dn)
                vars[dn] = dv
                v['dimension'][i] = dn
    rout['args'].extend(extra_args)
    need_interface = bool(extra_args)

    ret = ['']

    def add(line, ret=ret):
        ret[0] = f'{ret[0]}\n      {line}'
    name = rout['name']
    fortranname = getfortranname(rout)
    f90mode = ismoduleroutine(rout)
    newname = f'{name}f2pywrap'

    if newname not in vars:
        vars[newname] = vars[name]
        args = [newname] + rout['args'][1:]
    else:
        args = [newname] + rout['args']

    l_tmpl = var2fixfortran(vars, name, '@@@NAME@@@', f90mode)
    if l_tmpl[:13] == 'character*(*)':
        if f90mode:
            l_tmpl = 'character(len=10)' + l_tmpl[13:]
        else:
            l_tmpl = 'character*10' + l_tmpl[13:]
        charselect = vars[name]['charselector']
        if charselect.get('*', '') == '(*)':
            charselect['*'] = '10'

    l1 = l_tmpl.replace('@@@NAME@@@', newname)
    rl = None

    useisoc = useiso_c_binding(rout)
    sargs = ', '.join(args)
    if f90mode:
        # gh-23598 fix warning
        # Essentially, this gets called again with modules where the name of the
        # function is added to the arguments, which is not required, and removed
        sargs = sargs.replace(f"{name}, ", '')
        args = [arg for arg in args if arg != name]
        rout['args'] = args
        add(f"subroutine f2pywrap_{rout['modulename']}_{name} ({sargs})")
        if not signature:
            add(f"use {rout['modulename']}, only : {fortranname}")
        if useisoc:
            add('use iso_c_binding')
    else:
        add(f'subroutine f2pywrap{name} ({sargs})')
        if useisoc:
            add('use iso_c_binding')
        if not need_interface:
            add(f'external {fortranname}')
            rl = l_tmpl.replace('@@@NAME@@@', '') + ' ' + fortranname

    if need_interface:
        for line in rout['saved_interface'].split('\n'):
            if line.lstrip().startswith('use ') and '__user__' not in line:
                add(line)

    args = args[1:]
    dumped_args = []

    # Collect names of external (callback) arguments that have a proper
    # interface block inside saved_interface.  For those we emit the
    # interface block instead of a bare ``external`` declaration to
    # satisfy strict Fortran compilers (gh-20157).
    cb_names_with_iface = set()
    if need_interface and 'saved_interface' in rout:
        for a in args:
            if isexternal(vars[a]):
                if _extract_cb_interface(rout['saved_interface'], a):
                    cb_names_with_iface.add(a)

    for a in args:
        if isexternal(vars[a]):
            if a in cb_names_with_iface:
                # Emit the full interface block for this callback at the
                # outer wrapper scope so the compiler knows its signature.
                # The extracted text already includes the enclosing
                # ``interface`` / ``end interface`` lines.
                cb_iface = _extract_cb_interface(
                    rout['saved_interface'], a)
                for iline in cb_iface.split('\n'):
                    add(iline.rstrip())
            else:
                add(f'external {a}')
            dumped_args.append(a)
    for a in args:
        if a in dumped_args:
            continue
        if isscalar(vars[a]):
            add(var2fixfortran(vars, a, f90mode=f90mode))
            dumped_args.append(a)
    for a in args:
        if a in dumped_args:
            continue
        if isintent_in(vars[a]):
            add(var2fixfortran(vars, a, f90mode=f90mode))
            dumped_args.append(a)
    for a in args:
        if a in dumped_args:
            continue
        add(var2fixfortran(vars, a, f90mode=f90mode))

    add(l1)
    if rl is not None:
        add(rl)

    if need_interface:
        if f90mode:
            # f90 module already defines needed interface
            pass
        else:
            # Clean redundant ``external`` declarations from the saved
            # interface when inline interface blocks already exist for
            # those names (gh-20157).
            cleaned = _clean_saved_interface(
                rout['saved_interface'], cb_names_with_iface)
            add('interface')
            add(cleaned.lstrip())
            add('end interface')

    sargs = ', '.join([a for a in args if a not in extra_args])

    if not signature:
        if islogicalfunction(rout):
            add(f'{newname} = .not.(.not.{fortranname}({sargs}))')
        else:
            add(f'{newname} = {fortranname}({sargs})')
    if f90mode:
        add(f"end subroutine f2pywrap_{rout['modulename']}_{name}")
    else:
        add('end')
    return ret[0]


def createsubrwrapper(rout, signature=0):
    assert issubroutine(rout)

    extra_args = []
    vars = rout['vars']
    for a in rout['args']:
        v = rout['vars'][a]
        for i, d in enumerate(v.get('dimension', [])):
            if d == ':':
                dn = f'f2py_{a}_d{i}'
                dv = {'typespec': 'integer', 'intent': ['hide']}
                dv['='] = f'shape({a}, {i})'
                extra_args.append(dn)
                vars[dn] = dv
                v['dimension'][i] = dn
    rout['args'].extend(extra_args)
    need_interface = bool(extra_args)

    ret = ['']

    def add(line, ret=ret):
        ret[0] = f'{ret[0]}\n      {line}'
    name = rout['name']
    fortranname = getfortranname(rout)
    f90mode = ismoduleroutine(rout)

    args = rout['args']

    useisoc = useiso_c_binding(rout)
    sargs = ', '.join(args)
    if f90mode:
        add(f"subroutine f2pywrap_{rout['modulename']}_{name} ({sargs})")
        if useisoc:
            add('use iso_c_binding')
        if not signature:
            add(f"use {rout['modulename']}, only : {fortranname}")
    else:
        add(f'subroutine f2pywrap{name} ({sargs})')
        if useisoc:
            add('use iso_c_binding')
        if not need_interface:
            add(f'external {fortranname}')

    if need_interface:
        for line in rout['saved_interface'].split('\n'):
            if line.lstrip().startswith('use ') and '__user__' not in line:
                add(line)

    dumped_args = []

    # Collect names of external (callback) arguments that have a proper
    # interface block inside saved_interface (gh-20157).
    cb_names_with_iface = set()
    if need_interface and 'saved_interface' in rout:
        for a in args:
            if isexternal(vars[a]):
                if _extract_cb_interface(rout['saved_interface'], a):
                    cb_names_with_iface.add(a)

    for a in args:
        if isexternal(vars[a]):
            if a in cb_names_with_iface:
                cb_iface = _extract_cb_interface(
                    rout['saved_interface'], a)
                for iline in cb_iface.split('\n'):
                    add(iline.rstrip())
            else:
                add(f'external {a}')
            dumped_args.append(a)
    for a in args:
        if a in dumped_args:
            continue
        if isscalar(vars[a]):
            add(var2fixfortran(vars, a, f90mode=f90mode))
            dumped_args.append(a)
    for a in args:
        if a in dumped_args:
            continue
        add(var2fixfortran(vars, a, f90mode=f90mode))

    if need_interface:
        if f90mode:
            # f90 module already defines needed interface
            pass
        else:
            # Clean redundant ``external`` declarations from the saved
            # interface when inline interface blocks already exist for
            # those names (gh-20157).
            cleaned = _clean_saved_interface(
                rout['saved_interface'], cb_names_with_iface)
            add('interface')
            for line in cleaned.split('\n'):
                if line.lstrip().startswith('use ') and '__user__' in line:
                    continue
                add(line)
            add('end interface')

    sargs = ', '.join([a for a in args if a not in extra_args])

    if not signature:
        add(f'call {fortranname}({sargs})')
    if f90mode:
        add(f"end subroutine f2pywrap_{rout['modulename']}_{name}")
    else:
        add('end')
    return ret[0]


def assubr(rout):
    if isfunction_wrap(rout):
        fortranname = getfortranname(rout)
        name = rout['name']
        outmess('\t\tCreating wrapper for Fortran function '
                f'"{name}"("{fortranname}")...\n')
        rout = copy.copy(rout)
        fname = name
        rname = fname
        if 'result' in rout:
            rname = rout['result']
            rout['vars'][fname] = rout['vars'][rname]
        fvar = rout['vars'][fname]
        if not isintent_out(fvar):
            if 'intent' not in fvar:
                fvar['intent'] = []
            fvar['intent'].append('out')
            flag = 1
            for i in fvar['intent']:
                if i.startswith('out='):
                    flag = 0
                    break
            if flag:
                fvar['intent'].append(f'out={rname}')
        rout['args'][:] = [fname] + rout['args']
        return rout, createfuncwrapper(rout)
    if issubroutine_wrap(rout):
        fortranname = getfortranname(rout)
        name = rout['name']
        outmess('\t\tCreating wrapper for Fortran subroutine '
                f'"{name}"("{fortranname}")...\n')
        rout = copy.copy(rout)
        return rout, createsubrwrapper(rout)
    return rout, ''
