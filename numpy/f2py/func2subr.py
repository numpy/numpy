"""

Rules for building C/API module with f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
import copy

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


# User-module blocks available while building F90 wrappers for one extension
# (set by rules.buildmodule from the um list; complements crackfortran.usermodules).
_active_user_modules = []


def set_active_user_modules(um):
    """Register python-module ``__user__`` blocks for the current buildmodule call."""
    global _active_user_modules
    _active_user_modules = list(um or [])


def _iter_routine_blocks(body):
    """Yield function/subroutine blocks nested under *body* (incl. interfaces)."""
    for b in body or []:
        btype = b.get('block')
        if btype in ('function', 'subroutine') and b.get('name'):
            yield b
        elif btype in ('interface', 'abstract interface'):
            yield from _iter_routine_blocks(b.get('body'))


def _user_module_catalog():
    """All known ``__user__`` modules for this build (globals + active um list)."""
    from . import crackfortran
    seen = set()
    out = []
    for um in list(crackfortran.usermodules) + list(_active_user_modules):
        name = um.get('name')
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(um)
    return out


def _remote_names_for_local(use_dict, local_name):
    """Names under which *local_name* may appear in a used user-module.

    Handles ``use m, only: f => fun`` style maps stored as
    ``use[m]['map'][local] = remote``.
    """
    names = {local_name.lower()}
    for _mname, spec in (use_dict or {}).items():
        mapping = (spec or {}).get('map') or {}
        for local, remote in mapping.items():
            if (local or '').lower() == local_name.lower():
                names.add((remote or local).lower())
            if (remote or '').lower() == local_name.lower():
                names.add((local or remote).lower())
    return names


def _callback_routine_blocks(rout):
    """Map lowercased *local* callback dummy name -> cracked routine block.

    Prefer definitions still on ``rout['body']``. After postcrack moves
    callback signatures into ``__user__`` python modules, resolve them via
    ``rout['use']`` against both ``crackfortran.usermodules`` and the
    ``um`` list active for the current ``buildmodule`` call (needed for the
    ``-h``/``.pyf`` workflow where usermodules may not be populated).
    """
    found = {}
    for b in _iter_routine_blocks(rout.get('body')):
        found[b['name'].lower()] = b

    use = rout.get('use') or {}
    if not use:
        return found

    # Index all routines in user modules by lowercased name
    by_name = {}
    um_by_modname = {}
    for um in _user_module_catalog():
        um_by_modname[um.get('name')] = um
        for b in _iter_routine_blocks(um.get('body')):
            by_name.setdefault(b['name'].lower(), b)

    for mname in use:
        um = um_by_modname.get(mname)
        if um is None:
            continue
        for b in _iter_routine_blocks(um.get('body')):
            by_name.setdefault(b['name'].lower(), b)

    # Externals may be listed on args and/or rout['externals']
    vars_ = rout.get('vars') or {}
    candidates = list(rout.get('args') or [])
    for e in rout.get('externals') or []:
        if e not in candidates:
            candidates.append(e)
    for a in candidates:
        if a.lower() in found:
            continue
        if a in vars_ and not isexternal(vars_[a]):
            continue
        # If vars is incomplete (unit tests), still try externals by name
        if a in vars_ or a in (rout.get('externals') or []):
            for remote in _remote_names_for_local(use, a):
                if remote in by_name:
                    found[a.lower()] = by_name[remote]
                    break
    return found


def _safe_ident(raw, prefix, taken, max_len=63):
    """Build a Fortran-safe identifier that does not collide with *taken*."""
    base = ''.join(c if (c.isalnum() or c == '_') else '_' for c in raw)
    if not base or base[0].isdigit():
        base = 'r_' + base
    name = f'{prefix}{base}'
    if len(name) > max_len:
        name = name[:max_len]
    if name.lower() not in taken:
        taken.add(name.lower())
        return name
    n = 2
    while True:
        suffix = f'_{n}'
        cand = (name[: max_len - len(suffix)] + suffix)
        if cand.lower() not in taken:
            taken.add(cand.lower())
            return cand
        n += 1


def _cb_iface_module_name(rout, taken=None):
    """Fortran module name holding abstract interfaces for this routine's callbacks."""
    # Per-routine modules avoid colliding two different signatures for the
    # same dummy name across wrappers in one extension (gh-20157 discussion).
    if taken is None:
        taken = set()
    raw = getfortranname(rout)
    return _safe_ident(raw, 'f2py_cb_ifaces_', taken)


def _abstract_iface_name(cbname_lower, taken=None):
    """Name of the abstract interface body for callback *cbname_lower*.

    Must differ from the dummy argument name so ``use`` of the module does
    not make the dummy name ambiguous (gfortran: "ambiguous reference").
    """
    if taken is None:
        taken = set()
    return _safe_ident(cbname_lower, 'f2py_ai_', taken)


def _collect_use_lines_for_module(cb_blocks, host_rout=None):
    """USE lines needed so abstract-interface kinds/types resolve (gh-20157)."""
    from .crackfortran import use2fortran
    seen = set()
    lines = []

    def add_from(block):
        use = (block or {}).get('use') or {}
        for mname, spec in use.items():
            if '__user__' in mname:
                continue
            key = (mname, repr(spec))
            if key in seen:
                continue
            seen.add(key)
            # use2fortran expects the full use dict; emit one module at a time
            chunk = use2fortran({mname: spec}, tab='')
            for raw in chunk.split('\n'):
                s = raw.strip()
                if s:
                    lines.append(s)

    if host_rout is not None:
        add_from(host_rout)
    for block in cb_blocks.values():
        add_from(block)
    return lines


def _rename_callback_block_for_abstract(block, abs_name):
    """Deep-copy *block* as abstract-interface body named *abs_name*.

    When the result is the function name (no ``result(...)``), retarget the
    typed result variable so crack2fortrangen does not drop the type.
    """
    b = copy.deepcopy(block)
    old = b.get('name')
    b['name'] = abs_name
    if not old:
        return b
    if b.get('result') == old:
        b['result'] = abs_name
    vars_ = b.setdefault('vars', {})
    if old in vars_ and abs_name not in vars_:
        vars_[abs_name] = vars_.pop(old)
    elif 'result' not in b and old in vars_:
        # Implicit result is the function name.
        b['result'] = abs_name
        vars_[abs_name] = vars_.pop(old)
    return b


def _build_callback_iface_module(mod_name, cb_blocks, host_rout=None,
                                 abs_map=None):
    """Emit a Fortran module of abstract interfaces for *cb_blocks*.

    *abs_map* maps local callback lower-name -> abstract interface body name.
    Callers ``use`` the module and declare
    ``procedure(<abs>) :: <local>`` for each callback dummy.
    """
    from .crackfortran import crack2fortrangen
    # Interface bodies are separate scoping units: kind/type names from a
    # host USE are not visible inside them unless the body itself has USE
    # or IMPORT.  Emit USE inside each abstract-interface body.
    use_lines = _collect_use_lines_for_module(cb_blocks, host_rout)
    lines = [
        f'module {mod_name}',
        '  implicit none',
        '  abstract interface',
    ]
    abs_map = abs_map or {}
    for key, block in sorted(cb_blocks.items(), key=lambda kv: kv[0]):
        abs_name = abs_map.get(key) or _abstract_iface_name(key)
        b = _rename_callback_block_for_abstract(block, abs_name)
        text = crack2fortrangen(b, tab='\n  ', as_interface=True)
        body_lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
        if not body_lines:
            continue
        # procedure header, then USE, then remaining specification statements
        lines.append(f'    {body_lines[0]}')
        for use_line in use_lines:
            lines.append(f'      {use_line}')
        for ln in body_lines[1:]:
            lines.append(f'    {ln}')
    # Abstract interfaces close with END INTERFACE (not END ABSTRACT INTERFACE).
    lines.append('  end interface')
    lines.append(f'end module {mod_name}')
    return '\n'.join(lines)


def _rewrite_saved_interface_use_module(saved_interface, cb_orig_names,
                                        mod_name, abs_map):
    """Adapt *saved_interface* for module-based callback interfaces.

    Drop bare EXTERNAL and nested interface blocks for the named callbacks.
    Emit all ``use`` statements (callback module first, then host USEs), then
    ``procedure(f2py_ai_*)`` declarations, then remaining specification
    statements — so USE never follows a declaration (gh-20157).
    """
    cb_set = {n.lower() for n in cb_orig_names}
    orig_by_lower = {}
    for n in cb_orig_names:
        orig_by_lower.setdefault(n.lower(), n)

    lines = saved_interface.split('\n')
    drop = [False] * len(lines)
    i = 0
    while i < len(lines):
        s = lines[i].strip().lower()
        if s == 'interface' or (s.startswith('interface ') and not s.startswith('end')):
            start = i
            depth = 1
            j = i + 1
            chunk = [lines[i]]
            while j < len(lines) and depth:
                sj = lines[j].strip().lower()
                if sj == 'interface' or (
                        sj.startswith('interface ') and not sj.startswith('end')):
                    depth += 1
                elif sj.startswith('end interface'):
                    depth -= 1
                chunk.append(lines[j])
                j += 1
            block_l = '\n'.join(chunk).lower()
            if any(
                f'function {cb}(' in block_l
                or f'subroutine {cb}(' in block_l
                or f'function {cb} ' in block_l
                or f'subroutine {cb} ' in block_l
                for cb in cb_set
            ):
                for k in range(start, j):
                    drop[k] = True
            i = j
            continue
        i += 1

    header = []
    use_lines = []
    other = []
    saw_header = False
    for idx, line in enumerate(lines):
        if drop[idx]:
            continue
        s = line.strip().lower()
        if s.startswith('external'):
            rest = s[len('external'):].lstrip(' :')
            names = [n.strip() for n in rest.split(',') if n.strip()]
            if names and all(n in cb_set for n in names):
                continue
        if not saw_header:
            header.append(line)
            is_header = (
                s.startswith('function ') or s.startswith('subroutine ')
                or ' function ' in f' {s}' or ' subroutine ' in f' {s}'
            )
            if is_header:
                saw_header = True
            continue
        if s.startswith('use '):
            # Drop f2py __user__ USE; real Fortran modules keep.
            if '__user__' in s:
                continue
            use_lines.append(line)
        else:
            other.append(line)

    out = list(header)
    out.append(f'          use {mod_name}')
    out.extend(use_lines)
    for low in sorted(orig_by_lower):
        abs_name = abs_map[low]
        out.append(
            f'          procedure({abs_name}) :: {orig_by_lower[low]}'
        )
    out.extend(other)
    return '\n'.join(out)


def _prepare_callback_module(rout, args, vars, need_interface):
    """Return (module_src, mod_name, cb_orig_names, abs_map).

    *abs_map* maps local dummy lower-name -> abstract interface body name.
    """
    empty = ('', None, [], {})
    if not need_interface:
        return empty
    all_blocks = _callback_routine_blocks(rout)
    needed = {}
    orig_names = []
    for a in args:
        if not isexternal(vars[a]):
            continue
        block = all_blocks.get(a.lower())
        if block is not None:
            needed[a.lower()] = block
            orig_names.append(a)
    if not needed:
        return empty
    taken = {a.lower() for a in args}
    taken.update(k.lower() for k in (vars or {}))
    taken.update(e.lower() for e in (rout.get('externals') or []))
    abs_map = {
        a.lower(): _abstract_iface_name(a.lower(), taken)
        for a in orig_names
    }
    mod_name = _cb_iface_module_name(rout, taken)
    return (
        _build_callback_iface_module(
            mod_name, needed, host_rout=rout, abs_map=abs_map),
        mod_name,
        orig_names,
        abs_map,
    )


def _declare_external_args(args, vars, cb_orig_names, abs_map, add):
    """Declare procedure(...) for module-backed callbacks, else EXTERNAL."""
    cb_lower = {n.lower() for n in cb_orig_names}
    dumped = []
    for a in args:
        if not isexternal(vars[a]):
            continue
        if a.lower() in cb_lower:
            add(f'procedure({abs_map[a.lower()]}) :: {a}')
            dumped.append(a)
            continue
        add(f'external {a}')
        dumped.append(a)
    return dumped


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

    args = args[1:]
    # Callback interface module (gh-20157): one definition, USE'd in the
    # wrapper and in the nested host interface.
    # f90mode (module CONTAINS): no nested host interface, skip module path.
    module_src, cb_mod_name, cb_via_module, abs_map = _prepare_callback_module(
        rout, args, vars, need_interface and not f90mode)

    if need_interface:
        for line in rout['saved_interface'].split('\n'):
            if line.lstrip().startswith('use ') and '__user__' not in line:
                add(line)
        if cb_mod_name:
            add(f'use {cb_mod_name}')

    dumped_args = _declare_external_args(
        args, vars, cb_via_module, abs_map, add)
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
            # Module CONTAINS path: use modulename only; no dual EXTERNAL+
            # interface conflict to fix (gh-20157 applies to free routines).
            pass
        else:
            saved = rout['saved_interface']
            if cb_mod_name:
                saved = _rewrite_saved_interface_use_module(
                    saved, cb_via_module, cb_mod_name, abs_map)
            add('interface')
            add(saved.lstrip())
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
    return module_src, ret[0]


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

    module_src, cb_mod_name, cb_via_module, abs_map = _prepare_callback_module(
        rout, args, vars, need_interface and not f90mode)

    if need_interface:
        for line in rout['saved_interface'].split('\n'):
            if line.lstrip().startswith('use ') and '__user__' not in line:
                add(line)
        if cb_mod_name:
            add(f'use {cb_mod_name}')

    dumped_args = _declare_external_args(
        args, vars, cb_via_module, abs_map, add)
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
            # Module CONTAINS path: no dual EXTERNAL+interface conflict.
            pass
        else:
            saved = rout['saved_interface']
            if cb_mod_name:
                saved = _rewrite_saved_interface_use_module(
                    saved, cb_via_module, cb_mod_name, abs_map)
            add('interface')
            for line in saved.split('\n'):
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
    return module_src, ret[0]




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
    return rout, ('', '')
