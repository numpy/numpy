#!/usr/bin/env python3
"""
takes templated file .xxx.pyt and produces .xxx file  where .xxx is
.i or .c or .h, using the following template rules

"""
import os, sys, re, textwrap, io, shutil
from textwrap import dedent, indent

__all__ = ['process_file', 'process_compile', 'interpret']
########################################################
### Template Utilities
########################################################
def p(*args, **kwargs):
    kwargs['end'] = kwargs.get('end', '')
    print(*args, **kwargs, flush=True)

_translate = dict()
def tr(r, *args):
    global _translate
    if not args:
        _translate.pop(r)
    else:
        _translate[r] = ''.join([str(a) for a in args])
    return ''

def clear_tr():
    global _translate
    _translate.clear()
    return ''

def translate(astr):
    global _translate
    for r, rwith in _translate.items():
        astr = astr.replace(r, rwith)
    return astr

_optree_cache = {}
def include(pyas_file, gl={}):
    optree = _optree_cache.get(pyas_file)
    if not optree:
        # TODO: cache into temp files
        _optree_cache[pyas_file] = optree = process_compile(pyas_file)
    exec(optree, globals(), gl)

########################################################
### The interpreter
########################################################
def _fetch_tags(lines, tags, plain_cb, lineno=0):
    ntags = []
    for sta_sym, end_sym, tag_cb, kwargs in tags:
        ntags.append((
            sta_sym, len(sta_sym),
            end_sym, len(end_sym),
            tag_cb, kwargs
        ))

    sta_pos = -1; end_pos = -1
    tag_lines = []; plain_lines = []; final_lines=[]
    for lineno, line in enumerate(lines, lineno+1):
        while True:
            if sta_pos == -1:
                for (
                    sta_sym, sta_len, end_sym, end_len,
                    tag_cb, kwargs
                ) in ntags:
                    sta_pos = line.find(sta_sym)
                    if sta_pos != -1:
                        break
                if sta_pos == -1:
                    plain_lines.append(line)
                    break
                # get str located before the tag
                before_tag = line[:sta_pos]
                if before_tag:
                    plain_lines.append(before_tag)
                if plain_lines:
                    final_lines += plain_cb(plain_lines, lineno - len(plain_lines))
                    plain_lines.clear()

                # shift to after the tag
                line = line[sta_pos+sta_len:]

            end_pos = line.find(end_sym)
            if end_pos == -1:
                tag_lines.append(line)
                break

            end_tag = line[:end_pos]
            if end_tag:
                tag_lines.append(end_tag)
            final_lines += tag_cb(tag_lines, lineno - len(tag_lines), **kwargs)
            tag_lines.clear()

            # re-parse the rest of the line.
            line = line[end_pos+end_len:]
            sta_pos = -1; end_pos = -1;
            if not line:
                break

    if sta_pos != -1:
        lineno = lineno - len(tag_lines)
        raise ValueError(f'Expected {end_sym} for {sta_sym}::{lineno}:{sta_pos+1}')

    if plain_lines:
        final_lines += plain_cb(plain_lines, lineno - len(plain_lines))
    return final_lines

def _easy_tpl(lines, lineno, wrap=None, indent=None):
    constants = (
        ('@LINENO@', len('@LINENO@'), lambda line, lineno, start, end:
            (line[:start] + str(lineno) + line[end:])
        ),
    )
    for i, line in enumerate(lines):
        for cname, lc, cb in constants:
            charno = line.find(cname)
            if charno == -1:
                continue
            line = cb(line, lineno + i, charno, charno+lc)
            lines[i] = line

    lines = ''.join(lines)
    if indent is not None:
        lines = textwrap.dedent(lines)
        lines = textwrap.indent(lines, ' '*indent)

    lines = lines.replace('{', '{{').replace('}', '}}')
    lines = lines.replace('{{{{', '{').replace('}}}}', '}')

    astr = f"translate(f'''{lines}''')"
    if wrap:
        return wrap(astr)
    return astr

def _py_tags(lines, lineno):
    wplain = lambda astr: f'p({astr})'
    tags = (
        ('<%%',  '%%>',  _easy_tpl, {}),
        ('<0%%', '%%0>', _easy_tpl, {'indent':4*0}),
        ('<1%%', '%%1>', _easy_tpl, {'indent':4*1}),
        ('<2%%', '%%2>', _easy_tpl, {'indent':4*2}),
        ('<3%%', '%%3>', _easy_tpl, {'indent':4*3}),
        ('<%',    '%>',  _easy_tpl, {'wrap':wplain}),
        ('<0%',  '%0>',  _easy_tpl, {'wrap':wplain, 'indent':4*0}),
        ('<1%',  '%1>',  _easy_tpl, {'wrap':wplain, 'indent':4*1}),
        ('<2%',  '%2>',  _easy_tpl, {'wrap':wplain, 'indent':4*2}),
        ('<3%',  '%3>',  _easy_tpl, {'wrap':wplain, 'indent':4*3}),
    )
    nop = lambda lines, lineno: lines
    return _fetch_tags(lines, tags, nop, lineno)

def _interpret(read_fd):
    pplain = lambda lines, lineno: [f"\np('''{''.join(lines)}''')"]
    wplain = lambda astr: f'\np({astr})'
    tags = (
        ('<3%', '%3>', _easy_tpl, {'wrap':wplain, 'indent':4*3}),
        ('<2%', '%2>', _easy_tpl, {'wrap':wplain, 'indent':4*2}),
        ('<1%', '%1>', _easy_tpl, {'wrap':wplain, 'indent':4*1}),
        ('<0%', '%0>', _easy_tpl, {'wrap':wplain, 'indent':4*0}),
        ('<%',  '%>',  _easy_tpl, {'wrap':wplain}),
        ('<?python', '?>',  _py_tags,  {}),
        ('<?',       '?>',  _py_tags,  {}),
    )
    return _fetch_tags(read_fd, tags, pplain)

def _run(source, outfile):
    source = os.path.abspath(source)
    outfile = os.path.abspath(outfile)

    os.chdir(os.path.dirname(source))
    sys.stdout = buf = io.StringIO()

    # TODO: cache into temp files
    optree = process_compile(source)
    exec(optree)
    with open(outfile, 'w') as fd:
        buf.seek(0)
        shutil.copyfileobj(buf, fd)

    buf.close()

def _run_exception(source, outfile, cpipe):
    import traceback
    try:
        _run(source, outfile)
        cpipe.send(None)
    except Exception:
        trace = traceback.format_exc()
        cpipe.send(trace)

def process_compile(source):
    # TODO: support file cache
    with open(source, 'r') as rfd:
        pycode = ''.join(_interpret(rfd))
        return compile(pycode, filename=source, mode='exec')

def process_file(source, outfile):
    from multiprocessing import Process, Pipe
    source = os.path.normcase(source).replace("\\", "\\\\")
    outfile = os.path.normcase(outfile).replace("\\", "\\\\")

    ppipe, cpipe = Pipe()
    p = Process(target=_run_exception, args=(source, outfile, cpipe))
    p.start()
    trace_back = ppipe.recv()
    p.join()
    if trace_back:
        raise SystemExit(trace_back)

def main():
    argc = len(sys.argv)
    if argc < 2:
        # TODO: What we suppose to do here?
        return
    if argc > 2:
        outfile = sys.argv[2]
    else:
        outfile = sys.argv[1].split(os.extsep)
        try:
            del outfile[-2]
        except IndexError:
            return
        outfile = os.extsep.join(outfile)

    _run(sys.argv[1], outfile)

if __name__ == "__main__":
    main()
