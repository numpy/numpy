#!/usr/bin/env python3
'''
Python as a templating language(`PyAS`)
---------------------------------------

If ``.src`` template does not fit your requirements then take the gloves off and fall-back to Python!.
`PyAS` doesn't come with a custom syntax just few tags aims to simplifying the use of Python in
generating C code, the way it works is very similar to the PHP language, so all you have to do is
writing the Python code between special tags. e.g. ``<? print("Hello World!") ?>``.

Same as ``.src``, ``PyAS`` requires special extention ``.pyas``, it takes template file `.xxx.pyas`
and produces `.xxx` file where `.xxx` is .i or .c or .h.

Currently ``PyAS`` supports three tags can be implied by any kind of syntax that are explained as follows:

``<? ?>`` to execute python code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: NumPyC

    #include "numpy/npy_common.h"
    <?
    for type in "byte ubyte short ushort".split(): print(f"""
        static void
        TIMEDELTA_to_{type.upper()}(void *input, void *output, npy_intp n, void *aip, void *aop)
        {{
             const npy_timedelta *ip = input;
             npy_{type} *op = output;
             while (n--) {{
                 *op++ = (npy_{type})*ip++;
             }}
        }}
        """)
    ?>

Just a pure Python code with f-strings and the only thing is need to mention here is function ``print()``,
since ``PyAS`` redirects `stdout` to the final generated source.

``<%% %%>`` friendly f-strings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
f-strings aren't friendly with *C* language since replacement fields surrounded by curly braces and it
+ requires to doubling the braces in order to escape it.

And here comes the role of this tag. It simply inverses the braces single to double and vice versa,
+ also provides extra options for field replacements and string indentations. for example:

.. code-block:: NumPyC

    #include "numpy/npy_common.h"
    <?
    for type in "byte ubyte short ushort".split(): print(<%%
        static void
        TIMEDELTA_to_{{type.upper()}}(void *input, void *output, npy_intp n, void *aip, void *aop)
        {
             const npy_timedelta *ip = input;
             npy_{{type}} *op = output;
             while (n--) {
                 *op++ = (npy_{{type}})*ip++;
             }
        }
        %%>)
    ?>

The use of curly double braces inside C-code may be a little annoying, especially with large codes,
+ or when we include multiple field replacements in adjacent lines, therefor this tag provides two
+ more options for string replacements:

**Local on fly**: any "local" variables can be replaced directly if the variable name starts with
+ capital letter and in-placed with dollar sign '$'. e.g.

.. code-block:: NumPyC

    <?
    for Type in "byte ubyte short ushort".split():
        print(<%%
        //...
        npy_$Type *op = output;
        while (n--) {
          *op++ = (npy_$Type)*ip++;
        }
        //...
        %%>)
    ?>

**Custom translations**: similar to "local on fly" except the replaced tokens specified via template
+ utility function ``tr()`` e.g.

.. code-block:: NumPyC

    <?
    for type in "byte ubyte short ushort".split():
        tr(auto=f"npy_{type})")
        print(<%%
        //...
        auto *op = output;
        while (n--) {
          *op++ = (auto)*ip++;
        }
        //...
        %%>)
    ?>

**Note**: the translation domain is local and subject to the frame that calls `tr()`

The last thing this tag trying to solve is the indentation, beautify the output can simplify
+ tracing errors and that can be easily done by adding the number of the required tabs next to the beginning of tag.

.. code-block:: NumPyC

    <?
    <%%0 // equivalent to textwrap.dedent(str) %%>
    <%%1 // equivalent to textwrap.indent(textwrap.dedent(str), ' '*4) %%>
    <%%2 // equivalent to textwrap.indent(textwrap.dedent(str), ' '*8) %%>
    ?>

**Note**: The maximum number of tabs is **nine** and each tab adds **four** spaces.

``<% %>`` print friendly f-strings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Its equivalent to ``print(<%%...%%>, end='', flush=True)``, same as the previous tag except it can
+ be reachable with or without Python tags.

.. code-block:: NumPyC

    #include "numpy/npy_common.h"
    <?for Type in "byte ubyte short ushort".split(): <%
        //...
        npy_$Type *op = output;
        while (n--) {
          *op++ = (npy_$Type)*ip++;
        }
        //...
    %>?>

Template utility
^^^^^^^^^^^^^^^^
'''
__all__ = ['lineno', 'filename', 'p', 'tr', 'include', 'process_file', 'process_str']

import os, sys, re, textwrap, io, shutil, inspect, ast, traceback
from contextlib import contextmanager
from textwrap import dedent, indent

########################################################
### Template utility
########################################################

def lineno():
    """
    Return the current line number.
    """
    return inspect.currentframe().f_back.f_lineno

def filename():
    """
    Return template filename.
    """
    return inspect.getframeinfo(inspect.currentframe().f_back).filename

def p(*args, **kwargs):
    """
    Print with no end line.
    """
    kwargs['end'] = kwargs.get('end', '')
    print(*args, **kwargs, flush=True)

def tr(**kwargs):
    """
    Translate certain tokens in f-strings tags.
    """
    _phrases = inspect.currentframe().f_back.f_locals.setdefault('_phrases', {})
    _phrases.update(kwargs)
    return ''

def _tr(astr):
    """
    internally used by `_process_interpret()` to translate the tokens
    in f-strings.
    """
    clocals = inspect.currentframe().f_back.f_locals
    lc = {'$'+k : str(v) for k, v in clocals.items() if k[0].isupper()}
    lc.update(clocals.get('_phrases', {}))
    for k, v in sorted(lc.items(), key=lambda k: len(k[0]), reverse=True):
        astr = astr.replace(k, v)
    return astr

_optree_cache = {}
def include(source, once=False):
    """
    Parse and execute a PyAS template file
    """
    source = os.path.abspath(source)
    optree = _optree_cache.get(source)
    if not optree:
        # TODO: cache into temp files
        with open(source, 'r') as rfd:
            astr = rfd.read()
        _optree_cache[source] = optree = _compile_pyas(astr, source)
    elif once:
        return
    exec(optree, globals(), globals())

########################################################
### The interpreter
########################################################
class _tag:
    """
    Properties of tag

    Parameters
    ----------

    start: str
        opening tag symbol without '<'

    end: str
        closing tag symbol without '>'

    callback: Callback[[str, lineno:int]]
        for raw text falls between the tag
    """
    def __init__(self, start, end, callback):
        self.start  = start
        self.end    = end
        self.cb     = callback

def _parse_tags(filename, astr, plain_cb, tags, pos=0, lineno=0):
    """
    Find all tags in `astr` according to a list of `_tag`

    Parameters
    ----------
    plain_cb: Callback[[str, lineno:int]]
        for raw text before the opening tag

    tags: Iterable[_tag]
        set of tags to find it in `astr`

    pos: int
        start position of `astr`

    lineno: int
        start position of line number counter
    """
    strln = len(astr)
    def next_char(char):
        nonlocal pos, lineno
        while pos < strln:
            c = astr[pos]
            pos += 1
            if c == '\n':
                lineno += 1
                continue
            if c == char:
                return True

    def next_start():
        nonlocal pos
        while next_char('<'):
            for t in tags:
                if astr.startswith(t.start, pos):
                    pos += len(t.start)
                    return t

    def next_end(tag):
        while next_char('>'):
            if astr.endswith(tag.end, 0, pos-1):
                return tag

    while 1:
        t_pos, t_lineno = pos, lineno
        tag = next_start()
        if not tag:
            plain_cb(astr[t_pos:], t_lineno)
            break
        plain_cb(astr[t_pos:pos-len(tag.start)-1], t_lineno)

        t_pos, t_lineno = pos, lineno
        if not next_end(tag):
            close_begin = t_pos-len(tag.start)-1
            close_start = astr[close_begin:close_begin+100]
            close_start = textwrap.indent(close_start, ' '*4)
            raise SyntaxError(
                f"Closing tag missing '{tag.end}>'", (filename, lineno, 1,
                    f"The opening tag '<{tag.end}' was found on line {t_lineno}\n{close_start}"
                )
            )
        tag.cb(astr[t_pos:pos-len(tag.end)-1], t_lineno)

def _compile_pyas(astr, filename=None):
    """
    Convert templates tags in `astr` into Python code object,
    any text falls out side the tags printed as raw text.

    Currently its only supports three kind of tags explained in the doc
    string of this file.
    """
    optree = ast.parse('', filename=filename)
    last_tpl_lineno = 0
    def interpret_ast(code, lineno):
        try:
            cp = ast.parse(code, filename=filename)
        except SyntaxError as e:
            e.filename = filename
            e.lineno += lineno
            raise
        ast.increment_lineno(cp, lineno)
        return cp

    def add_ast(*args):
        nonlocal optree
        optree.body += interpret_ast(*args).body

    def wrap_tpl(astr, lineno, p=False):
        try:
            indent = int(astr[0])*4
            astr = textwrap.dedent(astr[1:])
            astr = textwrap.indent(astr, ' '*indent)
        except ValueError:
            indent = -1
        astr = astr.replace('{', '{{').replace('}', '}}')
        astr = astr.replace('{{{{', '{').replace('}}}}', '}')
        astr = "_tr(f'''" + astr + "''')"
        # dummy parsing to correct line number error of multi-line f-strings
        interpret_ast(astr, lineno)
        if p:
            return 'p('+astr+')'
        return astr

    def visit_py(astr, lineno):
        astr = textwrap.dedent(astr)
        pycode_sub = []
        tags_sub = (
            _tag('%%', '%%', lambda a, *args: pycode_sub.append(wrap_tpl(a, *args))),
            _tag('%',   '%', lambda a, *args: pycode_sub.append(wrap_tpl(a, *args, p=True)))
        )
        plain_cb = lambda a, *_: pycode_sub.append(a)
        _parse_tags(filename, astr, plain_cb, tags_sub, lineno=lineno)
        add_ast(''.join(pycode_sub), lineno)

    def visit_tpl(astr, *args):
        add_ast(wrap_tpl(astr, *args, p=True), *args)

    def visit_plain(astr, *args):
        add_ast(f"p('''{astr}''')", *args)

    _parse_tags(filename, astr, visit_plain, (
        _tag('?', '?', visit_py),
        _tag('%', '%', visit_tpl),
    ))
    return compile(optree, filename=filename, mode='exec', dont_inherit=True)

@contextmanager
def _redirect_os(path, stdout):
    bk_cwd, bk_stdout = os.getcwd(), sys.stdout
    os.chdir(path)
    sys.stdout = stdout
    try:
        yield
    finally:
        os.chdir(bk_cwd)
        sys.stdout = bk_stdout

def process_str(astr, filename=__file__):
    # TODO: cache into temp file
    optree = _compile_pyas(astr, filename)
    stdout = io.StringIO()
    srcdir = os.path.dirname(filename)
    with _redirect_os(srcdir, stdout):
        exec(optree, globals())
    stdout.seek(0)
    return stdout

def process_file(source, outfile):
    source = os.path.abspath(source)
    with open(source, 'r') as rfd:
        astr = rfd.read()
    stdout = process_str(astr, source)
    outfile = os.path.abspath(outfile)
    with open(outfile, 'w') as fd:
        shutil.copyfileobj(stdout, fd)
    stdout.close()

def main():
    argc = len(sys.argv)
    if argc < 2:
        raise SystemExit("requires template path as an argument")
    elif argc > 2:
        outfile = sys.argv[2]
    else:
        outfile = sys.argv[1].split(os.extsep)
        try:
            del outfile[-2]
        except IndexError:
            return
        outfile = os.extsep.join(outfile)

    process_file(sys.argv[1], outfile)

if __name__ == "__main__":
    main()
