"""
Get API information encoded in C files.

See ``find_function`` for how functions should be formatted, and
``read_order`` for how the order of the functions should be
specified.

"""
from __future__ import division, absolute_import, print_function

import sys, os, re
import hashlib

import textwrap

from os.path import join

__docformat__ = 'restructuredtext'

# The files under src/ that are scanned for API functions
API_FILES = [join('multiarray', 'alloc.c'),
             join('multiarray', 'array_assign_array.c'),
             join('multiarray', 'array_assign_scalar.c'),
             join('multiarray', 'arrayobject.c'),
             join('multiarray', 'arraytypes.c.src'),
             join('multiarray', 'buffer.c'),
             join('multiarray', 'calculation.c'),
             join('multiarray', 'conversion_utils.c'),
             join('multiarray', 'convert.c'),
             join('multiarray', 'convert_datatype.c'),
             join('multiarray', 'ctors.c'),
             join('multiarray', 'datetime.c'),
             join('multiarray', 'datetime_busday.c'),
             join('multiarray', 'datetime_busdaycal.c'),
             join('multiarray', 'datetime_strings.c'),
             join('multiarray', 'descriptor.c'),
             join('multiarray', 'einsum.c.src'),
             join('multiarray', 'flagsobject.c'),
             join('multiarray', 'getset.c'),
             join('multiarray', 'item_selection.c'),
             join('multiarray', 'iterators.c'),
             join('multiarray', 'mapping.c'),
             join('multiarray', 'methods.c'),
             join('multiarray', 'multiarraymodule.c'),
             join('multiarray', 'nditer_api.c'),
             join('multiarray', 'nditer_constr.c'),
             join('multiarray', 'nditer_pywrap.c'),
             join('multiarray', 'nditer_templ.c.src'),
             join('multiarray', 'number.c'),
             join('multiarray', 'refcount.c'),
             join('multiarray', 'scalartypes.c.src'),
             join('multiarray', 'scalarapi.c'),
             join('multiarray', 'sequence.c'),
             join('multiarray', 'shape.c'),
             join('multiarray', 'strfuncs.c'),
             join('multiarray', 'usertypes.c'),
             join('umath', 'loops.c.src'),
             join('umath', 'ufunc_object.c'),
             join('umath', 'ufunc_type_resolution.c'),
             join('umath', 'reduction.c'),
            ]
THIS_DIR = os.path.dirname(__file__)
API_FILES = [os.path.join(THIS_DIR, '..', 'src', a) for a in API_FILES]

def file_in_this_dir(filename):
    return os.path.join(THIS_DIR, filename)

def remove_whitespace(s):
    return ''.join(s.split())

def _repl(str):
    return str.replace('Bool', 'npy_bool')


class StealRef(object):
    def __init__(self, arg):
        self.arg = arg # counting from 1

    def __str__(self):
        try:
            return ' '.join('NPY_STEALS_REF_TO_ARG(%d)' % x for x in self.arg)
        except TypeError:
            return 'NPY_STEALS_REF_TO_ARG(%d)' % self.arg


class NonNull(object):
    def __init__(self, arg):
        self.arg = arg # counting from 1

    def __str__(self):
        try:
            return ' '.join('NPY_GCC_NONNULL(%d)' % x for x in self.arg)
        except TypeError:
            return 'NPY_GCC_NONNULL(%d)' % self.arg


class Function(object):
    def __init__(self, name, return_type, args, doc=''):
        self.name = name
        self.return_type = _repl(return_type)
        self.args = args
        self.doc = doc

    def _format_arg(self, typename, name):
        if typename.endswith('*'):
            return typename + name
        else:
            return typename + ' ' + name

    def __str__(self):
        argstr = ', '.join([self._format_arg(*a) for a in self.args])
        if self.doc:
            doccomment = '/* %s */\n' % self.doc
        else:
            doccomment = ''
        return '%s%s %s(%s)' % (doccomment, self.return_type, self.name, argstr)

    def to_ReST(self):
        lines = ['::', '', '  ' + self.return_type]
        argstr = ',\000'.join([self._format_arg(*a) for a in self.args])
        name = '  %s' % (self.name,)
        s = textwrap.wrap('(%s)' % (argstr,), width=72,
                          initial_indent=name,
                          subsequent_indent=' ' * (len(name)+1),
                          break_long_words=False)
        for l in s:
            lines.append(l.replace('\000', ' ').rstrip())
        lines.append('')
        if self.doc:
            lines.append(textwrap.dedent(self.doc))
        return '\n'.join(lines)

    def api_hash(self):
        m = hashlib.md5()
        m.update(remove_whitespace(self.return_type))
        m.update('\000')
        m.update(self.name)
        m.update('\000')
        for typename, name in self.args:
            m.update(remove_whitespace(typename))
            m.update('\000')
        return m.hexdigest()[:8]

class ParseError(Exception):
    def __init__(self, filename, lineno, msg):
        self.filename = filename
        self.lineno = lineno
        self.msg = msg

    def __str__(self):
        return '%s:%s:%s' % (self.filename, self.lineno, self.msg)

def skip_brackets(s, lbrac, rbrac):
    count = 0
    for i, c in enumerate(s):
        if c == lbrac:
            count += 1
        elif c == rbrac:
            count -= 1
        if count == 0:
            return i
    raise ValueError("no match '%s' for '%s' (%r)" % (lbrac, rbrac, s))

def split_arguments(argstr):
    arguments = []
    bracket_counts = {'(': 0, '[': 0}
    current_argument = []
    state = 0
    i = 0
    def finish_arg():
        if current_argument:
            argstr = ''.join(current_argument).strip()
            m = re.match(r'(.*(\s+|[*]))(\w+)$', argstr)
            if m:
                typename = m.group(1).strip()
                name = m.group(3)
            else:
                typename = argstr
                name = ''
            arguments.append((typename, name))
            del current_argument[:]
    while i < len(argstr):
        c = argstr[i]
        if c == ',':
            finish_arg()
        elif c == '(':
            p = skip_brackets(argstr[i:], '(', ')')
            current_argument += argstr[i:i+p]
            i += p-1
        else:
            current_argument += c
        i += 1
    finish_arg()
    return arguments


def find_functions(filename, tag='API'):
    """
    Scan the file, looking for tagged functions.

    Assuming ``tag=='API'``, a tagged function looks like::

        /*API*/
        static returntype*
        function_name(argtype1 arg1, argtype2 arg2)
        {
        }

    where the return type must be on a separate line, the function
    name must start the line, and the opening ``{`` must start the line.

    An optional documentation comment in ReST format may follow the tag,
    as in::

        /*API
          This function does foo...
         */
    """
    fo = open(filename, 'r')
    functions = []
    return_type = None
    function_name = None
    function_args = []
    doclist = []
    SCANNING, STATE_DOC, STATE_RETTYPE, STATE_NAME, STATE_ARGS = list(range(5))
    state = SCANNING
    tagcomment = '/*' + tag
    for lineno, line in enumerate(fo):
        try:
            line = line.strip()
            if state == SCANNING:
                if line.startswith(tagcomment):
                    if line.endswith('*/'):
                        state = STATE_RETTYPE
                    else:
                        state = STATE_DOC
            elif state == STATE_DOC:
                if line.startswith('*/'):
                    state = STATE_RETTYPE
                else:
                    line = line.lstrip(' *')
                    doclist.append(line)
            elif state == STATE_RETTYPE:
                # first line of declaration with return type
                m = re.match(r'NPY_NO_EXPORT\s+(.*)$', line)
                if m:
                    line = m.group(1)
                return_type = line
                state = STATE_NAME
            elif state == STATE_NAME:
                # second line, with function name
                m = re.match(r'(\w+)\s*\(', line)
                if m:
                    function_name = m.group(1)
                else:
                    raise ParseError(filename, lineno+1,
                                     'could not find function name')
                function_args.append(line[m.end():])
                state = STATE_ARGS
            elif state == STATE_ARGS:
                if line.startswith('{'):
                    # finished
                    fargs_str = ' '.join(function_args).rstrip(' )')
                    fargs = split_arguments(fargs_str)
                    f = Function(function_name, return_type, fargs,
                                 '\n'.join(doclist))
                    functions.append(f)
                    return_type = None
                    function_name = None
                    function_args = []
                    doclist = []
                    state = SCANNING
                else:
                    function_args.append(line)
        except Exception:
            print(filename, lineno + 1)
            raise
    fo.close()
    return functions

def should_rebuild(targets, source_files):
    from distutils.dep_util import newer_group
    for t in targets:
        if not os.path.exists(t):
            return True
    sources = API_FILES + list(source_files) + [__file__]
    if newer_group(sources, targets[0], missing='newer'):
        return True
    return False

def write_file(filename, data):
    """
    Write data to filename
    Only write changed data to avoid updating timestamps unnecessarily
    """
    if os.path.exists(filename):
        with open(filename) as f:
            if data == f.read():
                return

    with open(filename, 'w') as fid:
        fid.write(data)


# Those *Api classes instances know how to output strings for the generated code
class TypeApi(object):
    def __init__(self, name, index, ptr_cast, api_name):
        self.index = index
        self.name = name
        self.ptr_cast = ptr_cast
        self.api_name = api_name

    def define_from_array_api_string(self):
        return "#define %s (*(%s *)%s[%d])" % (self.name,
                                               self.ptr_cast,
                                               self.api_name,
                                               self.index)

    def array_api_define(self):
        return "        (void *) &%s" % self.name

    def internal_define(self):
        astr = """\
extern NPY_NO_EXPORT PyTypeObject %(type)s;
""" % {'type': self.name}
        return astr

class GlobalVarApi(object):
    def __init__(self, name, index, type, api_name):
        self.name = name
        self.index = index
        self.type = type
        self.api_name = api_name

    def define_from_array_api_string(self):
        return "#define %s (*(%s *)%s[%d])" % (self.name,
                                                        self.type,
                                                        self.api_name,
                                                        self.index)

    def array_api_define(self):
        return "        (%s *) &%s" % (self.type, self.name)

    def internal_define(self):
        astr = """\
extern NPY_NO_EXPORT %(type)s %(name)s;
""" % {'type': self.type, 'name': self.name}
        return astr

# Dummy to be able to consistently use *Api instances for all items in the
# array api
class BoolValuesApi(object):
    def __init__(self, name, index, api_name):
        self.name = name
        self.index = index
        self.type = 'PyBoolScalarObject'
        self.api_name = api_name

    def define_from_array_api_string(self):
        return "#define %s ((%s *)%s[%d])" % (self.name,
                                              self.type,
                                              self.api_name,
                                              self.index)

    def array_api_define(self):
        return "        (void *) &%s" % self.name

    def internal_define(self):
        astr = """\
extern NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[2];
"""
        return astr

class FunctionApi(object):
    def __init__(self, name, index, annotations, return_type, args, api_name):
        self.name = name
        self.index = index
        self.annotations = annotations
        self.return_type = return_type
        self.args = args
        self.api_name = api_name

    def _argtypes_string(self):
        if not self.args:
            return 'void'
        argstr = ', '.join([_repl(a[0]) for a in self.args])
        return argstr

    def define_from_array_api_string(self):
        define = """\
#define %s \\\n        (*(%s (*)(%s)) \\
         %s[%d])""" % (self.name,
                                self.return_type,
                                self._argtypes_string(),
                                self.api_name,
                                self.index)
        return define

    def array_api_define(self):
        return "        (void *) %s" % self.name

    def internal_define(self):
        annstr = []
        for a in self.annotations:
            annstr.append(str(a))
        annstr = ' '.join(annstr)
        astr = """\
NPY_NO_EXPORT %s %s %s \\\n       (%s);""" % (annstr, self.return_type,
                                              self.name,
                                              self._argtypes_string())
        return astr

def order_dict(d):
    """Order dict by its values."""
    o = list(d.items())
    def _key(x):
        return x[1] + (x[0],)
    return sorted(o, key=_key)

def merge_api_dicts(dicts):
    ret = {}
    for d in dicts:
        for k, v in d.items():
            ret[k] = v

    return ret

def check_api_dict(d):
    """Check that an api dict is valid (does not use the same index twice)."""
    # remove the extra value fields that aren't the index
    index_d = {k: v[0] for k, v in d.items()}

    # We have if a same index is used twice: we 'revert' the dict so that index
    # become keys. If the length is different, it means one index has been used
    # at least twice
    revert_dict = {v: k for k, v in index_d.items()}
    if not len(revert_dict) == len(index_d):
        # We compute a dict index -> list of associated items
        doubled = {}
        for name, index in index_d.items():
            try:
                doubled[index].append(name)
            except KeyError:
                doubled[index] = [name]
        fmt = "Same index has been used twice in api definition: {}"
        val = ''.join(
            '\n\tindex {} -> {}'.format(index, names)
            for index, names in doubled.items() if len(names) != 1
        )
        raise ValueError(fmt.format(val))

    # No 'hole' in the indexes may be allowed, and it must starts at 0
    indexes = set(index_d.values())
    expected = set(range(len(indexes)))
    if indexes != expected:
        diff = expected.symmetric_difference(indexes)
        msg = "There are some holes in the API indexing: " \
              "(symmetric diff is %s)" % diff
        raise ValueError(msg)

def get_api_functions(tagname, api_dict):
    """Parse source files to get functions tagged by the given tag."""
    functions = []
    for f in API_FILES:
        functions.extend(find_functions(f, tagname))
    dfunctions = []
    for func in functions:
        o = api_dict[func.name][0]
        dfunctions.append( (o, func) )
    dfunctions.sort()
    return [a[1] for a in dfunctions]

def fullapi_hash(api_dicts):
    """Given a list of api dicts defining the numpy C API, compute a checksum
    of the list of items in the API (as a string)."""
    a = []
    for d in api_dicts:
        for name, data in order_dict(d):
            a.extend(name)
            a.extend(','.join(map(str, data)))

    return hashlib.md5(''.join(a).encode('ascii')).hexdigest()

# To parse strings like 'hex = checksum' where hex is e.g. 0x1234567F and
# checksum a 128 bits md5 checksum (hex format as well)
VERRE = re.compile(r'(^0x[\da-f]{8})\s*=\s*([\da-f]{32})')

def get_versions_hash():
    d = []

    file = os.path.join(os.path.dirname(__file__), 'cversions.txt')
    fid = open(file, 'r')
    try:
        for line in fid:
            m = VERRE.match(line)
            if m:
                d.append((int(m.group(1), 16), m.group(2)))
    finally:
        fid.close()

    return dict(d)

def main():
    tagname = sys.argv[1]
    order_file = sys.argv[2]
    functions = get_api_functions(tagname, order_file)
    m = hashlib.md5(tagname)
    for func in functions:
        print(func)
        ah = func.api_hash()
        m.update(ah)
        print(hex(int(ah, 16)))
    print(hex(int(m.hexdigest()[:8], 16)))

if __name__ == '__main__':
    main()
