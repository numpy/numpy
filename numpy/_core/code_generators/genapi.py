"""
Get API information encoded in C files.

See ``find_function`` for how functions should be formatted, and
``read_order`` for how the order of the functions should be
specified.

"""
import hashlib
import importlib.util
import io
import os
import re
import sys
import textwrap
from os.path import join


def get_processor():
    # Convoluted because we can't import from numpy.distutils
    # (numpy is not yet built)
    conv_template_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'distutils', 'conv_template.py'
    )
    spec = importlib.util.spec_from_file_location(
        'conv_template', conv_template_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.process_file


process_c_file = get_processor()


__docformat__ = 'restructuredtext'

# The files under src/ that are scanned for API functions
API_FILES = [join('multiarray', 'alloc.c'),
             join('multiarray', 'abstractdtypes.c'),
             join('multiarray', 'arrayfunction_override.c'),
             join('multiarray', 'array_api_standard.c'),
             join('multiarray', 'array_assign_array.c'),
             join('multiarray', 'array_assign_scalar.c'),
             join('multiarray', 'array_coercion.c'),
             join('multiarray', 'array_converter.c'),
             join('multiarray', 'array_method.c'),
             join('multiarray', 'arrayobject.c'),
             join('multiarray', 'arraytypes.c.src'),
             join('multiarray', 'buffer.c'),
             join('multiarray', 'calculation.c'),
             join('multiarray', 'common_dtype.c'),
             join('multiarray', 'conversion_utils.c'),
             join('multiarray', 'convert.c'),
             join('multiarray', 'convert_datatype.c'),
             join('multiarray', 'ctors.c'),
             join('multiarray', 'datetime.c'),
             join('multiarray', 'datetime_busday.c'),
             join('multiarray', 'datetime_busdaycal.c'),
             join('multiarray', 'datetime_strings.c'),
             join('multiarray', 'descriptor.c'),
             join('multiarray', 'dlpack.c'),
             join('multiarray', 'dtypemeta.c'),
             join('multiarray', 'einsum.c.src'),
             join('multiarray', 'public_dtype_api.c'),
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
             join('multiarray', 'stringdtype', 'static_string.c'),
             join('multiarray', 'strfuncs.c'),
             join('multiarray', 'usertypes.c'),
             join('umath', 'dispatching.cpp'),
             join('umath', 'extobj.c'),
             join('umath', 'loops.c.src'),
             join('umath', 'reduction.c'),
             join('umath', 'ufunc_object.c'),
             join('umath', 'ufunc_type_resolution.c'),
             join('umath', 'wrapping_array_method.c'),
            ]
THIS_DIR = os.path.dirname(__file__)
API_FILES = [os.path.join(THIS_DIR, '..', 'src', a) for a in API_FILES]

def file_in_this_dir(filename):
    return os.path.join(THIS_DIR, filename)

def remove_whitespace(s):
    return ''.join(s.split())

def _repl(str):
    return str.replace('Bool', 'npy_bool')


class MinVersion:
    def __init__(self, version):
        """ Version should be the normal NumPy version, e.g. "1.25" """
        major, minor = version.split(".")
        self.version = f"NPY_{major}_{minor}_API_VERSION"

    def __str__(self):
        # Used by version hashing:
        return self.version

    def add_guard(self, name, normal_define):
        """Wrap a definition behind a version guard"""
        wrap = textwrap.dedent(f"""
            #if NPY_FEATURE_VERSION >= {self.version}
            {{define}}
            #endif""")

        # we only insert `define` later to avoid confusing dedent:
        return wrap.format(define=normal_define)


class StealRef:
    def __init__(self, arg):
        self.arg = arg  # counting from 1

    def __str__(self):
        try:
            return ' '.join('NPY_STEALS_REF_TO_ARG(%d)' % x for x in self.arg)
        except TypeError:
            return 'NPY_STEALS_REF_TO_ARG(%d)' % self.arg


class Function:
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
            doccomment = f'/* {self.doc} */\n'
        else:
            doccomment = ''
        return f'{doccomment}{self.return_type} {self.name}({argstr})'

    def api_hash(self):
        m = hashlib.md5(usedforsecurity=False)
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
        return f'{self.filename}:{self.lineno}:{self.msg}'

def skip_brackets(s, lbrac, rbrac):
    count = 0
    for i, c in enumerate(s):
        if c == lbrac:
            count += 1
        elif c == rbrac:
            count -= 1
        if count == 0:
            return i
    raise ValueError(f"no match '{lbrac}' for '{rbrac}' ({s!r})")

def split_arguments(argstr):
    arguments = []
    current_argument = []
    i = 0

    def finish_arg():
        if current_argument:
            argstr = ''.join(current_argument).strip()
            m = re.match(r'(.*(\s+|\*))(\w+)$', argstr)
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
            current_argument += argstr[i:i + p]
            i += p - 1
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
    if filename.endswith(('.c.src', '.h.src')):
        fo = io.StringIO(process_c_file(filename))
    else:
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
                    raise ParseError(filename, lineno + 1,
                                     'could not find function name')
                function_args.append(line[m.end():])
                state = STATE_ARGS
            elif state == STATE_ARGS:
                if line.startswith('{'):
                    # finished
                    # remove any white space and the closing bracket:
                    fargs_str = ' '.join(function_args).rstrip()[:-1].rstrip()
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
        except ParseError:
            raise
        except Exception as e:
            msg = "see chained exception for details"
            raise ParseError(filename, lineno + 1, msg) from e
    fo.close()
    return functions


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
class TypeApi:
    def __init__(self, name, index, ptr_cast, api_name, internal_type=None):
        self.index = index
        self.name = name
        self.ptr_cast = ptr_cast
        self.api_name = api_name
        # The type used internally, if None, same as exported (ptr_cast)
        self.internal_type = internal_type

    def define_from_array_api_string(self):
        return "#define %s (*(%s *)%s[%d])" % (self.name,
                                               self.ptr_cast,
                                               self.api_name,
                                               self.index)

    def array_api_define(self):
        return f"        (void *) &{self.name}"

    def internal_define(self):
        if self.internal_type is None:
            return f"extern NPY_NO_EXPORT {self.ptr_cast} {self.name};\n"

        # If we are here, we need to define a larger struct internally, which
        # the type can be cast safely. But we want to normally use the original
        # type, so name mangle:
        mangled_name = f"{self.name}Full"
        astr = (
            # Create the mangled name:
            f"extern NPY_NO_EXPORT {self.internal_type} {mangled_name};\n"
            # And define the name as: (*(type *)(&mangled_name))
            f"#define {self.name} (*({self.ptr_cast} *)(&{mangled_name}))\n"
        )
        return astr

class GlobalVarApi:
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
        return f"        ({self.type} *) &{self.name}"

    def internal_define(self):
        astr = f"""extern NPY_NO_EXPORT {self.type} {self.name};
"""
        return astr

# Dummy to be able to consistently use *Api instances for all items in the
# array api
class BoolValuesApi:
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
        return f"        (void *) &{self.name}"

    def internal_define(self):
        astr = """\
extern NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[2];
"""
        return astr

class FunctionApi:
    def __init__(self, name, index, annotations, return_type, args, api_name):
        self.name = name
        self.index = index

        self.min_version = None
        self.annotations = []
        for annotation in annotations:
            # String checks, because manual import breaks isinstance
            if type(annotation).__name__ == "StealRef":
                self.annotations.append(annotation)
            elif type(annotation).__name__ == "MinVersion":
                if self.min_version is not None:
                    raise ValueError("Two minimum versions specified!")
                self.min_version = annotation
            else:
                raise ValueError(f"unknown annotation {annotation}")

        self.return_type = return_type
        self.args = args
        self.api_name = api_name

    def _argtypes_string(self):
        if not self.args:
            return 'void'
        argstr = ', '.join([_repl(a[0]) for a in self.args])
        return argstr

    def define_from_array_api_string(self):
        arguments = self._argtypes_string()
        define = textwrap.dedent(f"""\
            #define {self.name} \\
                    (*({self.return_type} (*)({arguments})) \\
                {self.api_name}[{self.index}])""")

        if self.min_version is not None:
            define = self.min_version.add_guard(self.name, define)
        return define

    def array_api_define(self):
        return f"        (void *) {self.name}"

    def internal_define(self):
        annstr = [str(a) for a in self.annotations]
        annstr = ' '.join(annstr)
        astr = f"""NPY_NO_EXPORT {annstr} {self.return_type} {self.name} \\
       ({self._argtypes_string()});"""
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
    """Check that an api dict is valid (does not use the same index twice)
    and removed `__unused_indices__` from it (which is important only here)
    """
    # Pop the `__unused_indices__` field:  These are known holes:
    removed = set(d.pop("__unused_indices__", []))
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
            f'\n\tindex {index} -> {names}'
            for index, names in doubled.items() if len(names) != 1
        )
        raise ValueError(fmt.format(val))

    # No 'hole' in the indexes may be allowed, and it must starts at 0
    indexes = set(index_d.values())
    expected = set(range(len(indexes) + len(removed)))
    if not indexes.isdisjoint(removed):
        raise ValueError("API index used but marked unused: "
                         f"{indexes.intersection(removed)}")
    if indexes.union(removed) != expected:
        diff = expected.symmetric_difference(indexes.union(removed))
        msg = f"There are some holes in the API indexing: (symmetric diff is {diff})"
        raise ValueError(msg)

def get_api_functions(tagname, api_dict):
    """Parse source files to get functions tagged by the given tag."""
    functions = []
    for f in API_FILES:
        functions.extend(find_functions(f, tagname))
    dfunctions = [(api_dict[func.name][0], func) for func in functions]
    dfunctions.sort()
    return [a[1] for a in dfunctions]

def fullapi_hash(api_dicts):
    """Given a list of api dicts defining the numpy C API, compute a checksum
    of the list of items in the API (as a string)."""
    a = []
    for d in api_dicts:
        d = d.copy()
        d.pop("__unused_indices__", None)
        for name, data in order_dict(d):
            a.extend(name)
            a.extend(','.join(map(str, data)))

    return hashlib.md5(
        ''.join(a).encode('ascii'), usedforsecurity=False
    ).hexdigest()


# To parse strings like 'hex = checksum' where hex is e.g. 0x1234567F and
# checksum a 128 bits md5 checksum (hex format as well)
VERRE = re.compile(r'(^0x[\da-f]{8})\s*=\s*([\da-f]{32})')

def get_versions_hash():
    d = []

    file = os.path.join(os.path.dirname(__file__), 'cversions.txt')
    with open(file) as fid:
        for line in fid:
            m = VERRE.match(line)
            if m:
                d.append((int(m.group(1), 16), m.group(2)))

    return dict(d)

def main():
    tagname = sys.argv[1]
    order_file = sys.argv[2]
    functions = get_api_functions(tagname, order_file)
    m = hashlib.md5(tagname, usedforsecurity=False)
    for func in functions:
        print(func)
        ah = func.api_hash()
        m.update(ah)
        print(hex(int(ah, 16)))
    print(hex(int(m.hexdigest()[:8], 16)))


if __name__ == '__main__':
    main()
