import sys

if sys.version_info[:2] < (2, 6):
    from sets import Set as set

from genapi import API_FILES, find_functions

# Those *Api classes instances know how to output strings for the generated code
class TypeApi:
    def __init__(self, name, index, ptr_cast):
        self.index = index
        self.name = name
        self.ptr_cast = ptr_cast

    def define_from_array_api_string(self):
        return "#define %s (*(%s *)PyArray_API[%d])" % (self.name,
                                                        self.ptr_cast,
                                                        self.index)

    def array_api_define(self):
        return "        (void *) &%s" % self.name

    def internal_define(self):
        astr = """\
#ifdef NPY_ENABLE_SEPARATE_COMPILATION
    extern NPY_NO_EXPORT PyTypeObject %(type)s;
#else
    NPY_NO_EXPORT PyTypeObject %(type)s;
#endif
""" % {'type': self.name}
        return astr

class GlobalVarApi:
    def __init__(self, name, index, type):
        self.name = name
        self.index = index
        self.type = type

    def define_from_array_api_string(self):
        return "#define %s (*(%s *)PyArray_API[%d])" % (self.name,
                                                        self.type,
                                                        self.index)

    def array_api_define(self):
        return "        (%s *) &%s" % (self.type, self.name)

    def internal_define(self):
        astr = """\
#ifdef NPY_ENABLE_SEPARATE_COMPILATION
    extern NPY_NO_EXPORT %(type)s %(name)s;
#else
    NPY_NO_EXPORT %(type)s %(name)s;
#endif
""" % {'type': self.type, 'name': self.name}
        return astr

# Dummy to be able to consistently use *Api instances for all items in the
# array api
class BoolValuesApi:
    def __init__(self, name, index):
        self.name = name
        self.index = index
        self.type = 'PyBoolScalarObject'

    def define_from_array_api_string(self):
        return "#define %s ((%s *)PyArray_API[%d])" % (self.name,
                                                        self.type,
                                                        self.index)

    def array_api_define(self):
        return "        (void *) &%s" % self.name

    def internal_define(self):
        astr = """\
#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[2];
#else
NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[2];
#endif
"""
        return astr

def _repl(str):
    return str.replace('intp', 'npy_intp').replace('Bool','npy_bool')

class FunctionApi:
    def __init__(self, name, index, return_type, args):
        self.name = name
        self.index = index
        self.return_type = return_type
        self.args = args

    def _argtypes_string(self):
        if not self.args:
            return 'void'
        argstr = ', '.join([_repl(a[0]) for a in self.args])
        return argstr

    def define_from_array_api_string(self):
        define = """\
#define %s \\\n        (*(%s (*)(%s)) \\
         PyArray_API[%d])""" % (self.name,
                                self.return_type,
                                self._argtypes_string(),
                                self.index)
        return define

    def array_api_define(self):
        return "        (void *) %s" % self.name

    def internal_define(self):
        astr = """\
NPY_NO_EXPORT %s %s \\\n       (%s);""" % (self.return_type,
                                           self.name,
                                           self._argtypes_string())
        return astr

def order_dict(d):
    """Order dict by its values."""
    o = d.items()
    def cmp(x, y):
        return x[1] - y[1]
    return sorted(o, cmp=cmp)

def merge_api_dicts(dicts):
    ret = {}
    for d in dicts:
        for k, v in d.items():
            ret[k] = v

    return ret

def check_api_dict(d):
    """Check that an api dict is valid (does not use the same index twice)."""
    # We have if a same index is used twice: we 'revert' the dict so that index
    # become keys. If the length is different, it means one index has been used
    # at least twice
    revert_dict = dict([(v, k) for k, v in d.items()])
    if not len(revert_dict) == len(d):
        # We compute a dict index -> list of associated items
        doubled = {}
        for name, index in d.items():
            try:
                doubled[index].append(name)
            except KeyError:
                doubled[index] = [name]
        msg = """\
Same index has been used twice in api definition: %s
""" % ['index %d -> %s' % (index, names) for index, names in doubled.items() \
                                          if len(names) != 1]
        raise ValueError(msg)

    # No 'hole' in the indexes may be allowed, and it must starts at 0
    indexes = set(d.values())
    expected = set(range(len(indexes)))
    if not indexes == expected:
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
        o = api_dict[func.name]
        dfunctions.append( (o, func) )
    dfunctions.sort()
    return [a[1] for a in dfunctions]
