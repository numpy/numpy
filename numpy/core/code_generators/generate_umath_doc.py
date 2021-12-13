import sys
import os
import textwrap

sys.path.insert(0, os.path.dirname(__file__))
import ufunc_docstrings as docstrings
sys.path.pop(0)

def normalize_doc(docstring):
    docstring = textwrap.dedent(docstring).strip()
    docstring = docstring.encode('unicode-escape').decode('ascii')
    docstring = docstring.replace(r'"', r'\"')
    docstring = docstring.replace(r"'", r"\'")
    # Split the docstring because some compilers (like MS) do not like big
    # string literal in C code. We split at endlines because textwrap.wrap
    # do not play well with \n
    docstring = '\\n\"\"'.join(docstring.split(r"\n"))
    return docstring

def write_code(target):
    with open(target, 'w') as fid:
        fid.write(
            "#ifndef NUMPY_CORE_INCLUDE__UMATH_DOC_GENERATED_H_\n"
            "#define NUMPY_CORE_INCLUDE__UMATH_DOC_GENERATED_H_\n"
        )
        for place, string in docstrings.docdict.items():
            cdef_name = f"DOC_{place.upper().replace('.', '_')}"
            cdef_str = normalize_doc(string)
            fid.write(f"#define {cdef_name} \"{cdef_str}\"\n")
        fid.write("#endif //NUMPY_CORE_INCLUDE__UMATH_DOC_GENERATED_H\n")
