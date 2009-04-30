import os
import genapi

h_template = r"""
#ifdef _UMATHMODULE

#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT PyTypeObject PyUFunc_Type;
#else
NPY_NO_EXPORT PyTypeObject PyUFunc_Type;
#endif

%s

#else

#if defined(PY_UFUNC_UNIQUE_SYMBOL)
#define PyUFunc_API PY_UFUNC_UNIQUE_SYMBOL
#endif

#if defined(NO_IMPORT) || defined(NO_IMPORT_UFUNC)
extern void **PyUFunc_API;
#else
#if defined(PY_UFUNC_UNIQUE_SYMBOL)
void **PyUFunc_API;
#else
static void **PyUFunc_API=NULL;
#endif
#endif

#define PyUFunc_Type (*(PyTypeObject *)PyUFunc_API[0])

%s

static int
_import_umath(void)
{
  PyObject *numpy = PyImport_ImportModule("numpy.core.umath");
  PyObject *c_api = NULL;

  if (numpy == NULL) return -1;
  c_api = PyObject_GetAttrString(numpy, "_UFUNC_API");
  if (c_api == NULL) {Py_DECREF(numpy); return -1;}
  if (PyCObject_Check(c_api)) {
      PyUFunc_API = (void **)PyCObject_AsVoidPtr(c_api);
  }
  Py_DECREF(c_api);
  Py_DECREF(numpy);
  if (PyUFunc_API == NULL) return -1;
  return 0;
}

#define import_umath() { UFUNC_NOFPE if (_import_umath() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.umath failed to import"); return; }}

#define import_umath1(ret) { UFUNC_NOFPE if (_import_umath() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.umath failed to import"); return ret; }}

#define import_umath2(msg, ret) { UFUNC_NOFPE if (_import_umath() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, msg); return ret; }}

#define import_ufunc() { UFUNC_NOFPE if (_import_umath() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.umath failed to import"); }}


#endif
"""

c_template = r"""
/* These pointers will be stored in the C-object for use in other
    extension modules
*/

void *PyUFunc_API[] = {
        (void *) &PyUFunc_Type,
%s
};
"""

def generate_api(output_dir, force=False):
    basename = 'ufunc_api'

    h_file = os.path.join(output_dir, '__%s.h' % basename)
    c_file = os.path.join(output_dir, '__%s.c' % basename)
    d_file = os.path.join(output_dir, '%s.txt' % basename)
    targets = (h_file, c_file, d_file)

    sources = ['ufunc_api_order.txt']

    if (not force and not genapi.should_rebuild(targets, sources + [__file__])):
        return targets
    else:
        do_generate_api(targets, sources)

    return targets

def do_generate_api(targets, sources):
    header_file = targets[0]
    c_file = targets[1]
    doc_file = targets[2]

    ufunc_api_list = genapi.get_api_functions('UFUNC_API', sources[0])

    # API fixes for __arrayobject_api.h

    fixed = 1
    nummulti = len(ufunc_api_list)
    numtotal = fixed + nummulti

    module_list = []
    extension_list = []
    init_list = []

    # set up object API
    genapi.add_api_list(fixed, 'PyUFunc_API', ufunc_api_list,
                        module_list, extension_list, init_list)

    # Write to header
    fid = open(header_file, 'w')
    s = h_template % ('\n'.join(module_list), '\n'.join(extension_list))
    fid.write(s)
    fid.close()

    # Write to c-code
    fid = open(c_file, 'w')
    s = c_template % '\n'.join(init_list)
    fid.write(s)
    fid.close()

    # Write to documentation
    fid = open(doc_file, 'w')
    fid.write('''
=================
Numpy Ufunc C-API
=================
''')
    for func in ufunc_api_list:
        fid.write(func.to_ReST())
        fid.write('\n\n')
    fid.close()

    return targets
