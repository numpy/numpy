import os
import genapi

UFUNC_API_ORDER = 'ufunc_api_order.txt'

h_template = r"""
#ifdef _UMATHMODULE

static PyTypeObject PyUFunc_Type;

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

#define import_umath() { if (_import_umath() < 0) {PyErr_Print(); Py_FatalError("numpy.core.umath failed to import... exiting.\n"); }}

#define import_ufunc import_umath

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
    header_file = os.path.join(output_dir, '__ufunc_api.h')
    c_file = os.path.join(output_dir, '__ufunc_api.c')
    doc_file = os.path.join(output_dir, 'ufunc_api.txt')

    targets = (header_file, c_file, doc_file)
    if (not force
            and not genapi.should_rebuild(targets,
                                          [UFUNC_API_ORDER, __file__])):
        return targets

    ufunc_api_list = genapi.get_api_functions('UFUNC_API', UFUNC_API_ORDER)

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
