#  doc is comment_documentation

# use list so order is preserved.
ufunc_api_list = [
    (r"""
    """,
     'FromFuncAndData', 'PyUFuncGenericFunction *, void **, char *, int, int, int, int, char *, char *, int', 'PyObject *'),

    (r"""
    """,
     'RegisterLoopForType','PyUFuncObject *, int, PyUFuncGenericFunction, void *', 'int'),

    (r"""
    """,
     'GenericFunction', 'PyUFuncObject *, PyObject *, PyArrayObject **', 'int'),

    (r"""
    """,
     'f_f_As_d_d','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'd_d','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'f_f','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'g_g','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'F_F_As_D_D','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'F_F','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'D_D','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'G_G','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'O_O','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'ff_f_As_dd_d','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'ff_f','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'dd_d','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'gg_g','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'FF_F_As_DD_D','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'DD_D','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'FF_F','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'GG_G','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'OO_O','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'O_O_method','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'On_Om', 'char **, intp *, intp *, void *', 'void'),

    (r"""
    """,
     'GetPyValues', 'char *, int *, int *, PyObject **', 'int'),
    
    (r"""
    """,
     'checkfperr', 'int, PyObject *', 'int'),

    (r"""
    """,
     'clearfperr', 'void', 'void')

]

# API fixes for __arrayobject_api.h

fixed = 1
nummulti = len(ufunc_api_list)
numtotal = fixed + nummulti


module_list = []
extension_list = []
init_list = []

#setup object API
for k, item in enumerate(ufunc_api_list):
    num = fixed + k
    astr = "static %s PyUFunc_%s \\\n       (%s);" % \
           (item[3],item[1],item[2])
    module_list.append(astr)
    astr = "#define PyUFunc_%s \\\n        (*(%s (*)(%s)) \\\n"\
           "         PyUFunc_API[%d])" % (item[1],item[3],item[2],num)
    extension_list.append(astr)
    astr = "        (void *) PyUFunc_%s," % item[1]
    init_list.append(astr)


outstr = r"""
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
import_ufunc(void)
{
  PyObject *numpy = PyImport_ImportModule("scipy.base.umath");
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

#endif

""" % ('\n'.join(module_list),
       '\n'.join(extension_list))

# Write to header
fid = open('__ufunc_api.h','w')
fid.write(outstr)
fid.close()


outstr = r"""
/* These pointers will be stored in the C-object for use in other
    extension modules
*/

void *PyUFunc_API[] = {
        (void *) &PyUFunc_Type,
%s
};
""" % '\n'.join(init_list)

# Write to c-code
fid = open('__ufunc_api.c','w')
fid.write(outstr)
fid.close()
