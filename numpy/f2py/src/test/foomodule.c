/* File: foomodule.c
 * Example of FortranObject usage. See also wrap.f foo.f foo90.f90.
 * Author: Pearu Peterson <pearu@ioc.ee>.
 * http://cens.ioc.ee/projects/f2py2e/
 * $Revision: 1.2 $
 * $Date: 2000/09/17 16:10:27 $
 */
#ifdef __cplusplus
extern "C" {
#endif

#include "Python.h"
#include "fortranobject.h"

static PyObject *foo_error;

#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F
#else
#define F_FUNC(f,F) f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F##_
#else
#define F_FUNC(f,F) f##_
#endif
#endif

/************* foo_bar *************/
static char doc_foo_bar[] = "\
Function signature:\n\
  bar()\n\
";
static PyObject *foo_bar(PyObject *capi_self, PyObject *capi_args,
                         PyObject *capi_keywds, void (*f2py_func)()) {
    PyObject *capi_buildvalue = NULL;
    static char *capi_kwlist[] = {NULL};
    if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
                                     "|:foo.bar",\
                                     capi_kwlist))
        goto capi_fail;
    (*f2py_func)();
    capi_buildvalue = Py_BuildValue("");
 capi_fail:
    return capi_buildvalue;
}
/************ mod_init **************/
static PyObject *mod_init(PyObject *capi_self, PyObject *capi_args,
                          PyObject *capi_keywds, void (*f2py_func)()) {
    PyObject *capi_buildvalue = NULL;
    static char *capi_kwlist[] = {NULL};
    if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
                                     "|:mod.init",\
                                     capi_kwlist))
        goto capi_fail;
    (*f2py_func)();
    capi_buildvalue = Py_BuildValue("");
 capi_fail:
    return capi_buildvalue;
}

/* F90 module */
static FortranDataDef f2py_mod_def[] = {
    {"a",0, {}, NPY_INT},
    {"b",0, {}, NPY_DOUBLE},
    {"c",1, {3}, NPY_DOUBLE},
    {"d",1, {-1}, NPY_DOUBLE},
    {"init",-1,{},0,NULL,(void *)mod_init},
    {NULL}
};
static void f2py_setup_mod(char *a,char *b,char *c,void (*d)(),char *init) {
    f2py_mod_def[0].data = a;
    f2py_mod_def[1].data = b;
    f2py_mod_def[2].data = c;
    f2py_mod_def[3].func = d;
    f2py_mod_def[4].data = init;
}
extern void F_FUNC(f2pyinitmod,F2PYINITMOD)();
                                           static void f2py_init_mod() {
                                               F_FUNC(f2pyinitmod,F2PYINITMOD)(f2py_setup_mod);
                                           }

/* COMMON block */
static FortranDataDef f2py_foodata_def[] = {
    {"a",0, {}, NPY_INT},
    {"b",0, {}, NPY_DOUBLE},
    {"c",1, {3}, NPY_DOUBLE},
    {NULL}
};
static void f2py_setup_foodata(char *a,char *b,char *c) {
    f2py_foodata_def[0].data = a;
    f2py_foodata_def[1].data = b;
    f2py_foodata_def[2].data = c;
}
extern void F_FUNC(f2pyinitfoodata,F2PYINITFOODATA)();
                                                   static void f2py_init_foodata() {
                                                       F_FUNC(f2pyinitfoodata,F2PYINITFOODATA)(f2py_setup_foodata);
                                                   }

/* Fortran routines (needs no initialization/setup function) */
extern void F_FUNC(bar,BAR)();
                           extern void F_FUNC(foo,FOO)();
                                                      static FortranDataDef f2py_routines_def[] = {
                                                          {"bar",-1, {}, 0, (char *)F_FUNC(bar,BAR),(void *)foo_bar,doc_foo_bar},
                                                          {"foo",-1, {}, 0, (char *)F_FUNC(foo,FOO),(void *)foo_bar,doc_foo_bar},
                                                          {NULL}
                                                      };

static PyMethodDef foo_module_methods[] = {
    /*eof method*/
    {NULL,NULL}
};

void initfoo() {
    int i;
    PyObject *m, *d, *s, *tmp;
    import_array();

    m = Py_InitModule("foo", foo_module_methods);

    d = PyModule_GetDict(m);
    s = PyString_FromString("This module 'foo' demonstrates the usage of fortranobject.");
    PyDict_SetItemString(d, "__doc__", s);

    /* Fortran objects: */
    tmp = PyFortranObject_New(f2py_mod_def,f2py_init_mod);
    PyDict_SetItemString(d, "mod", tmp);
    Py_DECREF(tmp);
    tmp = PyFortranObject_New(f2py_foodata_def,f2py_init_foodata);
    PyDict_SetItemString(d, "foodata", tmp);
    Py_DECREF(tmp);
    for(i=0;f2py_routines_def[i].name!=NULL;i++) {
        tmp = PyFortranObject_NewAsAttr(&f2py_routines_def[i]);
        PyDict_SetItemString(d, f2py_routines_def[i].name, tmp);
        Py_DECREF(tmp);
    }

    Py_DECREF(s);

    if (PyErr_Occurred())
        Py_FatalError("can't initialize module foo");
}

#ifdef __cplusplus
}
#endif
