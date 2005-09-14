#ifdef __cplusplus
extern "C" {
#endif

#include "Python.h"

#ifdef HAVE_X11
#include "X11/X.h"
#include "X11/Xlib.h"
#endif

static
int have_x11(void) {
#ifdef HAVE_X11
  return 1;
#else
  return 0;
#endif  
}

static
int try_XOpenDisplay(const char* display_name) {
#ifdef HAVE_X11
  Display* display = XOpenDisplay(display_name);
  if (display==NULL) {
    return 0;
  } else {
    XCloseDisplay(display);
    return 1;
  }
#else
  return 0;
#endif
}

static char have_x11_doc[] = "have_x11() -> bool";

static PyObject *have_x11_wrap(PyObject *self, PyObject *args) {
  int result = 0;

  if (!PyArg_ParseTuple(args,":display_test.try_XOpenDisplay"))
    return NULL;
  result = have_x11();
  return Py_BuildValue("i",result);
}

static char try_XOpenDisplay_doc[] = "try_XOpenDisplay([display_name]) -> bool";

static PyObject *try_XOpenDisplay_wrap(PyObject *self,
				       PyObject *args,
				       PyObject *keywds) {
  int result = 0;
  char* display_name = NULL;
  static char *kwlist[] = {"display_name",NULL};

  if (!PyArg_ParseTupleAndKeywords(args,keywds,\
    "|s:display_test.try_XOpenDisplay",kwlist,&display_name))
    return NULL;
  result = try_XOpenDisplay(display_name);
  return Py_BuildValue("i",result);
}

static PyMethodDef module_methods[] = {
  {"have_x11",have_x11_wrap,METH_VARARGS,have_x11_doc},
  {"try_XOpenDisplay",try_XOpenDisplay_wrap,METH_VARARGS | METH_KEYWORDS,
   try_XOpenDisplay_doc},
  {NULL,NULL}
};

DL_EXPORT(void) initdisplay_test(void) {
  Py_InitModule("display_test", module_methods);
  if (PyErr_Occurred())
    Py_FatalError("can't initialize module display_test");
}

#ifdef __cplusplus
}
#endif
