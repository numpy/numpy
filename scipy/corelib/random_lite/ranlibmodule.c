#include "Python.h"
#include "scipy/arrayobject.h"
#include "ranlib.h"
#include "stdio.h"

/* ----------------------------------------------------- */

static PyObject*
get_continuous_random(int num_dist_params, PyObject* self, PyObject* args, void* fun) {
  PyArrayObject *op;
  double *out_ptr;
  int i, n=-1;
  float a, b, c;

  switch(num_dist_params) {
  case 0:
    if( !PyArg_ParseTuple(args, "|i", &n) )
      return NULL;
    break;
  case 1:
    if( !PyArg_ParseTuple(args, "f|i", &a, &n) )
      return NULL;
    break;
  case 2:
    if( !PyArg_ParseTuple(args, "ff|i", &a, &b, &n) )
      return NULL;
    break;
  case 3:
    if( !PyArg_ParseTuple(args, "fff|i", &a, &b, &c, &n) )
      return NULL;
    break;
  }
  if( n == -1 )
    n = 1;

  /* Create a 1 dimensional array of length n of type double */
  op = (PyArrayObject*) PyArray_FromDims(1, &n, PyArray_DOUBLE);
  if( op == NULL )
    return NULL;


  out_ptr = (double *) op->data;
  for(i=0; i<n; i++) {
    switch(num_dist_params) {
    case 0:
      *out_ptr = (double) ((float (*)(void)) fun)();
      break;
    case 1:
      *out_ptr = (double) ((float (*)(float)) fun)(a);
      break;
    case 2:
      *out_ptr = (double) ((float (*)(float, float)) fun)(a,b);
      break;
    case 3:
      *out_ptr = (double) ((float (*)(float, float, float)) fun)(a,b,c);
      break;
    }
    out_ptr++;
  }

  return PyArray_Return(op);
}


static PyObject*
get_discrete_scalar_random(int num_integer_args, PyObject* self, PyObject* args, void* fun) {
  long int_arg;
  int n=-1, i;
  long* out_ptr;
  PyArrayObject* op;
  float float_arg;

  switch( num_integer_args ) {
  case 0:
    if( !PyArg_ParseTuple(args, "f|i", &float_arg, &n) ) {
      return NULL;
    }
    break;
  case 1:
    if( !PyArg_ParseTuple(args, "lf|i", &int_arg, &float_arg, &n) ) {
      return NULL;
    }
    break;
  }
  if( n==-1 ) {
    n = 1;
  }

  /* Create a 1 dimensional array of length n of type long */
  op = (PyArrayObject*) PyArray_FromDims(1, &n, PyArray_LONG);
  if( op == NULL ) {
    return NULL;
  }

  out_ptr = (long*) op->data;
  for(i=0; i<n; i++) {
    switch( num_integer_args ) {
    case 0:
      *out_ptr = ((long (*)(float)) fun)(float_arg);
      break;
    case 1:
      *out_ptr = ((long (*)(long, float)) fun)(int_arg, float_arg);
      break;
    }
    out_ptr++;
  }

  return PyArray_Return(op);
}


static char random_sample__doc__[] ="";

static PyObject *
random_sample(PyObject *self, PyObject *args) {
  PyArrayObject *op;
  double *out_ptr;
  int i, n=1;

  if (!PyArg_ParseTuple(args, "|i", &n)) {
      return NULL;
  }

  /* Create a 1 dimensional array of length n of type double */
  op = (PyArrayObject*) PyArray_FromDims(1, &n, PyArray_DOUBLE);
  if (op == NULL) {
    return NULL;
  }


  out_ptr = (double *) op->data;
  for (i=0; i<n; i++) {
      *out_ptr = ranf();
      out_ptr++;
  }

  return PyArray_Return(op);
}


static char standard_normal__doc__[] ="";

static PyObject *
standard_normal(PyObject *self, PyObject *args) {
  return get_continuous_random(0, self, args, snorm);
}


static char beta__doc__[] ="";

static PyObject *
beta(PyObject *self, PyObject *args) {
  return get_continuous_random(2, self, args, genbet);
}


static char gamma__doc__[] ="";

static PyObject *
/* there is a function named `gamma' in some libm's */
_gamma(PyObject *self, PyObject *args) {
  return get_continuous_random(2, self, args, gengam);
}


static char f__doc__[] ="";

static PyObject *
f(PyObject *self, PyObject *args) {
  return get_continuous_random(2, self, args, genf);
}


static char noncentral_f__doc__[] ="";

static PyObject *
noncentral_f(PyObject *self, PyObject *args) {
  return get_continuous_random(3, self, args, gennf);
}


static char noncentral_chisquare__doc__[] ="";

static PyObject *
noncentral_chisquare(PyObject *self, PyObject *args) {
  return get_continuous_random(2, self, args, gennch);
}


static char chisquare__doc__[] ="";

static PyObject *
chisquare(PyObject *self, PyObject *args) {
  return get_continuous_random(1, self, args, genchi);
}


static char binomial__doc__[] ="";

static PyObject *
binomial(PyObject *self, PyObject *args) {
  return get_discrete_scalar_random(1, self, args, ignbin);
}


static char negative_binomial__doc__[] ="";

static PyObject *
negative_binomial(PyObject *self, PyObject *args) {
  return get_discrete_scalar_random(1, self, args, ignnbn);
}

static char poisson__doc__[] ="";

static PyObject *
poisson(PyObject *self, PyObject *args) {
  return get_discrete_scalar_random(0, self, args, ignpoi);
}


static char multinomial__doc__[] ="";

static PyObject*
multinomial(PyObject* self, PyObject* args) {
  int n=-1, i;
  long num_trials, num_categories;
  char* out_ptr;
  PyArrayObject* priors_array;
  PyObject* priors_object;
  PyArrayObject* op;
  int out_dimensions[2];

  if( !PyArg_ParseTuple(args, "lO|i", &num_trials, &priors_object, &n) ) {
    return NULL;
  }
  priors_array = (PyArrayObject*) PyArray_ContiguousFromObject(priors_object, PyArray_FLOAT, 1, 1);
  if( priors_array == NULL ) {
    return NULL;
  }
  num_categories = priors_array->dimensions[0]+1;
  if( n==-1 ) {
    n = 1;
  }

  /* Create an n by num_categories array of long */
  out_dimensions[0] = n;
  out_dimensions[1] = num_categories;
  op = (PyArrayObject*) PyArray_FromDims(2, out_dimensions, PyArray_LONG);
  if( op == NULL ) {
    return NULL;
  }

  out_ptr = op->data;
  for(i=0; i<n; i++) {
    genmul(num_trials, (float*)(priors_array->data), num_categories, (long*) out_ptr);
    out_ptr += op->strides[0];
  }

  return PyArray_Return(op);
}


static PyObject *
random_set_seeds(PyObject *self, PyObject *args)
{
  long seed1, seed2;

  if (!PyArg_ParseTuple(args, "ll", &seed1, &seed2)) return NULL;


  setall(seed1, seed2);
  if (PyErr_Occurred ()) return NULL;
  Py_INCREF(Py_None);
  return (PyObject *)Py_None;
}


static PyObject *
random_get_seeds(PyObject *self, PyObject *args)
{
  long seed1, seed2;

  if (!PyArg_ParseTuple(args, "")) return NULL;

  getsd(&seed1, &seed2);

  return Py_BuildValue("ll", seed1, seed2);
}


/* Missing interfaces to */
/* exponential (genexp), multivariate normal (genmn),
   normal (gennor), permutation (genprm), uniform (genunf),
   standard exponential (sexpo), standard gamma (sgamma) */

/* List of methods defined in the module */

static struct PyMethodDef random_methods[] = {
 {"sample",     random_sample,          1,      random_sample__doc__},
 {"standard_normal", standard_normal,   1,      standard_normal__doc__},
 {"beta",	beta,                   1,      beta__doc__},
 {"gamma",	_gamma,                  1,      gamma__doc__},
 {"f",	        f,                      1,      f__doc__},
 {"noncentral_f", noncentral_f,         1,      noncentral_f__doc__},
 {"chisquare",	chisquare,              1,      chisquare__doc__},
 {"noncentral_chisquare", noncentral_chisquare,
                                        1,      noncentral_chisquare__doc__},
 {"binomial",	binomial,               1,      binomial__doc__},
 {"negative_binomial", negative_binomial,
                                        1,      negative_binomial__doc__},
 {"multinomial", multinomial,           1,      multinomial__doc__},
 {"poisson",    poisson,                1,      poisson__doc__},
 {"set_seeds",  random_set_seeds,       1, },
 {"get_seeds",  random_get_seeds,       1, },
 {NULL,		NULL}		/* sentinel */
};


/* Initialization function for the module (*must* be called initranlib) */

static char random_module_documentation[] =
""
;

DL_EXPORT(void)
initranlib(void)
{
	PyObject *m;

	/* Create the module and add the functions */
	m = Py_InitModule4("ranlib", random_methods,
		random_module_documentation,
		(PyObject*)NULL,PYTHON_API_VERSION);

	/* Import the array object */
	import_array();

	/* XXXX Add constants here */

	/* Check for errors */
	if (PyErr_Occurred())
		Py_FatalError("can't initialize module ranlib");
}
