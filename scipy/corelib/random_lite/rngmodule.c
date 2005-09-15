/*
 * Univariate random number generator. Based on (and upwards compatible
 * with) Paul Dubois' URNG module. This module provides other important
 * distributions in addition to the uniform [0., 1.) distribution.
 *
 * Written by Konrad Hinsen (based on the original URNG by Paul Dubois)
 * last revision: 1997-11-6

 * Modified 3/11/98 Source from P. Stoll to make it ANSI
 * , allow for C++, Windows. P. Dubois fix the &PyType_Type problems.
 */


#include "Python.h"
#include "Numeric/arrayobject.h"

#include "ranf.h"

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

/* Define the Python interface */

static PyObject *ErrorObject;

static PyObject *ErrorReturn(char *message)
{
  PyErr_SetString(ErrorObject,message);
  return NULL;
}

/* ---------------------------------------------------------------- */

/* Declarations for objects of type distribution */

typedef struct {
	PyObject_HEAD
	double (*density)(double x, double *param);
	void (*sample)(double *buffer, int n, double *param);
	PyArrayObject *parameters;
} distributionobject;

staticforward PyTypeObject distributiontype;


static char dist_call__doc__[] = 
"density(x) = The distribution function at x"
;

static PyObject *
dist_call(distributionobject *self, PyObject *args, PyObject *kw)
{
	double x, p;

	if (!PyArg_ParseTuple(args, "d", &x))
		return NULL;

	p = (*self->density)(x, (double *)self->parameters->data);

	return PyFloat_FromDouble(p);
}

static void
dist_sample(distributionobject *self, double *buffer, int n)
{
	(*self->sample)(buffer, n, (double *)self->parameters->data);
}


static struct PyMethodDef dist_methods[] = {
	{"density",	(PyCFunction)dist_call,	1,	dist_call__doc__},
	{NULL,		NULL}		/* sentinel */
};


static distributionobject *
newdistributionobject(void)
{
	distributionobject *self;

	self = PyObject_NEW(distributionobject, &distributiontype);
	if (self == NULL)
		return NULL;
	self->density = NULL;
	self->sample = NULL;
	self->parameters = NULL;
	return self;
}

static void
dist_dealloc(distributionobject *self)
{
	Py_XDECREF(self->parameters);
	PyMem_DEL(self);
}


static PyObject *
dist_getattr(distributionobject *self, char *name)
{
	return Py_FindMethod(dist_methods, (PyObject *)self, name);
}

static char distributiontype__doc__[] = 
"Random number distribution"
;

static PyTypeObject distributiontype = {
  PyObject_HEAD_INIT(0)
    0,				/*ob_size*/
    "random_distribution",	/*tp_name*/
    sizeof(distributionobject),	/*tp_basicsize*/
    0,				/*tp_itemsize*/
    /* methods */
    (destructor) dist_dealloc,	/*tp_dealloc*/
    (printfunc)0,		/*tp_print*/
    (getattrfunc)dist_getattr,	/*tp_getattr*/
    (setattrfunc)0,	/*tp_setattr*/
    (cmpfunc)0,		/*tp_compare*/
    (reprfunc)0,		/*tp_repr*/
    0,			/*tp_as_number*/
    0,		/*tp_as_sequence*/
    0,		/*tp_as_mapping*/
    (hashfunc)0,		/*tp_hash*/
    (ternaryfunc)dist_call,	/*tp_call*/
    (reprfunc)0,		/*tp_str*/
    
    /* Space for future expansion */
    0L,0L,0L,0L,
    distributiontype__doc__ /* Documentation string */
};

/* ---------------------------------------------------------------- */

/* Specific distributions */

static PyObject *default_distribution;

/* ------------------------------------------------------ */

/* Default: uniform in [0., 1.) */

static double
default_density(double x, double *param)
{
	return (x < 0. || x >= 1.) ? 0. : 1. ;
}

static void
default_sample(double *buffer, int n, double *param)
{
	int i;
	for(i = 0; i < n; i++)
	  buffer[i] = Ranf();
}

static PyObject *
create_default_distribution(void)
{
	distributionobject *self = newdistributionobject();

	if (self != NULL) {
	  int dims[1] = { 0 };
	  self->density = default_density;
	  self->sample = default_sample;
	  self->parameters = 
	    (PyArrayObject *)PyArray_FromDims(1, dims, PyArray_DOUBLE);
	}

	return (PyObject *)self;
}


/* ------------------------------------------------------ */

/* Uniform in [a, b) */

static double
uniform_density(double x, double *param)
{
	return (x < param[0] || x >= param[1]) ? 0. : 1./(param[1]-param[0]);
}

static void
uniform_sample(double *buffer, int n, double *param)
{
	double w = param[1]-param[0];
	int i;
	for (i = 0; i < n; i++)
		buffer[i] = param[0]+w*Ranf();
}

static PyObject *
RNG_UniformDistribution(PyObject *self, PyObject *args)
{
	distributionobject *dist;
	double a, b;

	if (!PyArg_ParseTuple(args, "dd", &a, &b))
		return NULL;
	if (a == b)
		return ErrorReturn("width of uniform distribution must be > 0");

	dist = newdistributionobject();
	if (dist != NULL) {
	  int dims[1] = { 2 };
	  double *data;
	  dist->density = uniform_density;
	  dist->sample = uniform_sample;
	  dist->parameters = 
	    (PyArrayObject *)PyArray_FromDims(1, dims, PyArray_DOUBLE);
	  data = (double *)dist->parameters->data;
	  data[0] = a < b ? a : b;
	  data[1] = a > b ? a : b;
	}

	return (PyObject *)dist;
}

static char RNG_UniformDistribution__doc__[] =
"UniformDistribution(a, b) returns a uniform distribution between a and b.\n\
";

/* ------------------------------------------------------ */

/* Normal (gaussian) with mean m and standard deviation s */

static double
normal_density(double x, double *param)
{
	double y = (x-param[0])/param[1];
	double n = 1./sqrt(2*M_PI)/param[1];
	return n*exp(-0.5*y*y);
}

static void
normal_sample(double *buffer, int n, double *param)
{
	int i;
	for (i = 0; i < n; i += 2) {
		double v1, v2, s;
		do {
			v1 = 2.*Ranf()-1.;
			v2 = 2.*Ranf()-1.;
			s = v1*v1 + v2*v2;
		} while (s >= 1. || s == 0.);
		s = param[1]*sqrt(-2.*log(s)/s);
		buffer[i] = param[0]+s*v1;
		buffer[i+1] = param[0]+s*v2;
	}
}

static PyObject *
RNG_NormalDistribution(PyObject *self, PyObject *args)
{
	distributionobject *dist;
	double m, s;

	if (!PyArg_ParseTuple(args, "dd", &m, &s))
		return NULL;
	if (s <= 0.)
		return ErrorReturn("standard deviation must be positive");

	dist = newdistributionobject();
	if (dist != NULL) {
	  int dims[1] = { 2 };
	  double *data;
	  dist->density = normal_density;
	  dist->sample = normal_sample;
	  dist->parameters = 
	    (PyArrayObject *)PyArray_FromDims(1, dims, PyArray_DOUBLE);
	  data = (double *)dist->parameters->data;
	  data[0] = m;
	  data[1] = s;
	}

	return (PyObject *)dist;
}

static char RNG_NormalDistribution__doc__[] =
"NormalDistribution(m, s) returns a normal distribution\n\
with mean m and standard deviation s.\n\
";

/* ------------------------------------------------------ */

/* Log Normal (gaussian) with mean m and standard deviation s */

static double
lognormal_density(double x, double *param)
{
	double y = (log(x)-param[2])/param[3];
	double n = 1./sqrt(2*M_PI)/param[3];
	return n*exp(-0.5*y*y)/x;
}

static void
lognormal_sample(double *buffer, int n, double *param)
{
	int i;
	for (i = 0; i < n; i += 2) {
		double v1, v2, s;
		do {
			v1 = 2.*Ranf()-1.;
			v2 = 2.*Ranf()-1.;
			s = v1*v1 + v2*v2;
		} while (s >= 1. || s == 0.);
		s = param[3]*sqrt(-2.*log(s)/s);
		buffer[i] = exp(param[2]+s*v1);
		buffer[i+1] = exp(param[2]+s*v2);
	}
}

static PyObject *
RNG_LogNormalDistribution(PyObject *self, PyObject *args)
{
	distributionobject *dist;
	double m, s, mn, sn;

	if (!PyArg_ParseTuple(args, "dd", &m, &s))
		return NULL;
	if (s <= 0.)
		return ErrorReturn("standard deviation must be positive");
	sn = log(1. + s*s/(m*m));
	mn = log(m) - 0.5*sn;
	sn = sqrt(sn);

	dist = newdistributionobject();
	if (dist != NULL) {
	  int dims[1] = { 4 };
	  double *data;
	  dist->density = lognormal_density;
	  dist->sample = lognormal_sample;
	  dist->parameters = 
	    (PyArrayObject *)PyArray_FromDims(1, dims, PyArray_DOUBLE);
	  data = (double *)dist->parameters->data;
	  data[0] = m;
	  data[1] = s;
	  data[2] = mn;
	  data[3] = sn;
	}

	return (PyObject *)dist;
}

static char RNG_LogNormalDistribution__doc__[] =
"LogNormalDistribution(m, s) returns a log normal distribution\n\
with mean m and standard deviation s.\n\
";

/* ------------------------------------------------------ */

/* Exponential distribution */

static double
expo_density(double x, double *param)
{
	return (x < 0) ? 0. : param[0]*exp(-param[0]*x);
}

static void
expo_sample(double *buffer, int n, double *param)
{
	int i;
	double r;
	for (i = 0; i < n; i++) {
		do {
			r = Ranf();
		} while (r == 0.);
		buffer[i] = -log(r)/param[0];
	}
}

static PyObject *
RNG_ExponentialDistribution(PyObject *self, PyObject *args)
{
	distributionobject *dist;
	double l;

	if (!PyArg_ParseTuple(args, "d", &l))
		return NULL;
	if (l <= 0.)
		return ErrorReturn("parameter must be positive");

	dist = newdistributionobject();
	if (dist != NULL) {
	  int dims[1] = { 1 };
	  double *data;
	  dist->density = expo_density;
	  dist->sample = expo_sample;
	  dist->parameters = 
	    (PyArrayObject *)PyArray_FromDims(1, dims, PyArray_DOUBLE);
	  data = (double *)dist->parameters->data;
	  data[0] = l;
	}

	return (PyObject *)dist;
}

static char RNG_ExponentialDistribution__doc__[] =
"ExponentialDistribution(l) returns an exponential distribution.\n\
";

/* ---------------------------------------------------------------- */

/* Declarations for objects of type random generator */

/* Make this bigger to reduce cost of making streams independent */
#define SAMPLE_SIZE 128
/* Make it smaller to save space */

typedef struct {
	PyObject_HEAD
	distributionobject *distribution;
	u32 seed[2];
	int position;
	double sample[SAMPLE_SIZE];
} rngobject;

staticforward PyTypeObject rngtype;


static char rng_ranf__doc__[] = 
"ranf() -- return next random number from this stream"
;

/* Get the next number in this stream */
static double 
rng_next(rngobject *self) 
{
  double d;

  d = self->sample[self->position++];
  if(self->position >= SAMPLE_SIZE) {
    self->position = 0;
    Setranf(self->seed);
    dist_sample(self->distribution, self->sample, SAMPLE_SIZE);
    Getranf(self->seed);
  }
  return d;
}

static PyObject *
rng_ranf(rngobject *self, PyObject *args)
{
	if (!PyArg_ParseTuple(args, ""))
		return NULL;
	return Py_BuildValue("d", rng_next(self));
}

static char rng_sample__doc__[] = 
"sample(n) = A vector of n random values"
;

static PyObject *
rng_sample(rngobject *self, PyObject *args)
{
	 /* sample(n) : Return a vector of n random numbers */
	int i;
	double *x;
	int dims[1];
	PyArrayObject *result;

	if (!PyArg_ParseTuple(args, "i", dims))
		return NULL;

	if (dims[0] <= 0) 
	  return ErrorReturn("RNG sample length cannot be <= 0.");

	result = (PyArrayObject *) PyArray_FromDims (1, dims, 'd');

	if (result == NULL) 
	  return ErrorReturn("RNG sample failed to create output array.");

	x = (double *) result->data;

	for(i=0; i < dims[0]; i ++) 
	  x[i] = rng_next(self);

	return PyArray_Return(result);
}

static struct PyMethodDef rng_methods[] = {
	{"ranf",	(PyCFunction)rng_ranf,	1,	rng_ranf__doc__},
	{"sample",	(PyCFunction)rng_sample,	1,	rng_sample__doc__},
 
	{NULL,		NULL}		/* sentinel */
};


static rngobject *
newrngobject(int seed, distributionobject *distribution)
{
	rngobject *self;

	self = PyObject_NEW(rngobject, &rngtype);
	if (self == NULL)
		return NULL;
	self->distribution = distribution;
	Py_INCREF(distribution);

	Mixranf(&seed, self->seed);
	self->position = 0;
	dist_sample(self->distribution, self->sample, SAMPLE_SIZE);
	Getranf(self->seed);


#ifdef RAN_DEBUG
	{
		int i;
/* Print first few elements of stored sample. */
for(i = 0; i < 6; i++) {
   fprintf(stderr,"sample[%i] = %f\n",i,self->sample[i]);
}
	}
#endif
	return self;
}

static void
rng_dealloc(rngobject *self)
{
	Py_DECREF(self->distribution);
	PyMem_DEL(self);
}


static PyObject *
rng_getattr(rngobject *self, char *name)
{
	return Py_FindMethod(rng_methods, (PyObject *)self, name);
}

static char rngtype__doc__[] = 
"Random number generator"
;

static PyTypeObject rngtype = {
  PyObject_HEAD_INIT(0)
    0,				/*ob_size*/
    "random_number_generator",	/*tp_name*/
    sizeof(rngobject),		/*tp_basicsize*/
    0,				/*tp_itemsize*/
    /* methods */
    (destructor) rng_dealloc,	/*tp_dealloc*/
    (printfunc)0,		/*tp_print*/
    (getattrfunc)rng_getattr,	/*tp_getattr*/
    (setattrfunc)0,	/*tp_setattr*/
    (cmpfunc)0,		/*tp_compare*/
    (reprfunc)0,		/*tp_repr*/
    0,			/*tp_as_number*/
    0,		/*tp_as_sequence*/
    0,		/*tp_as_mapping*/
    (hashfunc)0,		/*tp_hash*/
    (ternaryfunc)0,		/*tp_call*/
    (reprfunc)0,		/*tp_str*/
    
    /* Space for future expansion */
    0L,0L,0L,0L,
    rngtype__doc__ /* Documentation string */
};

/* End of code for randomgenerator objects */

/* ---------------------------------------------------------------- */


static char RNG_CreateGenerator__doc__[] =
"CreateGenerator(s, d) returns an independent random number stream generator.\n\
   s < 0  ==>  Use the default initial seed value.\n\
   s = 0  ==>  Set a random value for the seed from the system clock.\n\
   s > 0  ==>  Set seed directly (32 bits only).\n\
   d (optional): distribution object.\n\
";


static PyObject *
RNG_CreateGenerator(PyObject *self, PyObject *args)
{
	int seed;
	PyObject *distribution = default_distribution;
	PyObject *result;

	if (!PyArg_ParseTuple(args, "i|O!", &seed,
			      &distributiontype, &distribution))
		return NULL;
	result = (PyObject *)newrngobject(seed,(distributionobject *)distribution);

	return result;
}

/* List of methods defined in the module */

static struct PyMethodDef RNG_methods[] = {
	{"CreateGenerator",	(PyCFunction) RNG_CreateGenerator,	1,	RNG_CreateGenerator__doc__},
	{"UniformDistribution", (PyCFunction) RNG_UniformDistribution, 1,
	 RNG_UniformDistribution__doc__},
	{"NormalDistribution", (PyCFunction) RNG_NormalDistribution, 1,
	 RNG_NormalDistribution__doc__},
	{"LogNormalDistribution", (PyCFunction) RNG_LogNormalDistribution, 1,
	 RNG_LogNormalDistribution__doc__},
	{"ExponentialDistribution", (PyCFunction) RNG_ExponentialDistribution,
	 1, RNG_ExponentialDistribution__doc__},

	{NULL,		NULL}		/* sentinel */
};


/* Initialization function for the module (*must* be called initURNG) */

static char RNG_module_documentation[] = 
"Random number generator: independent random number streams."
;

DL_EXPORT(void)
initRNG(void)
{
	PyObject *m, *d;
    distributiontype.ob_type = &PyType_Type;
    rngtype.ob_type = &PyType_Type;

	/* Create the module and add the functions */
	m = Py_InitModule4("RNG", RNG_methods,
		RNG_module_documentation,
		(PyObject*)NULL,PYTHON_API_VERSION);

	/* Import array module */
#ifdef import_array
	import_array();
#endif

	/* Add some symbolic constants to the module */
	d = PyModule_GetDict(m);
	ErrorObject = PyErr_NewException("RNG.error", NULL, NULL);
	PyDict_SetItemString(d, "error", ErrorObject);
	default_distribution = create_default_distribution();
	PyDict_SetItemString(d, "default_distribution", default_distribution);

	/* Check for errors */
	if (PyErr_Occurred())
		Py_FatalError("can't initialize module RNG");
}

