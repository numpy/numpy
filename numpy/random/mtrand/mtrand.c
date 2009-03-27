/* 0.9.7.2 on Sat Mar 28 01:29:04 2009 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"
#ifndef PY_LONG_LONG
  #define PY_LONG_LONG LONG_LONG
#endif
#if PY_VERSION_HEX < 0x02050000
  typedef int Py_ssize_t;
  #define PY_SSIZE_T_MAX INT_MAX
  #define PY_SSIZE_T_MIN INT_MIN
  #define PyInt_FromSsize_t(z) PyInt_FromLong(z)
  #define PyInt_AsSsize_t(o)	PyInt_AsLong(o)
#endif
#ifndef WIN32
  #ifndef __stdcall
    #define __stdcall
  #endif
  #ifndef __cdecl
    #define __cdecl
  #endif
#endif
#ifdef __cplusplus
#define __PYX_EXTERN_C extern "C"
#else
#define __PYX_EXTERN_C extern
#endif
#include <math.h>
#include "string.h"
#include "math.h"
#include "numpy/arrayobject.h"
#include "randomkit.h"
#include "distributions.h"
#include "initarray.h"


typedef struct {PyObject **p; char *s;} __Pyx_InternTabEntry; /*proto*/
typedef struct {PyObject **p; char *s; long n;} __Pyx_StringTabEntry; /*proto*/

static PyObject *__pyx_m;
static PyObject *__pyx_b;
static int __pyx_lineno;
static char *__pyx_filename;
static char **__pyx_f;

static int __Pyx_GetStarArgs(PyObject **args, PyObject **kwds, char *kwd_list[],     Py_ssize_t nargs, PyObject **args2, PyObject **kwds2, char rqd_kwds[]); /*proto*/

static PyObject *__Pyx_Import(PyObject *name, PyObject *from_list); /*proto*/

static PyObject *__Pyx_GetName(PyObject *dict, PyObject *name); /*proto*/

static int __Pyx_SetItemInt(PyObject *o, Py_ssize_t i, PyObject *v); /*proto*/

static PyObject *__Pyx_GetItemInt(PyObject *o, Py_ssize_t i); /*proto*/

static void __Pyx_Raise(PyObject *type, PyObject *value, PyObject *tb); /*proto*/

static PyObject *__Pyx_UnpackItem(PyObject *); /*proto*/
static int __Pyx_EndUnpack(PyObject *); /*proto*/

static int __Pyx_GetException(PyObject **type, PyObject **value, PyObject **tb); /*proto*/

static int __Pyx_TypeTest(PyObject *obj, PyTypeObject *type); /*proto*/

static int __Pyx_InternStrings(__Pyx_InternTabEntry *t); /*proto*/

static int __Pyx_InitStrings(__Pyx_StringTabEntry *t); /*proto*/

static PyTypeObject *__Pyx_ImportType(char *module_name, char *class_name, long size);  /*proto*/

static PyObject *__Pyx_ImportModule(char *name); /*proto*/

static void __Pyx_AddTraceback(char *funcname); /*proto*/

/* Declarations from mtrand */

typedef double (*__pyx_t_6mtrand_rk_cont0)(rk_state *);

typedef double (*__pyx_t_6mtrand_rk_cont1)(rk_state *,double);

typedef double (*__pyx_t_6mtrand_rk_cont2)(rk_state *,double,double);

typedef double (*__pyx_t_6mtrand_rk_cont3)(rk_state *,double,double,double);

typedef long (*__pyx_t_6mtrand_rk_disc0)(rk_state *);

typedef long (*__pyx_t_6mtrand_rk_discnp)(rk_state *,long,double);

typedef long (*__pyx_t_6mtrand_rk_discdd)(rk_state *,double,double);

typedef long (*__pyx_t_6mtrand_rk_discnmN)(rk_state *,long,long,long);

typedef long (*__pyx_t_6mtrand_rk_discd)(rk_state *,double);

struct __pyx_obj_6mtrand_RandomState {
  PyObject_HEAD
  rk_state *internal_state;
};


static PyTypeObject *__pyx_ptype_6mtrand_dtype = 0;
static PyTypeObject *__pyx_ptype_6mtrand_ndarray = 0;
static PyTypeObject *__pyx_ptype_6mtrand_flatiter = 0;
static PyTypeObject *__pyx_ptype_6mtrand_broadcast = 0;
static PyTypeObject *__pyx_ptype_6mtrand_RandomState = 0;
static PyObject *__pyx_k2;
static PyObject *__pyx_k3;
static PyObject *__pyx_k4;
static PyObject *__pyx_k5;
static PyObject *__pyx_k6;
static PyObject *__pyx_k7;
static PyObject *__pyx_k8;
static PyObject *__pyx_k9;
static PyObject *__pyx_k10;
static PyObject *__pyx_k11;
static PyObject *__pyx_k12;
static PyObject *__pyx_k13;
static PyObject *__pyx_k14;
static PyObject *__pyx_k15;
static PyObject *__pyx_k16;
static PyObject *__pyx_k17;
static PyObject *__pyx_k18;
static PyObject *__pyx_k19;
static PyObject *__pyx_k20;
static PyObject *__pyx_k21;
static PyObject *__pyx_k22;
static PyObject *__pyx_k23;
static PyObject *__pyx_k24;
static PyObject *__pyx_k25;
static PyObject *__pyx_k26;
static PyObject *__pyx_k27;
static PyObject *__pyx_k28;
static PyObject *__pyx_k29;
static PyObject *__pyx_k30;
static PyObject *__pyx_k31;
static PyObject *__pyx_k32;
static PyObject *__pyx_k33;
static PyObject *__pyx_k34;
static PyObject *__pyx_k35;
static PyObject *__pyx_k36;
static PyObject *__pyx_k37;
static PyObject *__pyx_k38;
static PyObject *__pyx_k39;
static PyObject *__pyx_k40;
static PyObject *__pyx_k41;
static PyObject *__pyx_k42;
static PyObject *__pyx_k43;
static PyObject *__pyx_k44;
static PyObject *__pyx_k45;
static PyObject *__pyx_k46;
static PyObject *__pyx_k47;
static PyObject *__pyx_k48;
static PyObject *__pyx_k49;
static PyObject *__pyx_k50;
static PyObject *__pyx_k51;
static PyObject *__pyx_k52;
static PyObject *__pyx_k53;
static PyObject *__pyx_k54;
static PyObject *__pyx_k55;
static PyObject *__pyx_k56;
static PyObject *__pyx_k57;
static PyObject *__pyx_k58;
static PyObject *__pyx_k59;
static PyObject *__pyx_k60;
static PyObject *__pyx_f_6mtrand_cont0_array(rk_state *,__pyx_t_6mtrand_rk_cont0,PyObject *); /*proto*/
static PyObject *__pyx_f_6mtrand_cont1_array_sc(rk_state *,__pyx_t_6mtrand_rk_cont1,PyObject *,double); /*proto*/
static PyObject *__pyx_f_6mtrand_cont1_array(rk_state *,__pyx_t_6mtrand_rk_cont1,PyObject *,PyArrayObject *); /*proto*/
static PyObject *__pyx_f_6mtrand_cont2_array_sc(rk_state *,__pyx_t_6mtrand_rk_cont2,PyObject *,double,double); /*proto*/
static PyObject *__pyx_f_6mtrand_cont2_array(rk_state *,__pyx_t_6mtrand_rk_cont2,PyObject *,PyArrayObject *,PyArrayObject *); /*proto*/
static PyObject *__pyx_f_6mtrand_cont3_array_sc(rk_state *,__pyx_t_6mtrand_rk_cont3,PyObject *,double,double,double); /*proto*/
static PyObject *__pyx_f_6mtrand_cont3_array(rk_state *,__pyx_t_6mtrand_rk_cont3,PyObject *,PyArrayObject *,PyArrayObject *,PyArrayObject *); /*proto*/
static PyObject *__pyx_f_6mtrand_disc0_array(rk_state *,__pyx_t_6mtrand_rk_disc0,PyObject *); /*proto*/
static PyObject *__pyx_f_6mtrand_discnp_array_sc(rk_state *,__pyx_t_6mtrand_rk_discnp,PyObject *,long,double); /*proto*/
static PyObject *__pyx_f_6mtrand_discnp_array(rk_state *,__pyx_t_6mtrand_rk_discnp,PyObject *,PyArrayObject *,PyArrayObject *); /*proto*/
static PyObject *__pyx_f_6mtrand_discdd_array_sc(rk_state *,__pyx_t_6mtrand_rk_discdd,PyObject *,double,double); /*proto*/
static PyObject *__pyx_f_6mtrand_discdd_array(rk_state *,__pyx_t_6mtrand_rk_discdd,PyObject *,PyArrayObject *,PyArrayObject *); /*proto*/
static PyObject *__pyx_f_6mtrand_discnmN_array_sc(rk_state *,__pyx_t_6mtrand_rk_discnmN,PyObject *,long,long,long); /*proto*/
static PyObject *__pyx_f_6mtrand_discnmN_array(rk_state *,__pyx_t_6mtrand_rk_discnmN,PyObject *,PyArrayObject *,PyArrayObject *,PyArrayObject *); /*proto*/
static PyObject *__pyx_f_6mtrand_discd_array_sc(rk_state *,__pyx_t_6mtrand_rk_discd,PyObject *,double); /*proto*/
static PyObject *__pyx_f_6mtrand_discd_array(rk_state *,__pyx_t_6mtrand_rk_discd,PyObject *,PyArrayObject *); /*proto*/
static double __pyx_f_6mtrand_kahan_sum(double *,long); /*proto*/


/* Implementation of mtrand */


static PyObject *__pyx_n_numpy;
static PyObject *__pyx_n_np;
static PyObject *__pyx_n__rand;
static PyObject *__pyx_n_seed;
static PyObject *__pyx_n_get_state;
static PyObject *__pyx_n_set_state;
static PyObject *__pyx_n_random_sample;
static PyObject *__pyx_n_randint;
static PyObject *__pyx_n_bytes;
static PyObject *__pyx_n_uniform;
static PyObject *__pyx_n_rand;
static PyObject *__pyx_n_randn;
static PyObject *__pyx_n_random_integers;
static PyObject *__pyx_n_standard_normal;
static PyObject *__pyx_n_normal;
static PyObject *__pyx_n_beta;
static PyObject *__pyx_n_exponential;
static PyObject *__pyx_n_standard_exponential;
static PyObject *__pyx_n_standard_gamma;
static PyObject *__pyx_n_gamma;
static PyObject *__pyx_n_f;
static PyObject *__pyx_n_noncentral_f;
static PyObject *__pyx_n_chisquare;
static PyObject *__pyx_n_noncentral_chisquare;
static PyObject *__pyx_n_standard_cauchy;
static PyObject *__pyx_n_standard_t;
static PyObject *__pyx_n_vonmises;
static PyObject *__pyx_n_pareto;
static PyObject *__pyx_n_weibull;
static PyObject *__pyx_n_power;
static PyObject *__pyx_n_laplace;
static PyObject *__pyx_n_gumbel;
static PyObject *__pyx_n_logistic;
static PyObject *__pyx_n_lognormal;
static PyObject *__pyx_n_rayleigh;
static PyObject *__pyx_n_wald;
static PyObject *__pyx_n_triangular;
static PyObject *__pyx_n_binomial;
static PyObject *__pyx_n_negative_binomial;
static PyObject *__pyx_n_poisson;
static PyObject *__pyx_n_zipf;
static PyObject *__pyx_n_geometric;
static PyObject *__pyx_n_hypergeometric;
static PyObject *__pyx_n_logseries;
static PyObject *__pyx_n_multivariate_normal;
static PyObject *__pyx_n_multinomial;
static PyObject *__pyx_n_dirichlet;
static PyObject *__pyx_n_shuffle;
static PyObject *__pyx_n_permutation;

static PyObject *__pyx_n_empty;
static PyObject *__pyx_n_float64;

static PyObject *__pyx_f_6mtrand_cont0_array(rk_state *__pyx_v_state,__pyx_t_6mtrand_rk_cont0 __pyx_v_func,PyObject *__pyx_v_size) {
  double *__pyx_v_array_data;
  PyArrayObject *arrayObject;
  long __pyx_v_length;
  long __pyx_v_i;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  Py_INCREF(__pyx_v_size);
  arrayObject = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":131 */
  __pyx_1 = __pyx_v_size == Py_None;
  if (__pyx_1) {
    __pyx_2 = PyFloat_FromDouble(__pyx_v_func(__pyx_v_state)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 132; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":134 */
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 134; goto __pyx_L1;}
    __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_empty); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 134; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 134; goto __pyx_L1;}
    __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_float64); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 134; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 134; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 1, __pyx_4);
    __pyx_4 = 0;
    __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 134; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_4)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_4);
    Py_DECREF(__pyx_4); __pyx_4 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":135 */
    __pyx_v_length = PyArray_SIZE(arrayObject);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":136 */
    __pyx_v_array_data = ((double *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":137 */
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_v_length; ++__pyx_v_i) {
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state);
    }

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":139 */
    Py_INCREF(((PyObject *)arrayObject));
    __pyx_r = ((PyObject *)arrayObject);
    goto __pyx_L0;
  }
  __pyx_L2:;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("mtrand.cont0_array");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_cont1_array_sc(rk_state *__pyx_v_state,__pyx_t_6mtrand_rk_cont1 __pyx_v_func,PyObject *__pyx_v_size,double __pyx_v_a) {
  double *__pyx_v_array_data;
  PyArrayObject *arrayObject;
  long __pyx_v_length;
  long __pyx_v_i;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  Py_INCREF(__pyx_v_size);
  arrayObject = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":148 */
  __pyx_1 = __pyx_v_size == Py_None;
  if (__pyx_1) {
    __pyx_2 = PyFloat_FromDouble(__pyx_v_func(__pyx_v_state,__pyx_v_a)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 149; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":151 */
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 151; goto __pyx_L1;}
    __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_empty); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 151; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 151; goto __pyx_L1;}
    __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_float64); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 151; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 151; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 1, __pyx_4);
    __pyx_4 = 0;
    __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 151; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_4)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_4);
    Py_DECREF(__pyx_4); __pyx_4 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":152 */
    __pyx_v_length = PyArray_SIZE(arrayObject);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":153 */
    __pyx_v_array_data = ((double *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":154 */
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_v_length; ++__pyx_v_i) {
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,__pyx_v_a);
    }

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":156 */
    Py_INCREF(((PyObject *)arrayObject));
    __pyx_r = ((PyObject *)arrayObject);
    goto __pyx_L0;
  }
  __pyx_L2:;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("mtrand.cont1_array_sc");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k61p;

static char __pyx_k61[] = "size is not compatible with inputs";

static PyObject *__pyx_f_6mtrand_cont1_array(rk_state *__pyx_v_state,__pyx_t_6mtrand_rk_cont1 __pyx_v_func,PyObject *__pyx_v_size,PyArrayObject *__pyx_v_oa) {
  double *__pyx_v_array_data;
  double *__pyx_v_oa_data;
  PyArrayObject *arrayObject;
  npy_intp __pyx_v_length;
  npy_intp __pyx_v_i;
  PyArrayIterObject *__pyx_v_itera;
  PyArrayMultiIterObject *__pyx_v_multi;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  npy_intp __pyx_5;
  Py_INCREF(__pyx_v_size);
  Py_INCREF(__pyx_v_oa);
  arrayObject = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_itera = ((PyArrayIterObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_multi = ((PyArrayMultiIterObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":167 */
  __pyx_1 = __pyx_v_size == Py_None;
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":168 */
    __pyx_2 = PyArray_SimpleNew(__pyx_v_oa->nd,__pyx_v_oa->dimensions,NPY_DOUBLE); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 168; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_2);
    Py_DECREF(__pyx_2); __pyx_2 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":169 */
    __pyx_v_length = PyArray_SIZE(arrayObject);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":170 */
    __pyx_v_array_data = ((double *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":171 */
    __pyx_2 = PyArray_IterNew(((PyObject *)__pyx_v_oa)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 171; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayIterObject *)__pyx_2)));
    Py_DECREF(((PyObject *)__pyx_v_itera));
    __pyx_v_itera = ((PyArrayIterObject *)__pyx_2);
    Py_DECREF(__pyx_2); __pyx_2 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":172 */
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_v_length; ++__pyx_v_i) {

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":173 */
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,(((double *)__pyx_v_itera->dataptr)[0]));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":174 */
      PyArray_ITER_NEXT(__pyx_v_itera);
    }
    goto __pyx_L2;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":176 */
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 176; goto __pyx_L1;}
    __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_empty); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 176; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 176; goto __pyx_L1;}
    __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_float64); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 176; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 176; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 1, __pyx_4);
    __pyx_4 = 0;
    __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 176; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_4)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_4);
    Py_DECREF(__pyx_4); __pyx_4 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":177 */
    __pyx_v_array_data = ((double *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":178 */
    __pyx_3 = PyArray_MultiIterNew(2,((void *)arrayObject),((void *)__pyx_v_oa)); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 178; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayMultiIterObject *)__pyx_3)));
    Py_DECREF(((PyObject *)__pyx_v_multi));
    __pyx_v_multi = ((PyArrayMultiIterObject *)__pyx_3);
    Py_DECREF(__pyx_3); __pyx_3 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":180 */
    __pyx_1 = (__pyx_v_multi->size != PyArray_SIZE(arrayObject));
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 181; goto __pyx_L1;}
      Py_INCREF(__pyx_k61p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k61p);
      __pyx_4 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 181; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_4, 0, 0);
      Py_DECREF(__pyx_4); __pyx_4 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 181; goto __pyx_L1;}
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":182 */
    __pyx_5 = __pyx_v_multi->size;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_5; ++__pyx_v_i) {

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":183 */
      __pyx_v_oa_data = ((double *)PyArray_MultiIter_DATA(__pyx_v_multi,1));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":184 */
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,(__pyx_v_oa_data[0]));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":185 */
      PyArray_MultiIter_NEXTi(__pyx_v_multi,1);
    }
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":186 */
  Py_INCREF(((PyObject *)arrayObject));
  __pyx_r = ((PyObject *)arrayObject);
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("mtrand.cont1_array");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject);
  Py_DECREF(__pyx_v_itera);
  Py_DECREF(__pyx_v_multi);
  Py_DECREF(__pyx_v_size);
  Py_DECREF(__pyx_v_oa);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_cont2_array_sc(rk_state *__pyx_v_state,__pyx_t_6mtrand_rk_cont2 __pyx_v_func,PyObject *__pyx_v_size,double __pyx_v_a,double __pyx_v_b) {
  double *__pyx_v_array_data;
  PyArrayObject *arrayObject;
  long __pyx_v_length;
  long __pyx_v_i;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  Py_INCREF(__pyx_v_size);
  arrayObject = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":195 */
  __pyx_1 = __pyx_v_size == Py_None;
  if (__pyx_1) {
    __pyx_2 = PyFloat_FromDouble(__pyx_v_func(__pyx_v_state,__pyx_v_a,__pyx_v_b)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 196; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":198 */
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 198; goto __pyx_L1;}
    __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_empty); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 198; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 198; goto __pyx_L1;}
    __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_float64); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 198; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 198; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 1, __pyx_4);
    __pyx_4 = 0;
    __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 198; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_4)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_4);
    Py_DECREF(__pyx_4); __pyx_4 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":199 */
    __pyx_v_length = PyArray_SIZE(arrayObject);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":200 */
    __pyx_v_array_data = ((double *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":201 */
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_v_length; ++__pyx_v_i) {
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,__pyx_v_a,__pyx_v_b);
    }

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":203 */
    Py_INCREF(((PyObject *)arrayObject));
    __pyx_r = ((PyObject *)arrayObject);
    goto __pyx_L0;
  }
  __pyx_L2:;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("mtrand.cont2_array_sc");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k62p;

static char __pyx_k62[] = "size is not compatible with inputs";

static PyObject *__pyx_f_6mtrand_cont2_array(rk_state *__pyx_v_state,__pyx_t_6mtrand_rk_cont2 __pyx_v_func,PyObject *__pyx_v_size,PyArrayObject *__pyx_v_oa,PyArrayObject *__pyx_v_ob) {
  double *__pyx_v_array_data;
  double *__pyx_v_oa_data;
  double *__pyx_v_ob_data;
  PyArrayObject *arrayObject;
  npy_intp __pyx_v_i;
  PyArrayMultiIterObject *__pyx_v_multi;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  npy_intp __pyx_3;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  Py_INCREF(__pyx_v_size);
  Py_INCREF(__pyx_v_oa);
  Py_INCREF(__pyx_v_ob);
  arrayObject = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_multi = ((PyArrayMultiIterObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":216 */
  __pyx_1 = __pyx_v_size == Py_None;
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":217 */
    __pyx_2 = PyArray_MultiIterNew(2,((void *)__pyx_v_oa),((void *)__pyx_v_ob)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 217; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayMultiIterObject *)__pyx_2)));
    Py_DECREF(((PyObject *)__pyx_v_multi));
    __pyx_v_multi = ((PyArrayMultiIterObject *)__pyx_2);
    Py_DECREF(__pyx_2); __pyx_2 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":218 */
    __pyx_2 = PyArray_SimpleNew(__pyx_v_multi->nd,__pyx_v_multi->dimensions,NPY_DOUBLE); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 218; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_2);
    Py_DECREF(__pyx_2); __pyx_2 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":219 */
    __pyx_v_array_data = ((double *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":220 */
    __pyx_3 = __pyx_v_multi->size;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_3; ++__pyx_v_i) {

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":221 */
      __pyx_v_oa_data = ((double *)PyArray_MultiIter_DATA(__pyx_v_multi,0));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":222 */
      __pyx_v_ob_data = ((double *)PyArray_MultiIter_DATA(__pyx_v_multi,1));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":223 */
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,(__pyx_v_oa_data[0]),(__pyx_v_ob_data[0]));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":224 */
      PyArray_MultiIter_NEXT(__pyx_v_multi);
    }
    goto __pyx_L2;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":226 */
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 226; goto __pyx_L1;}
    __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_empty); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 226; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 226; goto __pyx_L1;}
    __pyx_5 = PyObject_GetAttr(__pyx_2, __pyx_n_float64); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 226; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 226; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 1, __pyx_5);
    __pyx_5 = 0;
    __pyx_5 = PyObject_CallObject(__pyx_4, __pyx_2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 226; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_5)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_5);
    Py_DECREF(__pyx_5); __pyx_5 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":227 */
    __pyx_v_array_data = ((double *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":228 */
    __pyx_4 = PyArray_MultiIterNew(3,((void *)arrayObject),((void *)__pyx_v_oa),((void *)__pyx_v_ob)); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 228; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayMultiIterObject *)__pyx_4)));
    Py_DECREF(((PyObject *)__pyx_v_multi));
    __pyx_v_multi = ((PyArrayMultiIterObject *)__pyx_4);
    Py_DECREF(__pyx_4); __pyx_4 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":229 */
    __pyx_1 = (__pyx_v_multi->size != PyArray_SIZE(arrayObject));
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 230; goto __pyx_L1;}
      Py_INCREF(__pyx_k62p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k62p);
      __pyx_5 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 230; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_5, 0, 0);
      Py_DECREF(__pyx_5); __pyx_5 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 230; goto __pyx_L1;}
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":231 */
    __pyx_3 = __pyx_v_multi->size;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_3; ++__pyx_v_i) {

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":232 */
      __pyx_v_oa_data = ((double *)PyArray_MultiIter_DATA(__pyx_v_multi,1));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":233 */
      __pyx_v_ob_data = ((double *)PyArray_MultiIter_DATA(__pyx_v_multi,2));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":234 */
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,(__pyx_v_oa_data[0]),(__pyx_v_ob_data[0]));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":235 */
      PyArray_MultiIter_NEXTi(__pyx_v_multi,1);

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":236 */
      PyArray_MultiIter_NEXTi(__pyx_v_multi,2);
    }
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":237 */
  Py_INCREF(((PyObject *)arrayObject));
  __pyx_r = ((PyObject *)arrayObject);
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.cont2_array");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject);
  Py_DECREF(__pyx_v_multi);
  Py_DECREF(__pyx_v_size);
  Py_DECREF(__pyx_v_oa);
  Py_DECREF(__pyx_v_ob);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_cont3_array_sc(rk_state *__pyx_v_state,__pyx_t_6mtrand_rk_cont3 __pyx_v_func,PyObject *__pyx_v_size,double __pyx_v_a,double __pyx_v_b,double __pyx_v_c) {
  double *__pyx_v_array_data;
  PyArrayObject *arrayObject;
  long __pyx_v_length;
  long __pyx_v_i;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  Py_INCREF(__pyx_v_size);
  arrayObject = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":247 */
  __pyx_1 = __pyx_v_size == Py_None;
  if (__pyx_1) {
    __pyx_2 = PyFloat_FromDouble(__pyx_v_func(__pyx_v_state,__pyx_v_a,__pyx_v_b,__pyx_v_c)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 248; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":250 */
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 250; goto __pyx_L1;}
    __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_empty); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 250; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 250; goto __pyx_L1;}
    __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_float64); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 250; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 250; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 1, __pyx_4);
    __pyx_4 = 0;
    __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 250; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_4)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_4);
    Py_DECREF(__pyx_4); __pyx_4 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":251 */
    __pyx_v_length = PyArray_SIZE(arrayObject);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":252 */
    __pyx_v_array_data = ((double *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":253 */
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_v_length; ++__pyx_v_i) {
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,__pyx_v_a,__pyx_v_b,__pyx_v_c);
    }

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":255 */
    Py_INCREF(((PyObject *)arrayObject));
    __pyx_r = ((PyObject *)arrayObject);
    goto __pyx_L0;
  }
  __pyx_L2:;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("mtrand.cont3_array_sc");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k63p;

static char __pyx_k63[] = "size is not compatible with inputs";

static PyObject *__pyx_f_6mtrand_cont3_array(rk_state *__pyx_v_state,__pyx_t_6mtrand_rk_cont3 __pyx_v_func,PyObject *__pyx_v_size,PyArrayObject *__pyx_v_oa,PyArrayObject *__pyx_v_ob,PyArrayObject *__pyx_v_oc) {
  double *__pyx_v_array_data;
  double *__pyx_v_oa_data;
  double *__pyx_v_ob_data;
  double *__pyx_v_oc_data;
  PyArrayObject *arrayObject;
  npy_intp __pyx_v_i;
  PyArrayMultiIterObject *__pyx_v_multi;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  npy_intp __pyx_3;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  Py_INCREF(__pyx_v_size);
  Py_INCREF(__pyx_v_oa);
  Py_INCREF(__pyx_v_ob);
  Py_INCREF(__pyx_v_oc);
  arrayObject = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_multi = ((PyArrayMultiIterObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":269 */
  __pyx_1 = __pyx_v_size == Py_None;
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":270 */
    __pyx_2 = PyArray_MultiIterNew(3,((void *)__pyx_v_oa),((void *)__pyx_v_ob),((void *)__pyx_v_oc)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 270; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayMultiIterObject *)__pyx_2)));
    Py_DECREF(((PyObject *)__pyx_v_multi));
    __pyx_v_multi = ((PyArrayMultiIterObject *)__pyx_2);
    Py_DECREF(__pyx_2); __pyx_2 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":271 */
    __pyx_2 = PyArray_SimpleNew(__pyx_v_multi->nd,__pyx_v_multi->dimensions,NPY_DOUBLE); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 271; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_2);
    Py_DECREF(__pyx_2); __pyx_2 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":272 */
    __pyx_v_array_data = ((double *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":273 */
    __pyx_3 = __pyx_v_multi->size;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_3; ++__pyx_v_i) {

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":274 */
      __pyx_v_oa_data = ((double *)PyArray_MultiIter_DATA(__pyx_v_multi,0));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":275 */
      __pyx_v_ob_data = ((double *)PyArray_MultiIter_DATA(__pyx_v_multi,1));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":276 */
      __pyx_v_oc_data = ((double *)PyArray_MultiIter_DATA(__pyx_v_multi,2));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":277 */
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,(__pyx_v_oa_data[0]),(__pyx_v_ob_data[0]),(__pyx_v_oc_data[0]));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":278 */
      PyArray_MultiIter_NEXT(__pyx_v_multi);
    }
    goto __pyx_L2;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":280 */
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 280; goto __pyx_L1;}
    __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_empty); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 280; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 280; goto __pyx_L1;}
    __pyx_5 = PyObject_GetAttr(__pyx_2, __pyx_n_float64); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 280; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 280; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 1, __pyx_5);
    __pyx_5 = 0;
    __pyx_5 = PyObject_CallObject(__pyx_4, __pyx_2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 280; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_5)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_5);
    Py_DECREF(__pyx_5); __pyx_5 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":281 */
    __pyx_v_array_data = ((double *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":282 */
    __pyx_4 = PyArray_MultiIterNew(4,((void *)arrayObject),((void *)__pyx_v_oa),((void *)__pyx_v_ob),((void *)__pyx_v_oc)); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 282; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayMultiIterObject *)__pyx_4)));
    Py_DECREF(((PyObject *)__pyx_v_multi));
    __pyx_v_multi = ((PyArrayMultiIterObject *)__pyx_4);
    Py_DECREF(__pyx_4); __pyx_4 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":284 */
    __pyx_1 = (__pyx_v_multi->size != PyArray_SIZE(arrayObject));
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 285; goto __pyx_L1;}
      Py_INCREF(__pyx_k63p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k63p);
      __pyx_5 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 285; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_5, 0, 0);
      Py_DECREF(__pyx_5); __pyx_5 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 285; goto __pyx_L1;}
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":286 */
    __pyx_3 = __pyx_v_multi->size;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_3; ++__pyx_v_i) {

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":287 */
      __pyx_v_oa_data = ((double *)PyArray_MultiIter_DATA(__pyx_v_multi,1));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":288 */
      __pyx_v_ob_data = ((double *)PyArray_MultiIter_DATA(__pyx_v_multi,2));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":289 */
      __pyx_v_oc_data = ((double *)PyArray_MultiIter_DATA(__pyx_v_multi,3));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":290 */
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,(__pyx_v_oa_data[0]),(__pyx_v_ob_data[0]),(__pyx_v_oc_data[0]));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":291 */
      PyArray_MultiIter_NEXT(__pyx_v_multi);
    }
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":292 */
  Py_INCREF(((PyObject *)arrayObject));
  __pyx_r = ((PyObject *)arrayObject);
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.cont3_array");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject);
  Py_DECREF(__pyx_v_multi);
  Py_DECREF(__pyx_v_size);
  Py_DECREF(__pyx_v_oa);
  Py_DECREF(__pyx_v_ob);
  Py_DECREF(__pyx_v_oc);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_disc0_array(rk_state *__pyx_v_state,__pyx_t_6mtrand_rk_disc0 __pyx_v_func,PyObject *__pyx_v_size) {
  long *__pyx_v_array_data;
  PyArrayObject *arrayObject;
  long __pyx_v_length;
  long __pyx_v_i;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  Py_INCREF(__pyx_v_size);
  arrayObject = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":300 */
  __pyx_1 = __pyx_v_size == Py_None;
  if (__pyx_1) {
    __pyx_2 = PyInt_FromLong(__pyx_v_func(__pyx_v_state)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 301; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":303 */
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 303; goto __pyx_L1;}
    __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_empty); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 303; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 303; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_size);
    Py_INCREF(((PyObject *)(&PyInt_Type)));
    PyTuple_SET_ITEM(__pyx_2, 1, ((PyObject *)(&PyInt_Type)));
    __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 303; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_4)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_4);
    Py_DECREF(__pyx_4); __pyx_4 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":304 */
    __pyx_v_length = PyArray_SIZE(arrayObject);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":305 */
    __pyx_v_array_data = ((long *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":306 */
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_v_length; ++__pyx_v_i) {
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state);
    }

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":308 */
    Py_INCREF(((PyObject *)arrayObject));
    __pyx_r = ((PyObject *)arrayObject);
    goto __pyx_L0;
  }
  __pyx_L2:;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("mtrand.disc0_array");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_discnp_array_sc(rk_state *__pyx_v_state,__pyx_t_6mtrand_rk_discnp __pyx_v_func,PyObject *__pyx_v_size,long __pyx_v_n,double __pyx_v_p) {
  long *__pyx_v_array_data;
  PyArrayObject *arrayObject;
  long __pyx_v_length;
  long __pyx_v_i;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  Py_INCREF(__pyx_v_size);
  arrayObject = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":316 */
  __pyx_1 = __pyx_v_size == Py_None;
  if (__pyx_1) {
    __pyx_2 = PyInt_FromLong(__pyx_v_func(__pyx_v_state,__pyx_v_n,__pyx_v_p)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 317; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":319 */
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 319; goto __pyx_L1;}
    __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_empty); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 319; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 319; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_size);
    Py_INCREF(((PyObject *)(&PyInt_Type)));
    PyTuple_SET_ITEM(__pyx_2, 1, ((PyObject *)(&PyInt_Type)));
    __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 319; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_4)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_4);
    Py_DECREF(__pyx_4); __pyx_4 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":320 */
    __pyx_v_length = PyArray_SIZE(arrayObject);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":321 */
    __pyx_v_array_data = ((long *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":322 */
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_v_length; ++__pyx_v_i) {
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,__pyx_v_n,__pyx_v_p);
    }

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":324 */
    Py_INCREF(((PyObject *)arrayObject));
    __pyx_r = ((PyObject *)arrayObject);
    goto __pyx_L0;
  }
  __pyx_L2:;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("mtrand.discnp_array_sc");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k64p;

static char __pyx_k64[] = "size is not compatible with inputs";

static PyObject *__pyx_f_6mtrand_discnp_array(rk_state *__pyx_v_state,__pyx_t_6mtrand_rk_discnp __pyx_v_func,PyObject *__pyx_v_size,PyArrayObject *__pyx_v_on,PyArrayObject *__pyx_v_op) {
  long *__pyx_v_array_data;
  PyArrayObject *arrayObject;
  npy_intp __pyx_v_i;
  double *__pyx_v_op_data;
  long *__pyx_v_on_data;
  PyArrayMultiIterObject *__pyx_v_multi;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  npy_intp __pyx_3;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  Py_INCREF(__pyx_v_size);
  Py_INCREF(__pyx_v_on);
  Py_INCREF(__pyx_v_op);
  arrayObject = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_multi = ((PyArrayMultiIterObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":335 */
  __pyx_1 = __pyx_v_size == Py_None;
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":336 */
    __pyx_2 = PyArray_MultiIterNew(2,((void *)__pyx_v_on),((void *)__pyx_v_op)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 336; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayMultiIterObject *)__pyx_2)));
    Py_DECREF(((PyObject *)__pyx_v_multi));
    __pyx_v_multi = ((PyArrayMultiIterObject *)__pyx_2);
    Py_DECREF(__pyx_2); __pyx_2 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":337 */
    __pyx_2 = PyArray_SimpleNew(__pyx_v_multi->nd,__pyx_v_multi->dimensions,NPY_LONG); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 337; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_2);
    Py_DECREF(__pyx_2); __pyx_2 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":338 */
    __pyx_v_array_data = ((long *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":339 */
    __pyx_3 = __pyx_v_multi->size;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_3; ++__pyx_v_i) {

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":340 */
      __pyx_v_on_data = ((long *)PyArray_MultiIter_DATA(__pyx_v_multi,0));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":341 */
      __pyx_v_op_data = ((double *)PyArray_MultiIter_DATA(__pyx_v_multi,1));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":342 */
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,(__pyx_v_on_data[0]),(__pyx_v_op_data[0]));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":343 */
      PyArray_MultiIter_NEXT(__pyx_v_multi);
    }
    goto __pyx_L2;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":345 */
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 345; goto __pyx_L1;}
    __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_empty); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 345; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 345; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_size);
    Py_INCREF(((PyObject *)(&PyInt_Type)));
    PyTuple_SET_ITEM(__pyx_2, 1, ((PyObject *)(&PyInt_Type)));
    __pyx_5 = PyObject_CallObject(__pyx_4, __pyx_2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 345; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_5)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_5);
    Py_DECREF(__pyx_5); __pyx_5 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":346 */
    __pyx_v_array_data = ((long *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":347 */
    __pyx_4 = PyArray_MultiIterNew(3,((void *)arrayObject),((void *)__pyx_v_on),((void *)__pyx_v_op)); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 347; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayMultiIterObject *)__pyx_4)));
    Py_DECREF(((PyObject *)__pyx_v_multi));
    __pyx_v_multi = ((PyArrayMultiIterObject *)__pyx_4);
    Py_DECREF(__pyx_4); __pyx_4 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":348 */
    __pyx_1 = (__pyx_v_multi->size != PyArray_SIZE(arrayObject));
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 349; goto __pyx_L1;}
      Py_INCREF(__pyx_k64p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k64p);
      __pyx_5 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 349; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_5, 0, 0);
      Py_DECREF(__pyx_5); __pyx_5 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 349; goto __pyx_L1;}
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":350 */
    __pyx_3 = __pyx_v_multi->size;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_3; ++__pyx_v_i) {

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":351 */
      __pyx_v_on_data = ((long *)PyArray_MultiIter_DATA(__pyx_v_multi,1));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":352 */
      __pyx_v_op_data = ((double *)PyArray_MultiIter_DATA(__pyx_v_multi,2));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":353 */
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,(__pyx_v_on_data[0]),(__pyx_v_op_data[0]));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":354 */
      PyArray_MultiIter_NEXTi(__pyx_v_multi,1);

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":355 */
      PyArray_MultiIter_NEXTi(__pyx_v_multi,2);
    }
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":357 */
  Py_INCREF(((PyObject *)arrayObject));
  __pyx_r = ((PyObject *)arrayObject);
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.discnp_array");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject);
  Py_DECREF(__pyx_v_multi);
  Py_DECREF(__pyx_v_size);
  Py_DECREF(__pyx_v_on);
  Py_DECREF(__pyx_v_op);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_discdd_array_sc(rk_state *__pyx_v_state,__pyx_t_6mtrand_rk_discdd __pyx_v_func,PyObject *__pyx_v_size,double __pyx_v_n,double __pyx_v_p) {
  long *__pyx_v_array_data;
  PyArrayObject *arrayObject;
  long __pyx_v_length;
  long __pyx_v_i;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  Py_INCREF(__pyx_v_size);
  arrayObject = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":365 */
  __pyx_1 = __pyx_v_size == Py_None;
  if (__pyx_1) {
    __pyx_2 = PyInt_FromLong(__pyx_v_func(__pyx_v_state,__pyx_v_n,__pyx_v_p)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 366; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":368 */
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 368; goto __pyx_L1;}
    __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_empty); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 368; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 368; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_size);
    Py_INCREF(((PyObject *)(&PyInt_Type)));
    PyTuple_SET_ITEM(__pyx_2, 1, ((PyObject *)(&PyInt_Type)));
    __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 368; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_4)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_4);
    Py_DECREF(__pyx_4); __pyx_4 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":369 */
    __pyx_v_length = PyArray_SIZE(arrayObject);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":370 */
    __pyx_v_array_data = ((long *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":371 */
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_v_length; ++__pyx_v_i) {
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,__pyx_v_n,__pyx_v_p);
    }

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":373 */
    Py_INCREF(((PyObject *)arrayObject));
    __pyx_r = ((PyObject *)arrayObject);
    goto __pyx_L0;
  }
  __pyx_L2:;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("mtrand.discdd_array_sc");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k65p;

static char __pyx_k65[] = "size is not compatible with inputs";

static PyObject *__pyx_f_6mtrand_discdd_array(rk_state *__pyx_v_state,__pyx_t_6mtrand_rk_discdd __pyx_v_func,PyObject *__pyx_v_size,PyArrayObject *__pyx_v_on,PyArrayObject *__pyx_v_op) {
  long *__pyx_v_array_data;
  PyArrayObject *arrayObject;
  npy_intp __pyx_v_i;
  double *__pyx_v_op_data;
  double *__pyx_v_on_data;
  PyArrayMultiIterObject *__pyx_v_multi;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  npy_intp __pyx_3;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  Py_INCREF(__pyx_v_size);
  Py_INCREF(__pyx_v_on);
  Py_INCREF(__pyx_v_op);
  arrayObject = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_multi = ((PyArrayMultiIterObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":384 */
  __pyx_1 = __pyx_v_size == Py_None;
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":385 */
    __pyx_2 = PyArray_MultiIterNew(2,((void *)__pyx_v_on),((void *)__pyx_v_op)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 385; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayMultiIterObject *)__pyx_2)));
    Py_DECREF(((PyObject *)__pyx_v_multi));
    __pyx_v_multi = ((PyArrayMultiIterObject *)__pyx_2);
    Py_DECREF(__pyx_2); __pyx_2 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":386 */
    __pyx_2 = PyArray_SimpleNew(__pyx_v_multi->nd,__pyx_v_multi->dimensions,NPY_LONG); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 386; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_2);
    Py_DECREF(__pyx_2); __pyx_2 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":387 */
    __pyx_v_array_data = ((long *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":388 */
    __pyx_3 = __pyx_v_multi->size;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_3; ++__pyx_v_i) {

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":389 */
      __pyx_v_on_data = ((double *)PyArray_MultiIter_DATA(__pyx_v_multi,0));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":390 */
      __pyx_v_op_data = ((double *)PyArray_MultiIter_DATA(__pyx_v_multi,1));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":391 */
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,(__pyx_v_on_data[0]),(__pyx_v_op_data[0]));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":392 */
      PyArray_MultiIter_NEXT(__pyx_v_multi);
    }
    goto __pyx_L2;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":394 */
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 394; goto __pyx_L1;}
    __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_empty); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 394; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 394; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_size);
    Py_INCREF(((PyObject *)(&PyInt_Type)));
    PyTuple_SET_ITEM(__pyx_2, 1, ((PyObject *)(&PyInt_Type)));
    __pyx_5 = PyObject_CallObject(__pyx_4, __pyx_2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 394; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_5)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_5);
    Py_DECREF(__pyx_5); __pyx_5 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":395 */
    __pyx_v_array_data = ((long *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":396 */
    __pyx_4 = PyArray_MultiIterNew(3,((void *)arrayObject),((void *)__pyx_v_on),((void *)__pyx_v_op)); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 396; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayMultiIterObject *)__pyx_4)));
    Py_DECREF(((PyObject *)__pyx_v_multi));
    __pyx_v_multi = ((PyArrayMultiIterObject *)__pyx_4);
    Py_DECREF(__pyx_4); __pyx_4 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":397 */
    __pyx_1 = (__pyx_v_multi->size != PyArray_SIZE(arrayObject));
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 398; goto __pyx_L1;}
      Py_INCREF(__pyx_k65p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k65p);
      __pyx_5 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 398; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_5, 0, 0);
      Py_DECREF(__pyx_5); __pyx_5 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 398; goto __pyx_L1;}
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":399 */
    __pyx_3 = __pyx_v_multi->size;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_3; ++__pyx_v_i) {

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":400 */
      __pyx_v_on_data = ((double *)PyArray_MultiIter_DATA(__pyx_v_multi,1));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":401 */
      __pyx_v_op_data = ((double *)PyArray_MultiIter_DATA(__pyx_v_multi,2));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":402 */
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,(__pyx_v_on_data[0]),(__pyx_v_op_data[0]));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":403 */
      PyArray_MultiIter_NEXTi(__pyx_v_multi,1);

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":404 */
      PyArray_MultiIter_NEXTi(__pyx_v_multi,2);
    }
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":406 */
  Py_INCREF(((PyObject *)arrayObject));
  __pyx_r = ((PyObject *)arrayObject);
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.discdd_array");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject);
  Py_DECREF(__pyx_v_multi);
  Py_DECREF(__pyx_v_size);
  Py_DECREF(__pyx_v_on);
  Py_DECREF(__pyx_v_op);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_discnmN_array_sc(rk_state *__pyx_v_state,__pyx_t_6mtrand_rk_discnmN __pyx_v_func,PyObject *__pyx_v_size,long __pyx_v_n,long __pyx_v_m,long __pyx_v_N) {
  long *__pyx_v_array_data;
  PyArrayObject *arrayObject;
  long __pyx_v_length;
  long __pyx_v_i;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  Py_INCREF(__pyx_v_size);
  arrayObject = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":415 */
  __pyx_1 = __pyx_v_size == Py_None;
  if (__pyx_1) {
    __pyx_2 = PyInt_FromLong(__pyx_v_func(__pyx_v_state,__pyx_v_n,__pyx_v_m,__pyx_v_N)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 416; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":418 */
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 418; goto __pyx_L1;}
    __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_empty); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 418; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 418; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_size);
    Py_INCREF(((PyObject *)(&PyInt_Type)));
    PyTuple_SET_ITEM(__pyx_2, 1, ((PyObject *)(&PyInt_Type)));
    __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 418; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_4)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_4);
    Py_DECREF(__pyx_4); __pyx_4 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":419 */
    __pyx_v_length = PyArray_SIZE(arrayObject);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":420 */
    __pyx_v_array_data = ((long *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":421 */
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_v_length; ++__pyx_v_i) {
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,__pyx_v_n,__pyx_v_m,__pyx_v_N);
    }

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":423 */
    Py_INCREF(((PyObject *)arrayObject));
    __pyx_r = ((PyObject *)arrayObject);
    goto __pyx_L0;
  }
  __pyx_L2:;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("mtrand.discnmN_array_sc");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k66p;

static char __pyx_k66[] = "size is not compatible with inputs";

static PyObject *__pyx_f_6mtrand_discnmN_array(rk_state *__pyx_v_state,__pyx_t_6mtrand_rk_discnmN __pyx_v_func,PyObject *__pyx_v_size,PyArrayObject *__pyx_v_on,PyArrayObject *__pyx_v_om,PyArrayObject *__pyx_v_oN) {
  long *__pyx_v_array_data;
  long *__pyx_v_on_data;
  long *__pyx_v_om_data;
  long *__pyx_v_oN_data;
  PyArrayObject *arrayObject;
  npy_intp __pyx_v_i;
  PyArrayMultiIterObject *__pyx_v_multi;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  npy_intp __pyx_3;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  Py_INCREF(__pyx_v_size);
  Py_INCREF(__pyx_v_on);
  Py_INCREF(__pyx_v_om);
  Py_INCREF(__pyx_v_oN);
  arrayObject = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_multi = ((PyArrayMultiIterObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":436 */
  __pyx_1 = __pyx_v_size == Py_None;
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":437 */
    __pyx_2 = PyArray_MultiIterNew(3,((void *)__pyx_v_on),((void *)__pyx_v_om),((void *)__pyx_v_oN)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 437; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayMultiIterObject *)__pyx_2)));
    Py_DECREF(((PyObject *)__pyx_v_multi));
    __pyx_v_multi = ((PyArrayMultiIterObject *)__pyx_2);
    Py_DECREF(__pyx_2); __pyx_2 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":438 */
    __pyx_2 = PyArray_SimpleNew(__pyx_v_multi->nd,__pyx_v_multi->dimensions,NPY_LONG); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 438; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_2);
    Py_DECREF(__pyx_2); __pyx_2 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":439 */
    __pyx_v_array_data = ((long *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":440 */
    __pyx_3 = __pyx_v_multi->size;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_3; ++__pyx_v_i) {

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":441 */
      __pyx_v_on_data = ((long *)PyArray_MultiIter_DATA(__pyx_v_multi,0));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":442 */
      __pyx_v_om_data = ((long *)PyArray_MultiIter_DATA(__pyx_v_multi,1));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":443 */
      __pyx_v_oN_data = ((long *)PyArray_MultiIter_DATA(__pyx_v_multi,2));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":444 */
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,(__pyx_v_on_data[0]),(__pyx_v_om_data[0]),(__pyx_v_oN_data[0]));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":445 */
      PyArray_MultiIter_NEXT(__pyx_v_multi);
    }
    goto __pyx_L2;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":447 */
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 447; goto __pyx_L1;}
    __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_empty); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 447; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 447; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_size);
    Py_INCREF(((PyObject *)(&PyInt_Type)));
    PyTuple_SET_ITEM(__pyx_2, 1, ((PyObject *)(&PyInt_Type)));
    __pyx_5 = PyObject_CallObject(__pyx_4, __pyx_2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 447; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_5)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_5);
    Py_DECREF(__pyx_5); __pyx_5 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":448 */
    __pyx_v_array_data = ((long *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":449 */
    __pyx_4 = PyArray_MultiIterNew(4,((void *)arrayObject),((void *)__pyx_v_on),((void *)__pyx_v_om),((void *)__pyx_v_oN)); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 449; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayMultiIterObject *)__pyx_4)));
    Py_DECREF(((PyObject *)__pyx_v_multi));
    __pyx_v_multi = ((PyArrayMultiIterObject *)__pyx_4);
    Py_DECREF(__pyx_4); __pyx_4 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":451 */
    __pyx_1 = (__pyx_v_multi->size != PyArray_SIZE(arrayObject));
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 452; goto __pyx_L1;}
      Py_INCREF(__pyx_k66p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k66p);
      __pyx_5 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 452; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_5, 0, 0);
      Py_DECREF(__pyx_5); __pyx_5 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 452; goto __pyx_L1;}
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":453 */
    __pyx_3 = __pyx_v_multi->size;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_3; ++__pyx_v_i) {

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":454 */
      __pyx_v_on_data = ((long *)PyArray_MultiIter_DATA(__pyx_v_multi,1));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":455 */
      __pyx_v_om_data = ((long *)PyArray_MultiIter_DATA(__pyx_v_multi,2));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":456 */
      __pyx_v_oN_data = ((long *)PyArray_MultiIter_DATA(__pyx_v_multi,3));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":457 */
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,(__pyx_v_on_data[0]),(__pyx_v_om_data[0]),(__pyx_v_oN_data[0]));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":458 */
      PyArray_MultiIter_NEXT(__pyx_v_multi);
    }
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":460 */
  Py_INCREF(((PyObject *)arrayObject));
  __pyx_r = ((PyObject *)arrayObject);
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.discnmN_array");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject);
  Py_DECREF(__pyx_v_multi);
  Py_DECREF(__pyx_v_size);
  Py_DECREF(__pyx_v_on);
  Py_DECREF(__pyx_v_om);
  Py_DECREF(__pyx_v_oN);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_discd_array_sc(rk_state *__pyx_v_state,__pyx_t_6mtrand_rk_discd __pyx_v_func,PyObject *__pyx_v_size,double __pyx_v_a) {
  long *__pyx_v_array_data;
  PyArrayObject *arrayObject;
  long __pyx_v_length;
  long __pyx_v_i;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  Py_INCREF(__pyx_v_size);
  arrayObject = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":468 */
  __pyx_1 = __pyx_v_size == Py_None;
  if (__pyx_1) {
    __pyx_2 = PyInt_FromLong(__pyx_v_func(__pyx_v_state,__pyx_v_a)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 469; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":471 */
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 471; goto __pyx_L1;}
    __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_empty); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 471; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 471; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_size);
    Py_INCREF(((PyObject *)(&PyInt_Type)));
    PyTuple_SET_ITEM(__pyx_2, 1, ((PyObject *)(&PyInt_Type)));
    __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 471; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_4)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_4);
    Py_DECREF(__pyx_4); __pyx_4 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":472 */
    __pyx_v_length = PyArray_SIZE(arrayObject);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":473 */
    __pyx_v_array_data = ((long *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":474 */
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_v_length; ++__pyx_v_i) {
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,__pyx_v_a);
    }

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":476 */
    Py_INCREF(((PyObject *)arrayObject));
    __pyx_r = ((PyObject *)arrayObject);
    goto __pyx_L0;
  }
  __pyx_L2:;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("mtrand.discd_array_sc");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k67p;

static char __pyx_k67[] = "size is not compatible with inputs";

static PyObject *__pyx_f_6mtrand_discd_array(rk_state *__pyx_v_state,__pyx_t_6mtrand_rk_discd __pyx_v_func,PyObject *__pyx_v_size,PyArrayObject *__pyx_v_oa) {
  long *__pyx_v_array_data;
  double *__pyx_v_oa_data;
  PyArrayObject *arrayObject;
  npy_intp __pyx_v_length;
  npy_intp __pyx_v_i;
  PyArrayMultiIterObject *__pyx_v_multi;
  PyArrayIterObject *__pyx_v_itera;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  npy_intp __pyx_5;
  Py_INCREF(__pyx_v_size);
  Py_INCREF(__pyx_v_oa);
  arrayObject = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_multi = ((PyArrayMultiIterObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_itera = ((PyArrayIterObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":487 */
  __pyx_1 = __pyx_v_size == Py_None;
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":488 */
    __pyx_2 = PyArray_SimpleNew(__pyx_v_oa->nd,__pyx_v_oa->dimensions,NPY_LONG); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 488; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_2);
    Py_DECREF(__pyx_2); __pyx_2 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":489 */
    __pyx_v_length = PyArray_SIZE(arrayObject);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":490 */
    __pyx_v_array_data = ((long *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":491 */
    __pyx_2 = PyArray_IterNew(((PyObject *)__pyx_v_oa)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 491; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayIterObject *)__pyx_2)));
    Py_DECREF(((PyObject *)__pyx_v_itera));
    __pyx_v_itera = ((PyArrayIterObject *)__pyx_2);
    Py_DECREF(__pyx_2); __pyx_2 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":492 */
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_v_length; ++__pyx_v_i) {

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":493 */
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,(((double *)__pyx_v_itera->dataptr)[0]));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":494 */
      PyArray_ITER_NEXT(__pyx_v_itera);
    }
    goto __pyx_L2;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":496 */
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 496; goto __pyx_L1;}
    __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_empty); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 496; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 496; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_size);
    Py_INCREF(((PyObject *)(&PyInt_Type)));
    PyTuple_SET_ITEM(__pyx_2, 1, ((PyObject *)(&PyInt_Type)));
    __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 496; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_4)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_4);
    Py_DECREF(__pyx_4); __pyx_4 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":497 */
    __pyx_v_array_data = ((long *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":498 */
    __pyx_3 = PyArray_MultiIterNew(2,((void *)arrayObject),((void *)__pyx_v_oa)); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 498; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayMultiIterObject *)__pyx_3)));
    Py_DECREF(((PyObject *)__pyx_v_multi));
    __pyx_v_multi = ((PyArrayMultiIterObject *)__pyx_3);
    Py_DECREF(__pyx_3); __pyx_3 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":499 */
    __pyx_1 = (__pyx_v_multi->size != PyArray_SIZE(arrayObject));
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 500; goto __pyx_L1;}
      Py_INCREF(__pyx_k67p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k67p);
      __pyx_4 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 500; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_4, 0, 0);
      Py_DECREF(__pyx_4); __pyx_4 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 500; goto __pyx_L1;}
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":501 */
    __pyx_5 = __pyx_v_multi->size;
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_5; ++__pyx_v_i) {

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":502 */
      __pyx_v_oa_data = ((double *)PyArray_MultiIter_DATA(__pyx_v_multi,1));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":503 */
      (__pyx_v_array_data[__pyx_v_i]) = __pyx_v_func(__pyx_v_state,(__pyx_v_oa_data[0]));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":504 */
      PyArray_MultiIter_NEXTi(__pyx_v_multi,1);
    }
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":505 */
  Py_INCREF(((PyObject *)arrayObject));
  __pyx_r = ((PyObject *)arrayObject);
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("mtrand.discd_array");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject);
  Py_DECREF(__pyx_v_multi);
  Py_DECREF(__pyx_v_itera);
  Py_DECREF(__pyx_v_size);
  Py_DECREF(__pyx_v_oa);
  return __pyx_r;
}

static double __pyx_f_6mtrand_kahan_sum(double *__pyx_v_darr,long __pyx_v_n) {
  double __pyx_v_c;
  double __pyx_v_y;
  double __pyx_v_t;
  double __pyx_v_sum;
  long __pyx_v_i;
  double __pyx_r;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":510 */
  __pyx_v_sum = (__pyx_v_darr[0]);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":511 */
  __pyx_v_c = 0.0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":512 */
  for (__pyx_v_i = 1; __pyx_v_i < __pyx_v_n; ++__pyx_v_i) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":513 */
    __pyx_v_y = ((__pyx_v_darr[__pyx_v_i]) - __pyx_v_c);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":514 */
    __pyx_v_t = (__pyx_v_sum + __pyx_v_y);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":515 */
    __pyx_v_c = ((__pyx_v_t - __pyx_v_sum) - __pyx_v_y);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":516 */
    __pyx_v_sum = __pyx_v_t;
  }

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":517 */
  __pyx_r = __pyx_v_sum;
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

static int __pyx_f_6mtrand_11RandomState___init__(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static int __pyx_f_6mtrand_11RandomState___init__(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_seed = 0;
  int __pyx_r;
  PyObject *__pyx_1 = 0;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  static char *__pyx_argnames[] = {"seed",0};
  __pyx_v_seed = __pyx_k2;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "|O", __pyx_argnames, &__pyx_v_seed)) return -1;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_seed);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":547 */
  ((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state = ((rk_state *)PyMem_Malloc((sizeof(rk_state))));

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":549 */
  __pyx_1 = PyObject_GetAttr(__pyx_v_self, __pyx_n_seed); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 549; goto __pyx_L1;}
  __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 549; goto __pyx_L1;}
  Py_INCREF(__pyx_v_seed);
  PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_seed);
  __pyx_3 = PyObject_CallObject(__pyx_1, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 549; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  __Pyx_AddTraceback("mtrand.RandomState.__init__");
  __pyx_r = -1;
  __pyx_L0:;
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_seed);
  return __pyx_r;
}

static void __pyx_f_6mtrand_11RandomState___dealloc__(PyObject *__pyx_v_self); /*proto*/
static void __pyx_f_6mtrand_11RandomState___dealloc__(PyObject *__pyx_v_self) {
  int __pyx_1;
  Py_INCREF(__pyx_v_self);
  __pyx_1 = (((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state != NULL);
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":553 */
    PyMem_Free(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":554 */
    ((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state = NULL;
    goto __pyx_L2;
  }
  __pyx_L2:;

  Py_DECREF(__pyx_v_self);
}

static PyObject *__pyx_n_integer;

static PyObject *__pyx_f_6mtrand_11RandomState_seed(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_seed[] = "\n        seed(seed=None)\n\n        Seed the generator.\n\n        seed can be an integer, an array (or other sequence) of integers of any\n        length, or None. If seed is None, then RandomState will try to read data\n        from /dev/urandom (or the Windows analogue) if available or seed from\n        the clock otherwise.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_seed(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_seed = 0;
  rk_error __pyx_v_errcode;
  PyArrayObject *arrayObject_obj;
  PyObject *__pyx_v_iseed;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  unsigned long __pyx_4;
  static char *__pyx_argnames[] = {"seed",0};
  __pyx_v_seed = __pyx_k3;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "|O", __pyx_argnames, &__pyx_v_seed)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_seed);
  arrayObject_obj = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_iseed = Py_None; Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":570 */
  __pyx_1 = __pyx_v_seed == Py_None;
  if (__pyx_1) {
    __pyx_v_errcode = rk_randomseed(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state);
    goto __pyx_L2;
  }
  __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 572; goto __pyx_L1;}
  Py_INCREF(__pyx_v_seed);
  PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_seed);
  __pyx_3 = PyObject_CallObject(((PyObject *)(&PyType_Type)), __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 572; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_1 = __pyx_3 == ((PyObject *)(&PyInt_Type));
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  if (__pyx_1) {
    __pyx_4 = PyInt_AsUnsignedLongMask(__pyx_v_seed); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 573; goto __pyx_L1;}
    rk_seed(__pyx_4,((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state);
    goto __pyx_L2;
  }
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 574; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_integer); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 574; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_1 = PyObject_IsInstance(__pyx_v_seed,__pyx_3); if (__pyx_1 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 574; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":575 */
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 575; goto __pyx_L1;}
    Py_INCREF(__pyx_v_seed);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_seed);
    __pyx_3 = PyObject_CallObject(((PyObject *)(&PyInt_Type)), __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 575; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_DECREF(__pyx_v_iseed);
    __pyx_v_iseed = __pyx_3;
    __pyx_3 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":576 */
    __pyx_4 = PyInt_AsUnsignedLongMask(__pyx_v_iseed); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 576; goto __pyx_L1;}
    rk_seed(__pyx_4,((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state);
    goto __pyx_L2;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":578 */
    __pyx_2 = PyArray_ContiguousFromObject(__pyx_v_seed,NPY_LONG,1,1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 578; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
    Py_DECREF(((PyObject *)arrayObject_obj));
    arrayObject_obj = ((PyArrayObject *)__pyx_2);
    Py_DECREF(__pyx_2); __pyx_2 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":579 */
    init_by_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,((unsigned long *)arrayObject_obj->data),(arrayObject_obj->dimensions[0]));
  }
  __pyx_L2:;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  __Pyx_AddTraceback("mtrand.RandomState.seed");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject_obj);
  Py_DECREF(__pyx_v_iseed);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_seed);
  return __pyx_r;
}

static PyObject *__pyx_n_uint;
static PyObject *__pyx_n_asarray;
static PyObject *__pyx_n_uint32;
static PyObject *__pyx_n_MT19937;


static PyObject *__pyx_f_6mtrand_11RandomState_get_state(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_get_state[] = "\n        get_state()\n\n        Return a tuple representing the internal state of the generator.\n\n        Returns\n        -------\n        out : tuple(string, list of 624 integers, int, int, float)\n            The returned tuple has the following items:\n\n            1. the string \'MT19937\'\n            2. a list of 624 integer keys\n            3. an integer pos\n            4. an integer has_gauss\n            5. and a float cached_gaussian\n\n        See Also\n        --------\n        set_state\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_get_state(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyArrayObject *arrayObject_state;
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  static char *__pyx_argnames[] = {0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "", __pyx_argnames)) return 0;
  Py_INCREF(__pyx_v_self);
  arrayObject_state = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":605 */
  __pyx_1 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 605; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_1, __pyx_n_empty); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 605; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  __pyx_1 = PyInt_FromLong(624); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 605; goto __pyx_L1;}
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 605; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_3, __pyx_n_uint); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 605; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyTuple_New(2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 605; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_3, 0, __pyx_1);
  PyTuple_SET_ITEM(__pyx_3, 1, __pyx_4);
  __pyx_1 = 0;
  __pyx_4 = 0;
  __pyx_1 = PyObject_CallObject(__pyx_2, __pyx_3); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 605; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_1)));
  Py_DECREF(((PyObject *)arrayObject_state));
  arrayObject_state = ((PyArrayObject *)__pyx_1);
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":606 */
  memcpy(((void *)arrayObject_state->data),((void *)((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state->key),(624 * (sizeof(long))));

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":607 */
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 607; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_4, __pyx_n_asarray); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 607; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 607; goto __pyx_L1;}
  __pyx_1 = PyObject_GetAttr(__pyx_3, __pyx_n_uint32); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 607; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_4 = PyTuple_New(2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 607; goto __pyx_L1;}
  Py_INCREF(((PyObject *)arrayObject_state));
  PyTuple_SET_ITEM(__pyx_4, 0, ((PyObject *)arrayObject_state));
  PyTuple_SET_ITEM(__pyx_4, 1, __pyx_1);
  __pyx_1 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 607; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)arrayObject_state));
  arrayObject_state = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":608 */
  __pyx_1 = PyInt_FromLong(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state->pos); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 608; goto __pyx_L1;}
  __pyx_2 = PyInt_FromLong(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state->has_gauss); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 609; goto __pyx_L1;}
  __pyx_4 = PyFloat_FromDouble(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state->gauss); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 609; goto __pyx_L1;}
  __pyx_3 = PyTuple_New(5); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 608; goto __pyx_L1;}
  Py_INCREF(__pyx_n_MT19937);
  PyTuple_SET_ITEM(__pyx_3, 0, __pyx_n_MT19937);
  Py_INCREF(((PyObject *)arrayObject_state));
  PyTuple_SET_ITEM(__pyx_3, 1, ((PyObject *)arrayObject_state));
  PyTuple_SET_ITEM(__pyx_3, 2, __pyx_1);
  PyTuple_SET_ITEM(__pyx_3, 3, __pyx_2);
  PyTuple_SET_ITEM(__pyx_3, 4, __pyx_4);
  __pyx_1 = 0;
  __pyx_2 = 0;
  __pyx_4 = 0;
  __pyx_r = __pyx_3;
  __pyx_3 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("mtrand.RandomState.get_state");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject_state);
  Py_DECREF(__pyx_v_self);
  return __pyx_r;
}

static PyObject *__pyx_k70p;
static PyObject *__pyx_k71p;

static char __pyx_k70[] = "algorithm must be 'MT19937'";
static char __pyx_k71[] = "state must be 624 longs";

static PyObject *__pyx_f_6mtrand_11RandomState_set_state(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_set_state[] = "\n        set_state(state)\n\n        Set the state from a tuple.\n\n        Parameters\n        ----------\n        state : tuple(string, list of 624 ints, int, int, float)\n            The `state` tuple is made up of\n\n            1. the string \'MT19937\'\n            2. a list of 624 integer keys\n            3. an integer pos\n            4. an integer has_gauss\n            5. and a float for the cached_gaussian\n\n        Returns\n        -------\n        out : None\n            Returns \'None\' on success.\n\n        See Also\n        --------\n        get_state\n\n        Notes\n        -----\n        For backwards compatibility, the following form is also accepted\n        although it is missing some information about the cached Gaussian value.\n\n        state = (\'MT19937\', int key[624], int pos)\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_set_state(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_state = 0;
  PyArrayObject *arrayObject_obj;
  int __pyx_v_pos;
  PyObject *__pyx_v_algorithm_name;
  PyObject *__pyx_v_key;
  PyObject *__pyx_v_has_gauss;
  PyObject *__pyx_v_cached_gaussian;
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  int __pyx_2;
  PyObject *__pyx_3 = 0;
  Py_ssize_t __pyx_4;
  PyObject *__pyx_5 = 0;
  PyObject *__pyx_6 = 0;
  double __pyx_7;
  static char *__pyx_argnames[] = {"state",0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O", __pyx_argnames, &__pyx_v_state)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_state);
  arrayObject_obj = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_algorithm_name = Py_None; Py_INCREF(Py_None);
  __pyx_v_key = Py_None; Py_INCREF(Py_None);
  __pyx_v_has_gauss = Py_None; Py_INCREF(Py_None);
  __pyx_v_cached_gaussian = Py_None; Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":647 */
  __pyx_1 = __Pyx_GetItemInt(__pyx_v_state, 0); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 647; goto __pyx_L1;}
  Py_DECREF(__pyx_v_algorithm_name);
  __pyx_v_algorithm_name = __pyx_1;
  __pyx_1 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":648 */
  if (PyObject_Cmp(__pyx_v_algorithm_name, __pyx_n_MT19937, &__pyx_2) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 648; goto __pyx_L1;}
  __pyx_2 = __pyx_2 != 0;
  if (__pyx_2) {
    __pyx_1 = PyTuple_New(1); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 649; goto __pyx_L1;}
    Py_INCREF(__pyx_k70p);
    PyTuple_SET_ITEM(__pyx_1, 0, __pyx_k70p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 649; goto __pyx_L1;}
    Py_DECREF(__pyx_1); __pyx_1 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 649; goto __pyx_L1;}
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":650 */
  __pyx_1 = PySequence_GetSlice(__pyx_v_state, 1, 3); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 650; goto __pyx_L1;}
  __pyx_3 = PyObject_GetIter(__pyx_1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 650; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  __pyx_1 = __Pyx_UnpackItem(__pyx_3); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 650; goto __pyx_L1;}
  Py_DECREF(__pyx_v_key);
  __pyx_v_key = __pyx_1;
  __pyx_1 = 0;
  __pyx_1 = __Pyx_UnpackItem(__pyx_3); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 650; goto __pyx_L1;}
  __pyx_2 = PyInt_AsLong(__pyx_1); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 650; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  __pyx_v_pos = __pyx_2;
  if (__Pyx_EndUnpack(__pyx_3) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 650; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":651 */
  __pyx_4 = PyObject_Length(__pyx_v_state); if (__pyx_4 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 651; goto __pyx_L1;}
  __pyx_2 = (__pyx_4 == 3);
  if (__pyx_2) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":652 */
    __pyx_1 = PyInt_FromLong(0); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 652; goto __pyx_L1;}
    Py_DECREF(__pyx_v_has_gauss);
    __pyx_v_has_gauss = __pyx_1;
    __pyx_1 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":653 */
    __pyx_3 = PyFloat_FromDouble(0.0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 653; goto __pyx_L1;}
    Py_DECREF(__pyx_v_cached_gaussian);
    __pyx_v_cached_gaussian = __pyx_3;
    __pyx_3 = 0;
    goto __pyx_L3;
  }
  /*else*/ {
    __pyx_1 = PySequence_GetSlice(__pyx_v_state, 3, 5); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 655; goto __pyx_L1;}
    __pyx_3 = PyObject_GetIter(__pyx_1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 655; goto __pyx_L1;}
    Py_DECREF(__pyx_1); __pyx_1 = 0;
    __pyx_1 = __Pyx_UnpackItem(__pyx_3); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 655; goto __pyx_L1;}
    Py_DECREF(__pyx_v_has_gauss);
    __pyx_v_has_gauss = __pyx_1;
    __pyx_1 = 0;
    __pyx_1 = __Pyx_UnpackItem(__pyx_3); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 655; goto __pyx_L1;}
    Py_DECREF(__pyx_v_cached_gaussian);
    __pyx_v_cached_gaussian = __pyx_1;
    __pyx_1 = 0;
    if (__Pyx_EndUnpack(__pyx_3) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 655; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
  }
  __pyx_L3:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":656 */
  /*try:*/ {
    __pyx_1 = PyArray_ContiguousFromObject(__pyx_v_key,NPY_ULONG,1,1); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 657; goto __pyx_L4;}
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_1)));
    Py_DECREF(((PyObject *)arrayObject_obj));
    arrayObject_obj = ((PyArrayObject *)__pyx_1);
    Py_DECREF(__pyx_1); __pyx_1 = 0;
  }
  goto __pyx_L5;
  __pyx_L4:;
  Py_XDECREF(__pyx_3); __pyx_3 = 0;
  Py_XDECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":658 */
  __pyx_2 = PyErr_ExceptionMatches(PyExc_TypeError);
  if (__pyx_2) {
    __Pyx_AddTraceback("mtrand.set_state");
    if (__Pyx_GetException(&__pyx_3, &__pyx_1, &__pyx_5) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 658; goto __pyx_L1;}
    __pyx_6 = PyArray_ContiguousFromObject(__pyx_v_key,NPY_LONG,1,1); if (!__pyx_6) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 660; goto __pyx_L1;}
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_6)));
    Py_DECREF(((PyObject *)arrayObject_obj));
    arrayObject_obj = ((PyArrayObject *)__pyx_6);
    Py_DECREF(__pyx_6); __pyx_6 = 0;
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    Py_DECREF(__pyx_1); __pyx_1 = 0;
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    goto __pyx_L5;
  }
  goto __pyx_L1;
  __pyx_L5:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":661 */
  __pyx_2 = ((arrayObject_obj->dimensions[0]) != 624);
  if (__pyx_2) {
    __pyx_6 = PyTuple_New(1); if (!__pyx_6) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 662; goto __pyx_L1;}
    Py_INCREF(__pyx_k71p);
    PyTuple_SET_ITEM(__pyx_6, 0, __pyx_k71p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_6); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 662; goto __pyx_L1;}
    Py_DECREF(__pyx_6); __pyx_6 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 662; goto __pyx_L1;}
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":663 */
  memcpy(((void *)((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state->key),((void *)arrayObject_obj->data),(624 * (sizeof(long))));

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":664 */
  ((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state->pos = __pyx_v_pos;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":665 */
  __pyx_2 = PyInt_AsLong(__pyx_v_has_gauss); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 665; goto __pyx_L1;}
  ((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state->has_gauss = __pyx_2;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":666 */
  __pyx_7 = PyFloat_AsDouble(__pyx_v_cached_gaussian); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 666; goto __pyx_L1;}
  ((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state->gauss = __pyx_7;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_5);
  Py_XDECREF(__pyx_6);
  __Pyx_AddTraceback("mtrand.RandomState.set_state");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject_obj);
  Py_DECREF(__pyx_v_algorithm_name);
  Py_DECREF(__pyx_v_key);
  Py_DECREF(__pyx_v_has_gauss);
  Py_DECREF(__pyx_v_cached_gaussian);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_state);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_11RandomState___getstate__(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyObject *__pyx_f_6mtrand_11RandomState___getstate__(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  PyObject *__pyx_2 = 0;
  static char *__pyx_argnames[] = {0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "", __pyx_argnames)) return 0;
  Py_INCREF(__pyx_v_self);
  __pyx_1 = PyObject_GetAttr(__pyx_v_self, __pyx_n_get_state); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 670; goto __pyx_L1;}
  __pyx_2 = PyObject_CallObject(__pyx_1, 0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 670; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  __pyx_r = __pyx_2;
  __pyx_2 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  Py_XDECREF(__pyx_2);
  __Pyx_AddTraceback("mtrand.RandomState.__getstate__");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_self);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_11RandomState___setstate__(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyObject *__pyx_f_6mtrand_11RandomState___setstate__(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_state = 0;
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  static char *__pyx_argnames[] = {"state",0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O", __pyx_argnames, &__pyx_v_state)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_state);
  __pyx_1 = PyObject_GetAttr(__pyx_v_self, __pyx_n_set_state); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 673; goto __pyx_L1;}
  __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 673; goto __pyx_L1;}
  Py_INCREF(__pyx_v_state);
  PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_state);
  __pyx_3 = PyObject_CallObject(__pyx_1, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 673; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  __Pyx_AddTraceback("mtrand.RandomState.__setstate__");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_state);
  return __pyx_r;
}

static PyObject *__pyx_n_random;
static PyObject *__pyx_n___RandomState_ctor;

static PyObject *__pyx_f_6mtrand_11RandomState___reduce__(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyObject *__pyx_f_6mtrand_11RandomState___reduce__(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  static char *__pyx_argnames[] = {0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "", __pyx_argnames)) return 0;
  Py_INCREF(__pyx_v_self);
  __pyx_1 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 676; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_1, __pyx_n_random); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 676; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  __pyx_1 = PyObject_GetAttr(__pyx_2, __pyx_n___RandomState_ctor); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 676; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = PyTuple_New(0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 676; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_v_self, __pyx_n_get_state); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 676; goto __pyx_L1;}
  __pyx_4 = PyObject_CallObject(__pyx_3, 0); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 676; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyTuple_New(3); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 676; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_3, 0, __pyx_1);
  PyTuple_SET_ITEM(__pyx_3, 1, __pyx_2);
  PyTuple_SET_ITEM(__pyx_3, 2, __pyx_4);
  __pyx_1 = 0;
  __pyx_2 = 0;
  __pyx_4 = 0;
  __pyx_r = __pyx_3;
  __pyx_3 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("mtrand.RandomState.__reduce__");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_self);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_11RandomState_random_sample(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_random_sample[] = "\n        random_sample(size=None)\n\n        Return random floats in the half-open interval [0.0, 1.0).\n\n        Parameters\n        ----------\n        size : shape tuple, optional\n            Defines the shape of the returned array of random floats.\n\n        Returns\n        -------\n        out : ndarray, floats\n            Array of random of floats with shape of `size`.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_random_sample(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_size = 0;
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  static char *__pyx_argnames[] = {"size",0};
  __pyx_v_size = __pyx_k4;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "|O", __pyx_argnames, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_size);
  __pyx_1 = __pyx_f_6mtrand_cont0_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_double,__pyx_v_size); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 696; goto __pyx_L1;}
  __pyx_r = __pyx_1;
  __pyx_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  __Pyx_AddTraceback("mtrand.RandomState.random_sample");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_11RandomState_tomaxint(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_tomaxint[] = "\n        tomaxint(size=None)\n\n        Uniformly sample discrete random integers `x` such that\n        ``0 <= x <= sys.maxint``.\n\n        Parameters\n        ----------\n        size : tuple of ints, int, optional\n            Shape of output.  If the given size is, for example, (m,n,k),\n            m*n*k samples are generated.  If no shape is specified, a single sample\n            is returned.\n\n        Returns\n        -------\n        out : ndarray\n            Drawn samples, with shape `size`.\n\n        See Also\n        --------\n        randint : Uniform sampling over a given half-open interval of integers.\n        random_integers : Uniform sampling over a given closed interval of\n            integers.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_tomaxint(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_size = 0;
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  static char *__pyx_argnames[] = {"size",0};
  __pyx_v_size = __pyx_k5;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "|O", __pyx_argnames, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_size);
  __pyx_1 = __pyx_f_6mtrand_disc0_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_long,__pyx_v_size); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 724; goto __pyx_L1;}
  __pyx_r = __pyx_1;
  __pyx_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  __Pyx_AddTraceback("mtrand.RandomState.tomaxint");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k72p;

static char __pyx_k72[] = "low >= high";

static PyObject *__pyx_f_6mtrand_11RandomState_randint(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_randint[] = "\n        randint(low, high=None, size=None)\n\n        Return random integers x such that low <= x < high.\n\n        If high is None, then 0 <= x < low.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_randint(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_low = 0;
  PyObject *__pyx_v_high = 0;
  PyObject *__pyx_v_size = 0;
  long __pyx_v_lo;
  long __pyx_v_hi;
  long __pyx_v_diff;
  long *__pyx_v_array_data;
  PyArrayObject *arrayObject;
  long __pyx_v_length;
  long __pyx_v_i;
  PyObject *__pyx_r;
  int __pyx_1;
  long __pyx_2;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"low","high","size",0};
  __pyx_v_high = __pyx_k6;
  __pyx_v_size = __pyx_k7;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O|OO", __pyx_argnames, &__pyx_v_low, &__pyx_v_high, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_low);
  Py_INCREF(__pyx_v_high);
  Py_INCREF(__pyx_v_size);
  arrayObject = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":741 */
  __pyx_1 = __pyx_v_high == Py_None;
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":742 */
    __pyx_v_lo = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":743 */
    __pyx_2 = PyInt_AsLong(__pyx_v_low); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 743; goto __pyx_L1;}
    __pyx_v_hi = __pyx_2;
    goto __pyx_L2;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":745 */
    __pyx_2 = PyInt_AsLong(__pyx_v_low); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 745; goto __pyx_L1;}
    __pyx_v_lo = __pyx_2;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":746 */
    __pyx_2 = PyInt_AsLong(__pyx_v_high); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 746; goto __pyx_L1;}
    __pyx_v_hi = __pyx_2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":748 */
  __pyx_v_diff = ((__pyx_v_hi - __pyx_v_lo) - 1);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":749 */
  __pyx_1 = (__pyx_v_diff < 0);
  if (__pyx_1) {
    __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 750; goto __pyx_L1;}
    Py_INCREF(__pyx_k72p);
    PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k72p);
    __pyx_4 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 750; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __Pyx_Raise(__pyx_4, 0, 0);
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 750; goto __pyx_L1;}
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":752 */
  __pyx_1 = __pyx_v_size == Py_None;
  if (__pyx_1) {
    __pyx_3 = PyInt_FromLong((((long)rk_interval(__pyx_v_diff,((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state)) + __pyx_v_lo)); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 753; goto __pyx_L1;}
    __pyx_r = __pyx_3;
    __pyx_3 = 0;
    goto __pyx_L0;
    goto __pyx_L4;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":755 */
    __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 755; goto __pyx_L1;}
    __pyx_3 = PyObject_GetAttr(__pyx_4, __pyx_n_empty); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 755; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    __pyx_4 = PyTuple_New(2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 755; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_v_size);
    Py_INCREF(((PyObject *)(&PyInt_Type)));
    PyTuple_SET_ITEM(__pyx_4, 1, ((PyObject *)(&PyInt_Type)));
    __pyx_5 = PyObject_CallObject(__pyx_3, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 755; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_5)));
    Py_DECREF(((PyObject *)arrayObject));
    arrayObject = ((PyArrayObject *)__pyx_5);
    Py_DECREF(__pyx_5); __pyx_5 = 0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":756 */
    __pyx_v_length = PyArray_SIZE(arrayObject);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":757 */
    __pyx_v_array_data = ((long *)arrayObject->data);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":758 */
    for (__pyx_v_i = 0; __pyx_v_i < __pyx_v_length; ++__pyx_v_i) {
      (__pyx_v_array_data[__pyx_v_i]) = (__pyx_v_lo + ((long)rk_interval(__pyx_v_diff,((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state)));
    }

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":760 */
    Py_INCREF(((PyObject *)arrayObject));
    __pyx_r = ((PyObject *)arrayObject);
    goto __pyx_L0;
  }
  __pyx_L4:;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.randint");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_low);
  Py_DECREF(__pyx_v_high);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_11RandomState_bytes(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_bytes[] = "\n        bytes(length)\n\n        Return random bytes.\n\n        Parameters\n        ----------\n        length : int\n            Number of random bytes.\n\n        Returns\n        -------\n        out : str\n            String of length `N`.\n\n        Examples\n        --------\n        >>> np.random.bytes(10)\n        \' eh\\x85\\x022SZ\\xbf\\xa4\' #random\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_bytes(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  unsigned int __pyx_v_length;
  void *__pyx_v_bytes;
  PyObject *__pyx_v_bytestring;
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  static char *__pyx_argnames[] = {"length",0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "I", __pyx_argnames, &__pyx_v_length)) return 0;
  Py_INCREF(__pyx_v_self);
  __pyx_v_bytestring = Py_None; Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":785 */
  __pyx_1 = PyString_FromStringAndSize(NULL,__pyx_v_length); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 785; goto __pyx_L1;}
  Py_DECREF(__pyx_v_bytestring);
  __pyx_v_bytestring = __pyx_1;
  __pyx_1 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":786 */
  __pyx_v_bytes = PyString_AS_STRING(__pyx_v_bytestring);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":787 */
  rk_fill(__pyx_v_bytes,__pyx_v_length,((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":788 */
  Py_INCREF(__pyx_v_bytestring);
  __pyx_r = __pyx_v_bytestring;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  __Pyx_AddTraceback("mtrand.RandomState.bytes");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_bytestring);
  Py_DECREF(__pyx_v_self);
  return __pyx_r;
}

static PyObject *__pyx_n_subtract;

static PyObject *__pyx_f_6mtrand_11RandomState_uniform(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_uniform[] = "\n        uniform(low=0.0, high=1.0, size=1)\n\n        Draw samples from a uniform distribution.\n\n        Samples are uniformly distributed over the half-open interval\n        ``[low, high)`` (includes low, but excludes high).  In other words,\n        any value within the given interval is equally likely to be drawn\n        by `uniform`.\n\n        Parameters\n        ----------\n        low : float, optional\n            Lower boundary of the output interval.  All values generated will be\n            greater than or equal to low.  The default value is 0.\n        high : float\n            Upper boundary of the output interval.  All values generated will be\n            less than high.  The default value is 1.0.\n        size : tuple of ints, int, optional\n            Shape of output.  If the given size is, for example, (m,n,k),\n            m*n*k samples are generated.  If no shape is specified, a single sample\n            is returned.\n\n        Returns\n        -------\n        out : ndarray\n            Drawn samples, with shape `size`.\n\n        See Also\n        --------\n        randint : Discrete uniform distribution, yielding integers.\n        random_integers : Discrete uniform distribution over the closed interval\n                          ``[low, high]``.\n        random_sample : Floats uniformly distributed over ``[0, 1)``.\n        random : Alias for `random_sample`.\n        rand : Convenience function that accepts dimensions as input, e.g.,\n               ``rand(2,2)`` would generate a 2-by-2 array of floats, uniformly\n               distributed over ``[0, 1)``.\n\n        Notes\n        -----\n        The probability density function of the uniform distribution is\n\n        .. math:: p(x) = \\frac{1}{b - a}\n\n        anywhere within the interval ``[a, b)``, and zero elsewhere.\n\n        Examples\n        --------\n        Draw samples from the distribution:\n\n        >>> s = np.random.uniform(-1,0,1000)\n\n        All values are within the given interval:\n\n        >>> np.all(s >= -1)\n        True\n\n        >>> np.all(s < 0)\n        True\n\n        Display the histogram of the samples, along with the\n        probability density function:\n\n        >>> import matplotlib.pyplot as plt\n        >>> count, bins, ignored = plt.hist(s, 15, normed=True)\n        >>> plt.plot(bins, np.ones_like(bins), linewidth=2, color=\'r\')\n        >>> plt.show()\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_uniform(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_low = 0;
  PyObject *__pyx_v_high = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_olow;
  PyArrayObject *__pyx_v_ohigh;
  PyArrayObject *__pyx_v_odiff;
  double __pyx_v_flow;
  double __pyx_v_fhigh;
  PyObject *__pyx_v_temp;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  static char *__pyx_argnames[] = {"low","high","size",0};
  __pyx_v_low = __pyx_k8;
  __pyx_v_high = __pyx_k9;
  __pyx_v_size = __pyx_k10;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "|OOO", __pyx_argnames, &__pyx_v_low, &__pyx_v_high, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_low);
  Py_INCREF(__pyx_v_high);
  Py_INCREF(__pyx_v_size);
  __pyx_v_olow = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_ohigh = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_odiff = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_temp = Py_None; Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":865 */
  __pyx_v_flow = PyFloat_AsDouble(__pyx_v_low);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":866 */
  __pyx_v_fhigh = PyFloat_AsDouble(__pyx_v_high);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":867 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {
    __pyx_2 = __pyx_f_6mtrand_cont2_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_uniform,__pyx_v_size,__pyx_v_flow,(__pyx_v_fhigh - __pyx_v_flow)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 868; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":869 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":870 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_low,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 870; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
  Py_DECREF(((PyObject *)__pyx_v_olow));
  __pyx_v_olow = ((PyArrayObject *)__pyx_2);
  Py_DECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":871 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_high,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 871; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
  Py_DECREF(((PyObject *)__pyx_v_ohigh));
  __pyx_v_ohigh = ((PyArrayObject *)__pyx_2);
  Py_DECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":872 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 872; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_subtract); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 872; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 872; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_ohigh));
  PyTuple_SET_ITEM(__pyx_2, 0, ((PyObject *)__pyx_v_ohigh));
  Py_INCREF(((PyObject *)__pyx_v_olow));
  PyTuple_SET_ITEM(__pyx_2, 1, ((PyObject *)__pyx_v_olow));
  __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 872; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_v_temp);
  __pyx_v_temp = __pyx_4;
  __pyx_4 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":873 */
  Py_INCREF(__pyx_v_temp);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":875 */
  __pyx_3 = PyArray_EnsureArray(__pyx_v_temp); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 875; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_odiff));
  __pyx_v_odiff = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":876 */
  __pyx_2 = __pyx_f_6mtrand_cont2_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_uniform,__pyx_v_size,__pyx_v_olow,__pyx_v_odiff); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 876; goto __pyx_L1;}
  __pyx_r = __pyx_2;
  __pyx_2 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("mtrand.RandomState.uniform");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_olow);
  Py_DECREF(__pyx_v_ohigh);
  Py_DECREF(__pyx_v_odiff);
  Py_DECREF(__pyx_v_temp);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_low);
  Py_DECREF(__pyx_v_high);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_n_size;


static PyObject *__pyx_f_6mtrand_11RandomState_rand(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_rand[] = "\n        rand(d0, d1, ..., dn)\n\n        Random values in a given shape.\n\n        Create an array of the given shape and propagate it with\n        random samples from a uniform distribution\n        over ``[0, 1)``.\n\n        Parameters\n        ----------\n        d0, d1, ..., dn : int\n            Shape of the output.\n\n        Returns\n        -------\n        out : ndarray, shape ``(d0, d1, ..., dn)``\n            Random values.\n\n        See Also\n        --------\n        random\n\n        Notes\n        -----\n        This is a convenience function. If you want an interface that\n        takes a shape-tuple as the first argument, refer to\n        `random`.\n\n        Examples\n        --------\n        >>> np.random.rand(3,2)\n        array([[ 0.14022471,  0.96360618],  #random\n               [ 0.37601032,  0.25528411],  #random\n               [ 0.49313049,  0.94909878]]) #random\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_rand(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_args = 0;
  PyObject *__pyx_r;
  Py_ssize_t __pyx_1;
  int __pyx_2;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  PyObject *__pyx_6 = 0;
  static char *__pyx_argnames[] = {0};
  if (__Pyx_GetStarArgs(&__pyx_args, &__pyx_kwds, __pyx_argnames, 0, &__pyx_v_args, 0, 0) < 0) return 0;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "", __pyx_argnames)) {
    Py_XDECREF(__pyx_args);
    Py_XDECREF(__pyx_kwds);
    Py_XDECREF(__pyx_v_args);
    return 0;
  }
  Py_INCREF(__pyx_v_self);
  __pyx_1 = PyObject_Length(__pyx_v_args); if (__pyx_1 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 916; goto __pyx_L1;}
  __pyx_2 = (__pyx_1 == 0);
  if (__pyx_2) {
    __pyx_3 = PyObject_GetAttr(__pyx_v_self, __pyx_n_random_sample); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 917; goto __pyx_L1;}
    __pyx_4 = PyObject_CallObject(__pyx_3, 0); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 917; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __pyx_r = __pyx_4;
    __pyx_4 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  /*else*/ {
    __pyx_3 = PyObject_GetAttr(__pyx_v_self, __pyx_n_random_sample); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 919; goto __pyx_L1;}
    __pyx_4 = PyTuple_New(0); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 919; goto __pyx_L1;}
    __pyx_5 = PyDict_New(); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 919; goto __pyx_L1;}
    if (PyDict_SetItem(__pyx_5, __pyx_n_size, __pyx_v_args) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 919; goto __pyx_L1;}
    __pyx_6 = PyEval_CallObjectWithKeywords(__pyx_3, __pyx_4, __pyx_5); if (!__pyx_6) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 919; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    __pyx_r = __pyx_6;
    __pyx_6 = 0;
    goto __pyx_L0;
  }
  __pyx_L2:;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  Py_XDECREF(__pyx_6);
  __Pyx_AddTraceback("mtrand.RandomState.rand");
  __pyx_r = 0;
  __pyx_L0:;
  Py_XDECREF(__pyx_v_args);
  Py_DECREF(__pyx_v_self);
  Py_XDECREF(__pyx_args);
  Py_XDECREF(__pyx_kwds);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_11RandomState_randn(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_randn[] = "\n        randn(d0, d1, ..., dn)\n\n        Returns zero-mean, unit-variance Gaussian random numbers in an\n        array of shape (d0, d1, ..., dn).\n\n        Note:  This is a convenience function. If you want an\n                    interface that takes a tuple as the first argument\n                    use numpy.random.standard_normal(shape_tuple).\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_randn(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_args = 0;
  PyObject *__pyx_r;
  Py_ssize_t __pyx_1;
  int __pyx_2;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {0};
  if (__Pyx_GetStarArgs(&__pyx_args, &__pyx_kwds, __pyx_argnames, 0, &__pyx_v_args, 0, 0) < 0) return 0;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "", __pyx_argnames)) {
    Py_XDECREF(__pyx_args);
    Py_XDECREF(__pyx_kwds);
    Py_XDECREF(__pyx_v_args);
    return 0;
  }
  Py_INCREF(__pyx_v_self);
  __pyx_1 = PyObject_Length(__pyx_v_args); if (__pyx_1 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 933; goto __pyx_L1;}
  __pyx_2 = (__pyx_1 == 0);
  if (__pyx_2) {
    __pyx_3 = PyObject_GetAttr(__pyx_v_self, __pyx_n_standard_normal); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 934; goto __pyx_L1;}
    __pyx_4 = PyObject_CallObject(__pyx_3, 0); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 934; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __pyx_r = __pyx_4;
    __pyx_4 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  /*else*/ {
    __pyx_3 = PyObject_GetAttr(__pyx_v_self, __pyx_n_standard_normal); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 936; goto __pyx_L1;}
    __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 936; goto __pyx_L1;}
    Py_INCREF(__pyx_v_args);
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_v_args);
    __pyx_5 = PyObject_CallObject(__pyx_3, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 936; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    __pyx_r = __pyx_5;
    __pyx_5 = 0;
    goto __pyx_L0;
  }
  __pyx_L2:;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.randn");
  __pyx_r = 0;
  __pyx_L0:;
  Py_XDECREF(__pyx_v_args);
  Py_DECREF(__pyx_v_self);
  Py_XDECREF(__pyx_args);
  Py_XDECREF(__pyx_kwds);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_11RandomState_random_integers(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_random_integers[] = "\n        random_integers(low, high=None, size=None)\n\n        Return random integers x such that low <= x <= high.\n\n        If high is None, then 1 <= x <= low.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_random_integers(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_low = 0;
  PyObject *__pyx_v_high = 0;
  PyObject *__pyx_v_size = 0;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  static char *__pyx_argnames[] = {"low","high","size",0};
  __pyx_v_high = __pyx_k11;
  __pyx_v_size = __pyx_k12;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O|OO", __pyx_argnames, &__pyx_v_low, &__pyx_v_high, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_low);
  Py_INCREF(__pyx_v_high);
  Py_INCREF(__pyx_v_size);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":947 */
  __pyx_1 = __pyx_v_high == Py_None;
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":948 */
    Py_INCREF(__pyx_v_low);
    Py_DECREF(__pyx_v_high);
    __pyx_v_high = __pyx_v_low;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":949 */
    __pyx_2 = PyInt_FromLong(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 949; goto __pyx_L1;}
    Py_DECREF(__pyx_v_low);
    __pyx_v_low = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":950 */
  __pyx_2 = PyObject_GetAttr(__pyx_v_self, __pyx_n_randint); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 950; goto __pyx_L1;}
  __pyx_3 = PyInt_FromLong(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 950; goto __pyx_L1;}
  __pyx_4 = PyNumber_Add(__pyx_v_high, __pyx_3); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 950; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyTuple_New(3); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 950; goto __pyx_L1;}
  Py_INCREF(__pyx_v_low);
  PyTuple_SET_ITEM(__pyx_3, 0, __pyx_v_low);
  PyTuple_SET_ITEM(__pyx_3, 1, __pyx_4);
  Py_INCREF(__pyx_v_size);
  PyTuple_SET_ITEM(__pyx_3, 2, __pyx_v_size);
  __pyx_4 = 0;
  __pyx_4 = PyObject_CallObject(__pyx_2, __pyx_3); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 950; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_r = __pyx_4;
  __pyx_4 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("mtrand.RandomState.random_integers");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_low);
  Py_DECREF(__pyx_v_high);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_11RandomState_standard_normal(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_standard_normal[] = "\n        standard_normal(size=None)\n\n        Returns samples from a Standard Normal distribution (mean=0, stdev=1).\n\n        Parameters\n        ----------\n        size : int, shape tuple, optional\n            Returns the number of samples required to satisfy the `size` parameter.\n            If not given or \'None\' indicates to return one sample.\n\n        Returns\n        -------\n        out : float, ndarray\n            Samples the Standard Normal distribution with a shape satisfying the\n            `size` parameter.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_standard_normal(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_size = 0;
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  static char *__pyx_argnames[] = {"size",0};
  __pyx_v_size = __pyx_k13;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "|O", __pyx_argnames, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_size);
  __pyx_1 = __pyx_f_6mtrand_cont0_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_gauss,__pyx_v_size); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 972; goto __pyx_L1;}
  __pyx_r = __pyx_1;
  __pyx_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  __Pyx_AddTraceback("mtrand.RandomState.standard_normal");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_n_any;
static PyObject *__pyx_n_less_equal;

static PyObject *__pyx_k74p;
static PyObject *__pyx_k75p;

static char __pyx_k74[] = "scale <= 0";
static char __pyx_k75[] = "scale <= 0";

static PyObject *__pyx_f_6mtrand_11RandomState_normal(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_normal[] = "\n        normal(loc=0.0, scale=1.0, size=None)\n\n        Draw random samples from a normal (Gaussian) distribution.\n\n        The probability density function of the normal distribution, first\n        derived by De Moivre and 200 years later by both Gauss and Laplace\n        independently [2]_, is often called the bell curve because of\n        its characteristic shape (see the example below).\n\n        The normal distributions occurs often in nature.  For example, it\n        describes the commonly occurring distribution of samples influenced\n        by a large number of tiny, random disturbances, each with its own\n        unique distribution [2]_.\n\n        Parameters\n        ----------\n        loc : float\n            Mean (\"centre\") of the distribution.\n        scale : float\n            Standard deviation (spread or \"width\") of the distribution.\n        size : tuple of ints\n            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then\n            ``m * n * k`` samples are drawn.\n\n        See Also\n        --------\n        scipy.stats.distributions.norm : probability density function,\n            distribution or cumulative density function, etc.\n\n        Notes\n        -----\n        The probability density for the Gaussian distribution is\n\n        .. math:: p(x) = \\frac{1}{\\sqrt{ 2 \\pi \\sigma^2 }}\n                         e^{ - \\frac{ (x - \\mu)^2 } {2 \\sigma^2} },\n\n        where :math:`\\mu` is the mean and :math:`\\sigma` the standard deviation.\n        The square of the standard deviation, :math:`\\sigma^2`, is called the\n        variance.\n\n        The function has its peak at the mean, and its \"spread\" increases with\n        the standard deviation (the function reaches 0.607 times its maximum at\n        :math:`x + \\sigma` and :math:`x - \\sigma` [2]_).  This implies that\n        `numpy.random.normal` is more likely to return samples lying close to the\n        mean, rather than those far away.\n\n        References\n        ----------\n        .. [1] Wikipedia, \"Normal distribution\",\n               http://en.wikipedia.org/wiki/Normal_distribution\n        .. [2] P. R. Peebles Jr., \"Central Limit Theorem\" in \"Probability, Random\n               Variables and Random Signal Principles\", 4th ed., 2001,\n               pp. 51, 51, 125.\n\n        Examples\n        --------\n        Draw samples from the distribution:\n\n        >>> mu, sigma = 0, 0.1 # mean and standard deviation\n        >>> s = np.random.normal(mu, sigma, 1000)\n\n        Verify the mean and the variance:\n\n        >>> abs(mu - np.mean(s)) < 0.01\n        True\n\n        >>> abs(sigma - np.std(s, ddof=1)) < 0.01\n        True\n\n        Display the histogram of the samples, along with\n        the probability density function:\n\n        >>> import matplotlib.pyplot as plt\n        >>> count, bins, ignored = plt.hist(s, 30, normed=True)\n        >>> plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *\n        ...                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),\n        ...          linewidth=2, color=\'r\')\n        >>> plt.show()\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_normal(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_loc = 0;
  PyObject *__pyx_v_scale = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_oloc;
  PyArrayObject *__pyx_v_oscale;
  double __pyx_v_floc;
  double __pyx_v_fscale;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"loc","scale","size",0};
  __pyx_v_loc = __pyx_k14;
  __pyx_v_scale = __pyx_k15;
  __pyx_v_size = __pyx_k16;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "|OOO", __pyx_argnames, &__pyx_v_loc, &__pyx_v_scale, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_loc);
  Py_INCREF(__pyx_v_scale);
  Py_INCREF(__pyx_v_size);
  __pyx_v_oloc = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_oscale = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1059 */
  __pyx_v_floc = PyFloat_AsDouble(__pyx_v_loc);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1060 */
  __pyx_v_fscale = PyFloat_AsDouble(__pyx_v_scale);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1061 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1062 */
    __pyx_1 = (__pyx_v_fscale <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1063; goto __pyx_L1;}
      Py_INCREF(__pyx_k74p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k74p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1063; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1063; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1064 */
    __pyx_2 = __pyx_f_6mtrand_cont2_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_normal,__pyx_v_size,__pyx_v_floc,__pyx_v_fscale); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1064; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1066 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1068 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_loc,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1068; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_oloc));
  __pyx_v_oloc = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1069 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_scale,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1069; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
  Py_DECREF(((PyObject *)__pyx_v_oscale));
  __pyx_v_oscale = ((PyArrayObject *)__pyx_2);
  Py_DECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1070 */
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1070; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_any); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1070; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1070; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_3, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1070; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyInt_FromLong(0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1070; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1070; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_oscale));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_oscale));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_3);
  __pyx_3 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1070; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1070; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_3);
  __pyx_3 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1070; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1070; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1071; goto __pyx_L1;}
    Py_INCREF(__pyx_k75p);
    PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k75p);
    __pyx_2 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1071; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __Pyx_Raise(__pyx_2, 0, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1071; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1072 */
  __pyx_4 = __pyx_f_6mtrand_cont2_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_normal,__pyx_v_size,__pyx_v_oloc,__pyx_v_oscale); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1072; goto __pyx_L1;}
  __pyx_r = __pyx_4;
  __pyx_4 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.normal");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_oloc);
  Py_DECREF(__pyx_v_oscale);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_loc);
  Py_DECREF(__pyx_v_scale);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k76p;
static PyObject *__pyx_k77p;
static PyObject *__pyx_k78p;
static PyObject *__pyx_k79p;

static char __pyx_k76[] = "a <= 0";
static char __pyx_k77[] = "b <= 0";
static char __pyx_k78[] = "a <= 0";
static char __pyx_k79[] = "b <= 0";

static PyObject *__pyx_f_6mtrand_11RandomState_beta(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_beta[] = "\n        beta(a, b, size=None)\n\n        The Beta distribution over ``[0, 1]``.\n\n        The Beta distribution is a special case of the Dirichlet distribution,\n        and is related to the Gamma distribution.  It has the probability\n        distribution function\n\n        .. math:: f(x; a,b) = \\frac{1}{B(\\alpha, \\beta)} x^{\\alpha - 1}\n                                                         (1 - x)^{\\beta - 1},\n\n        where the normalisation, B, is the beta function,\n\n        .. math:: B(\\alpha, \\beta) = \\int_0^1 t^{\\alpha - 1}\n                                     (1 - t)^{\\beta - 1} dt.\n\n        It is often seen in Bayesian inference and order statistics.\n\n        Parameters\n        ----------\n        a : float\n            Alpha, non-negative.\n        b : float\n            Beta, non-negative.\n        size : tuple of ints, optional\n            The number of samples to draw.  The ouput is packed according to\n            the size given.\n\n        Returns\n        -------\n        out : ndarray\n            Array of the given shape, containing values drawn from a\n            Beta distribution.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_beta(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_a = 0;
  PyObject *__pyx_v_b = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_oa;
  PyArrayObject *__pyx_v_ob;
  double __pyx_v_fa;
  double __pyx_v_fb;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"a","b","size",0};
  __pyx_v_size = __pyx_k17;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "OO|O", __pyx_argnames, &__pyx_v_a, &__pyx_v_b, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_a);
  Py_INCREF(__pyx_v_b);
  Py_INCREF(__pyx_v_size);
  __pyx_v_oa = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_ob = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1114 */
  __pyx_v_fa = PyFloat_AsDouble(__pyx_v_a);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1115 */
  __pyx_v_fb = PyFloat_AsDouble(__pyx_v_b);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1116 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1117 */
    __pyx_1 = (__pyx_v_fa <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1118; goto __pyx_L1;}
      Py_INCREF(__pyx_k76p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k76p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1118; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1118; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1119 */
    __pyx_1 = (__pyx_v_fb <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1120; goto __pyx_L1;}
      Py_INCREF(__pyx_k77p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k77p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1120; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1120; goto __pyx_L1;}
      goto __pyx_L4;
    }
    __pyx_L4:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1121 */
    __pyx_2 = __pyx_f_6mtrand_cont2_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_beta,__pyx_v_size,__pyx_v_fa,__pyx_v_fb); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1121; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1123 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1125 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_a,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1125; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_oa));
  __pyx_v_oa = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1126 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_b,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1126; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
  Py_DECREF(((PyObject *)__pyx_v_ob));
  __pyx_v_ob = ((PyArrayObject *)__pyx_2);
  Py_DECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1127 */
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1127; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_any); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1127; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1127; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_3, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1127; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyInt_FromLong(0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1127; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1127; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_oa));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_oa));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_3);
  __pyx_3 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1127; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1127; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_3);
  __pyx_3 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1127; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1127; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1128; goto __pyx_L1;}
    Py_INCREF(__pyx_k78p);
    PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k78p);
    __pyx_2 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1128; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __Pyx_Raise(__pyx_2, 0, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1128; goto __pyx_L1;}
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1129 */
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1129; goto __pyx_L1;}
  __pyx_5 = PyObject_GetAttr(__pyx_4, __pyx_n_any); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1129; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1129; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_less_equal); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1129; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_4 = PyInt_FromLong(0); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1129; goto __pyx_L1;}
  __pyx_3 = PyTuple_New(2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1129; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_ob));
  PyTuple_SET_ITEM(__pyx_3, 0, ((PyObject *)__pyx_v_ob));
  PyTuple_SET_ITEM(__pyx_3, 1, __pyx_4);
  __pyx_4 = 0;
  __pyx_4 = PyObject_CallObject(__pyx_2, __pyx_3); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1129; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1129; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_2, 0, __pyx_4);
  __pyx_4 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_5, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1129; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_3); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1129; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  if (__pyx_1) {
    __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1130; goto __pyx_L1;}
    Py_INCREF(__pyx_k79p);
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_k79p);
    __pyx_5 = PyObject_CallObject(PyExc_ValueError, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1130; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    __Pyx_Raise(__pyx_5, 0, 0);
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1130; goto __pyx_L1;}
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1131 */
  __pyx_2 = __pyx_f_6mtrand_cont2_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_beta,__pyx_v_size,__pyx_v_oa,__pyx_v_ob); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1131; goto __pyx_L1;}
  __pyx_r = __pyx_2;
  __pyx_2 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.beta");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_oa);
  Py_DECREF(__pyx_v_ob);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_a);
  Py_DECREF(__pyx_v_b);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k80p;
static PyObject *__pyx_k81p;

static char __pyx_k80[] = "scale <= 0";
static char __pyx_k81[] = "scale <= 0";

static PyObject *__pyx_f_6mtrand_11RandomState_exponential(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_exponential[] = "\n        exponential(scale=1.0, size=None)\n\n        Exponential distribution.\n\n        Its probability density function is\n\n        .. math:: f(x; \\frac{1}{\\beta}) = \\frac{1}{\\beta} \\exp(-\\frac{x}{\\beta}),\n\n        for ``x > 0`` and 0 elsewhere. :math:`\\beta` is the scale parameter,\n        which is the inverse of the rate parameter :math:`\\lambda = 1/\\beta`.\n        The rate parameter is an alternative, widely used parameterization\n        of the exponential distribution [3]_.\n\n        The exponential distribution is a continuous analogue of the\n        geometric distribution.  It describes many common situations, such as\n        the size of raindrops measured over many rainstorms [1]_, or the time\n        between page requests to Wikipedia [2]_.\n\n        Parameters\n        ----------\n        scale : float\n            The scale parameter, :math:`\\beta = 1/\\lambda`.\n        size : tuple of ints\n            Number of samples to draw.  The output is shaped\n            according to `size`.\n\n        References\n        ----------\n        .. [1] Peyton Z. Peebles Jr., \"Probability, Random Variables and\n               Random Signal Principles\", 4th ed, 2001, p. 57.\n        .. [2] \"Poisson Process\", Wikipedia,\n               http://en.wikipedia.org/wiki/Poisson_process\n        .. [3] \"Exponential Distribution, Wikipedia,\n               http://en.wikipedia.org/wiki/Exponential_distribution\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_exponential(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_scale = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_oscale;
  double __pyx_v_fscale;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"scale","size",0};
  __pyx_v_scale = __pyx_k18;
  __pyx_v_size = __pyx_k19;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "|OO", __pyx_argnames, &__pyx_v_scale, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_scale);
  Py_INCREF(__pyx_v_size);
  __pyx_v_oscale = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1174 */
  __pyx_v_fscale = PyFloat_AsDouble(__pyx_v_scale);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1175 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1176 */
    __pyx_1 = (__pyx_v_fscale <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1177; goto __pyx_L1;}
      Py_INCREF(__pyx_k80p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k80p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1177; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1177; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1178 */
    __pyx_2 = __pyx_f_6mtrand_cont1_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_exponential,__pyx_v_size,__pyx_v_fscale); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1178; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1180 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1182 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_scale,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1182; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_oscale));
  __pyx_v_oscale = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1183 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1183; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_any); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1183; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1183; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1183; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = PyFloat_FromDouble(0.0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1183; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1183; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_oscale));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_oscale));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_2);
  __pyx_2 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1183; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1183; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_2);
  __pyx_2 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_3, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1183; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1183; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1184; goto __pyx_L1;}
    Py_INCREF(__pyx_k81p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k81p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1184; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1184; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1185 */
  __pyx_4 = __pyx_f_6mtrand_cont1_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_exponential,__pyx_v_size,__pyx_v_oscale); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1185; goto __pyx_L1;}
  __pyx_r = __pyx_4;
  __pyx_4 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.exponential");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_oscale);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_scale);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_11RandomState_standard_exponential(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_standard_exponential[] = "\n        standard_exponential(size=None)\n\n        Standard exponential distribution (scale=1).\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_standard_exponential(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_size = 0;
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  static char *__pyx_argnames[] = {"size",0};
  __pyx_v_size = __pyx_k20;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "|O", __pyx_argnames, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_size);
  __pyx_1 = __pyx_f_6mtrand_cont0_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_standard_exponential,__pyx_v_size); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1194; goto __pyx_L1;}
  __pyx_r = __pyx_1;
  __pyx_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  __Pyx_AddTraceback("mtrand.RandomState.standard_exponential");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k82p;
static PyObject *__pyx_k83p;

static char __pyx_k82[] = "shape <= 0";
static char __pyx_k83[] = "shape <= 0";

static PyObject *__pyx_f_6mtrand_11RandomState_standard_gamma(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_standard_gamma[] = "\n        standard_gamma(shape, size=None)\n\n        Standard Gamma distribution.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_standard_gamma(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_shape = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_oshape;
  double __pyx_v_fshape;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"shape","size",0};
  __pyx_v_size = __pyx_k21;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O|O", __pyx_argnames, &__pyx_v_shape, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_shape);
  Py_INCREF(__pyx_v_size);
  __pyx_v_oshape = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1206 */
  __pyx_v_fshape = PyFloat_AsDouble(__pyx_v_shape);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1207 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1208 */
    __pyx_1 = (__pyx_v_fshape <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1209; goto __pyx_L1;}
      Py_INCREF(__pyx_k82p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k82p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1209; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1209; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1210 */
    __pyx_2 = __pyx_f_6mtrand_cont1_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_standard_gamma,__pyx_v_size,__pyx_v_fshape); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1210; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1212 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1213 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_shape,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1213; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_oshape));
  __pyx_v_oshape = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1214 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1214; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_any); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1214; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1214; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1214; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = PyFloat_FromDouble(0.0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1214; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1214; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_oshape));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_oshape));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_2);
  __pyx_2 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1214; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1214; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_2);
  __pyx_2 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_3, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1214; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1214; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1215; goto __pyx_L1;}
    Py_INCREF(__pyx_k83p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k83p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1215; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1215; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1216 */
  __pyx_4 = __pyx_f_6mtrand_cont1_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_standard_gamma,__pyx_v_size,__pyx_v_oshape); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1216; goto __pyx_L1;}
  __pyx_r = __pyx_4;
  __pyx_4 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.standard_gamma");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_oshape);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_shape);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k84p;
static PyObject *__pyx_k85p;
static PyObject *__pyx_k86p;
static PyObject *__pyx_k87p;

static char __pyx_k84[] = "shape <= 0";
static char __pyx_k85[] = "scale <= 0";
static char __pyx_k86[] = "shape <= 0";
static char __pyx_k87[] = "scale <= 0";

static PyObject *__pyx_f_6mtrand_11RandomState_gamma(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_gamma[] = "\n        gamma(shape, scale=1.0, size=None)\n\n        Draw samples from a Gamma distribution.\n\n        Samples are drawn from a Gamma distribution with specified parameters,\n        `shape` (sometimes designated \"k\") and `scale` (sometimes designated\n        \"theta\"), where both parameters are > 0.\n\n        Parameters\n        ----------\n        shape : scalar > 0\n            The shape of the gamma distribution.\n        scale : scalar > 0, optional\n            The scale of the gamma distribution.  Default is equal to 1.\n        size : shape_tuple, optional\n            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then\n            ``m * n * k`` samples are drawn.\n\n        Returns\n        -------\n        out : ndarray, float\n            Returns one sample unless `size` parameter is specified.\n\n        See Also\n        --------\n        scipy.stats.distributions.gamma : probability density function,\n            distribution or cumulative density function, etc.\n\n        Notes\n        -----\n        The probability density for the Gamma distribution is\n\n        .. math:: p(x) = x^{k-1}\\frac{e^{-x/\\theta}}{\\theta^k\\Gamma(k)},\n\n        where :math:`k` is the shape and :math:`\\theta` the scale,\n        and :math:`\\Gamma` is the Gamma function.\n\n        The Gamma distribution is often used to model the times to failure of\n        electronic components, and arises naturally in processes for which the\n        waiting times between Poisson distributed events are relevant.\n\n        References\n        ----------\n        .. [1] Weisstein, Eric W. \"Gamma Distribution.\" From MathWorld--A\n               Wolfram Web Resource.\n               http://mathworld.wolfram.com/GammaDistribution.html\n        .. [2] Wikipedia, \"Gamma-distribution\",\n               http://en.wikipedia.org/wiki/Gamma-distribution\n\n        Examples\n        --------\n        Draw samples from the distribution:\n\n        >>> shape, scale = 2., 2. # mean and dispersion\n        >>> s = np.random.gamma(shape, scale, 1000)\n\n        Display the histogram of the samples, along with\n        the probability density function:\n\n        >>> import matplotlib.pyplot as plt\n        >>> import scipy.special as sps\n        >>> count, bins, ignored = plt.hist(s, 50, normed=True)\n        >>> y = bins**(shape-1)*((exp(-bins/scale))/\\\n            (sps.gamma(shape)*scale**shape))\n        >>> plt.plot(bins, y, linewidth=2, color=\'r\')\n        >>> plt.show()\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_gamma(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_shape = 0;
  PyObject *__pyx_v_scale = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_oshape;
  PyArrayObject *__pyx_v_oscale;
  double __pyx_v_fshape;
  double __pyx_v_fscale;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"shape","scale","size",0};
  __pyx_v_scale = __pyx_k22;
  __pyx_v_size = __pyx_k23;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O|OO", __pyx_argnames, &__pyx_v_shape, &__pyx_v_scale, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_shape);
  Py_INCREF(__pyx_v_scale);
  Py_INCREF(__pyx_v_size);
  __pyx_v_oshape = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_oscale = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1291 */
  __pyx_v_fshape = PyFloat_AsDouble(__pyx_v_shape);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1292 */
  __pyx_v_fscale = PyFloat_AsDouble(__pyx_v_scale);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1293 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1294 */
    __pyx_1 = (__pyx_v_fshape <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1295; goto __pyx_L1;}
      Py_INCREF(__pyx_k84p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k84p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1295; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1295; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1296 */
    __pyx_1 = (__pyx_v_fscale <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1297; goto __pyx_L1;}
      Py_INCREF(__pyx_k85p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k85p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1297; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1297; goto __pyx_L1;}
      goto __pyx_L4;
    }
    __pyx_L4:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1298 */
    __pyx_2 = __pyx_f_6mtrand_cont2_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_gamma,__pyx_v_size,__pyx_v_fshape,__pyx_v_fscale); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1298; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1300 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1301 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_shape,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1301; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_oshape));
  __pyx_v_oshape = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1302 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_scale,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1302; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
  Py_DECREF(((PyObject *)__pyx_v_oscale));
  __pyx_v_oscale = ((PyArrayObject *)__pyx_2);
  Py_DECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1303 */
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1303; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_any); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1303; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1303; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_3, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1303; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyFloat_FromDouble(0.0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1303; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1303; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_oshape));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_oshape));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_3);
  __pyx_3 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1303; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1303; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_3);
  __pyx_3 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1303; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1303; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1304; goto __pyx_L1;}
    Py_INCREF(__pyx_k86p);
    PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k86p);
    __pyx_2 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1304; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __Pyx_Raise(__pyx_2, 0, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1304; goto __pyx_L1;}
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1305 */
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1305; goto __pyx_L1;}
  __pyx_5 = PyObject_GetAttr(__pyx_4, __pyx_n_any); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1305; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1305; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_less_equal); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1305; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_4 = PyFloat_FromDouble(0.0); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1305; goto __pyx_L1;}
  __pyx_3 = PyTuple_New(2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1305; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_oscale));
  PyTuple_SET_ITEM(__pyx_3, 0, ((PyObject *)__pyx_v_oscale));
  PyTuple_SET_ITEM(__pyx_3, 1, __pyx_4);
  __pyx_4 = 0;
  __pyx_4 = PyObject_CallObject(__pyx_2, __pyx_3); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1305; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1305; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_2, 0, __pyx_4);
  __pyx_4 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_5, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1305; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_3); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1305; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  if (__pyx_1) {
    __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1306; goto __pyx_L1;}
    Py_INCREF(__pyx_k87p);
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_k87p);
    __pyx_5 = PyObject_CallObject(PyExc_ValueError, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1306; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    __Pyx_Raise(__pyx_5, 0, 0);
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1306; goto __pyx_L1;}
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1307 */
  __pyx_2 = __pyx_f_6mtrand_cont2_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_gamma,__pyx_v_size,__pyx_v_oshape,__pyx_v_oscale); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1307; goto __pyx_L1;}
  __pyx_r = __pyx_2;
  __pyx_2 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.gamma");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_oshape);
  Py_DECREF(__pyx_v_oscale);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_shape);
  Py_DECREF(__pyx_v_scale);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k88p;
static PyObject *__pyx_k89p;
static PyObject *__pyx_k90p;
static PyObject *__pyx_k91p;

static char __pyx_k88[] = "shape <= 0";
static char __pyx_k89[] = "scale <= 0";
static char __pyx_k90[] = "dfnum <= 0";
static char __pyx_k91[] = "dfden <= 0";

static PyObject *__pyx_f_6mtrand_11RandomState_f(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_f[] = "\n        f(dfnum, dfden, size=None)\n\n        Draw samples from a F distribution.\n\n        Samples are drawn from an F distribution with specified parameters,\n        `dfnum` (degrees of freedom in numerator) and `dfden` (degrees of freedom\n        in denominator), where both parameters should be greater than zero.\n\n        The random variate of the F distribution (also known as the\n        Fisher distribution) is a continuous probability distribution\n        that arises in ANOVA tests, and is the ratio of two chi-square\n        variates.\n\n        Parameters\n        ----------\n        dfnum : float\n            Degrees of freedom in numerator. Should be greater than zero.\n        dfden : float\n            Degrees of freedom in denominator. Should be greater than zero.\n        size : {tuple, int}, optional\n            Output shape.  If the given shape is, e.g., ``(m, n, k)``,\n            then ``m * n * k`` samples are drawn. By default only one sample\n            is returned.\n\n        Returns\n        -------\n        samples : {ndarray, scalar}\n            Samples from the Fisher distribution.\n\n        See Also\n        --------\n        scipy.stats.distributions.f : probability density function,\n            distribution or cumulative density function, etc.\n\n        Notes\n        -----\n\n        The F statistic is used to compare in-group variances to between-group\n        variances. Calculating the distribution depends on the sampling, and\n        so it is a function of the respective degrees of freedom in the\n        problem.  The variable `dfnum` is the number of samples minus one, the\n        between-groups degrees of freedom, while `dfden` is the within-groups\n        degrees of freedom, the sum of the number of samples in each group\n        minus the number of groups.\n\n        References\n        ----------\n        .. [1] Glantz, Stanton A. \"Primer of Biostatistics.\", McGraw-Hill,\n               Fifth Edition, 2002.\n        .. [2] Wikipedia, \"F-distribution\",\n               http://en.wikipedia.org/wiki/F-distribution\n\n        Examples\n        --------\n        An example from Glantz[1], pp 47-40.\n        Two groups, children of diabetics (25 people) and children from people\n        without diabetes (25 controls). Fasting blood glucose was measured,\n        case group had a mean value of 86.1, controls had a mean value of\n        82.2. Standard deviations were 2.09 and 2.49 respectively. Are these\n        data consistent with the null hypothesis that the parents diabetic\n        status does not affect their children\'s blood glucose levels?\n        Calculating the F statistic from the data gives a value of 36.01.\n\n        Draw samples from the distribution:\n\n        >>> dfnum = 1. # between group degrees of freedom\n        >>> dfden = 48. # within groups degrees of freedom\n        >>> s = np.random.f(dfnum, dfden, 1000)\n\n        The lower bound for the top 1% of the samples is :\n\n        >>> sort(s)[-10]\n        7.61988120985\n\n        So there is about a 1% chance that the F statistic will exceed 7.62,\n        the measured value is 36, so the null hypothesis is rejected at the 1%\n        level.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_f(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_dfnum = 0;
  PyObject *__pyx_v_dfden = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_odfnum;
  PyArrayObject *__pyx_v_odfden;
  double __pyx_v_fdfnum;
  double __pyx_v_fdfden;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"dfnum","dfden","size",0};
  __pyx_v_size = __pyx_k24;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "OO|O", __pyx_argnames, &__pyx_v_dfnum, &__pyx_v_dfden, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_dfnum);
  Py_INCREF(__pyx_v_dfden);
  Py_INCREF(__pyx_v_size);
  __pyx_v_odfnum = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_odfden = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1393 */
  __pyx_v_fdfnum = PyFloat_AsDouble(__pyx_v_dfnum);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1394 */
  __pyx_v_fdfden = PyFloat_AsDouble(__pyx_v_dfden);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1395 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1396 */
    __pyx_1 = (__pyx_v_fdfnum <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1397; goto __pyx_L1;}
      Py_INCREF(__pyx_k88p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k88p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1397; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1397; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1398 */
    __pyx_1 = (__pyx_v_fdfden <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1399; goto __pyx_L1;}
      Py_INCREF(__pyx_k89p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k89p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1399; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1399; goto __pyx_L1;}
      goto __pyx_L4;
    }
    __pyx_L4:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1400 */
    __pyx_2 = __pyx_f_6mtrand_cont2_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_f,__pyx_v_size,__pyx_v_fdfnum,__pyx_v_fdfden); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1400; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1402 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1404 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_dfnum,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1404; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_odfnum));
  __pyx_v_odfnum = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1405 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_dfden,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1405; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
  Py_DECREF(((PyObject *)__pyx_v_odfden));
  __pyx_v_odfden = ((PyArrayObject *)__pyx_2);
  Py_DECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1406 */
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1406; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_any); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1406; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1406; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_3, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1406; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyFloat_FromDouble(0.0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1406; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1406; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_odfnum));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_odfnum));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_3);
  __pyx_3 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1406; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1406; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_3);
  __pyx_3 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1406; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1406; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1407; goto __pyx_L1;}
    Py_INCREF(__pyx_k90p);
    PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k90p);
    __pyx_2 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1407; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __Pyx_Raise(__pyx_2, 0, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1407; goto __pyx_L1;}
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1408 */
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1408; goto __pyx_L1;}
  __pyx_5 = PyObject_GetAttr(__pyx_4, __pyx_n_any); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1408; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1408; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_less_equal); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1408; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_4 = PyFloat_FromDouble(0.0); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1408; goto __pyx_L1;}
  __pyx_3 = PyTuple_New(2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1408; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_odfden));
  PyTuple_SET_ITEM(__pyx_3, 0, ((PyObject *)__pyx_v_odfden));
  PyTuple_SET_ITEM(__pyx_3, 1, __pyx_4);
  __pyx_4 = 0;
  __pyx_4 = PyObject_CallObject(__pyx_2, __pyx_3); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1408; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1408; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_2, 0, __pyx_4);
  __pyx_4 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_5, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1408; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_3); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1408; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  if (__pyx_1) {
    __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1409; goto __pyx_L1;}
    Py_INCREF(__pyx_k91p);
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_k91p);
    __pyx_5 = PyObject_CallObject(PyExc_ValueError, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1409; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    __Pyx_Raise(__pyx_5, 0, 0);
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1409; goto __pyx_L1;}
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1410 */
  __pyx_2 = __pyx_f_6mtrand_cont2_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_f,__pyx_v_size,__pyx_v_odfnum,__pyx_v_odfden); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1410; goto __pyx_L1;}
  __pyx_r = __pyx_2;
  __pyx_2 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.f");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_odfnum);
  Py_DECREF(__pyx_v_odfden);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_dfnum);
  Py_DECREF(__pyx_v_dfden);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_n_less;

static PyObject *__pyx_k92p;
static PyObject *__pyx_k93p;
static PyObject *__pyx_k94p;
static PyObject *__pyx_k95p;
static PyObject *__pyx_k96p;
static PyObject *__pyx_k97p;

static char __pyx_k92[] = "dfnum <= 1";
static char __pyx_k93[] = "dfden <= 0";
static char __pyx_k94[] = "nonc < 0";
static char __pyx_k95[] = "dfnum <= 1";
static char __pyx_k96[] = "dfden <= 0";
static char __pyx_k97[] = "nonc < 0";

static PyObject *__pyx_f_6mtrand_11RandomState_noncentral_f(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_noncentral_f[] = "\n        noncentral_f(dfnum, dfden, nonc, size=None)\n\n        Noncentral F distribution.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_noncentral_f(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_dfnum = 0;
  PyObject *__pyx_v_dfden = 0;
  PyObject *__pyx_v_nonc = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_odfnum;
  PyArrayObject *__pyx_v_odfden;
  PyArrayObject *__pyx_v_ononc;
  double __pyx_v_fdfnum;
  double __pyx_v_fdfden;
  double __pyx_v_fnonc;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"dfnum","dfden","nonc","size",0};
  __pyx_v_size = __pyx_k25;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "OOO|O", __pyx_argnames, &__pyx_v_dfnum, &__pyx_v_dfden, &__pyx_v_nonc, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_dfnum);
  Py_INCREF(__pyx_v_dfden);
  Py_INCREF(__pyx_v_nonc);
  Py_INCREF(__pyx_v_size);
  __pyx_v_odfnum = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_odfden = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_ononc = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1422 */
  __pyx_v_fdfnum = PyFloat_AsDouble(__pyx_v_dfnum);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1423 */
  __pyx_v_fdfden = PyFloat_AsDouble(__pyx_v_dfden);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1424 */
  __pyx_v_fnonc = PyFloat_AsDouble(__pyx_v_nonc);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1425 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1426 */
    __pyx_1 = (__pyx_v_fdfnum <= 1);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1427; goto __pyx_L1;}
      Py_INCREF(__pyx_k92p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k92p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1427; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1427; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1428 */
    __pyx_1 = (__pyx_v_fdfden <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1429; goto __pyx_L1;}
      Py_INCREF(__pyx_k93p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k93p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1429; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1429; goto __pyx_L1;}
      goto __pyx_L4;
    }
    __pyx_L4:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1430 */
    __pyx_1 = (__pyx_v_fnonc < 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1431; goto __pyx_L1;}
      Py_INCREF(__pyx_k94p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k94p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1431; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1431; goto __pyx_L1;}
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1432 */
    __pyx_2 = __pyx_f_6mtrand_cont3_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_noncentral_f,__pyx_v_size,__pyx_v_fdfnum,__pyx_v_fdfden,__pyx_v_fnonc); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1432; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1435 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1437 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_dfnum,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1437; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_odfnum));
  __pyx_v_odfnum = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1438 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_dfden,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1438; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
  Py_DECREF(((PyObject *)__pyx_v_odfden));
  __pyx_v_odfden = ((PyArrayObject *)__pyx_2);
  Py_DECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1439 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_nonc,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1439; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_ononc));
  __pyx_v_ononc = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1441 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1441; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_any); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1441; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1441; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1441; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = PyFloat_FromDouble(1.0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1441; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1441; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_odfnum));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_odfnum));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_2);
  __pyx_2 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1441; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1441; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_2);
  __pyx_2 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_3, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1441; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1441; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1442; goto __pyx_L1;}
    Py_INCREF(__pyx_k95p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k95p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1442; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1442; goto __pyx_L1;}
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1443 */
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1443; goto __pyx_L1;}
  __pyx_5 = PyObject_GetAttr(__pyx_4, __pyx_n_any); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1443; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1443; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_less_equal); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1443; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_4 = PyFloat_FromDouble(0.0); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1443; goto __pyx_L1;}
  __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1443; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_odfden));
  PyTuple_SET_ITEM(__pyx_2, 0, ((PyObject *)__pyx_v_odfden));
  PyTuple_SET_ITEM(__pyx_2, 1, __pyx_4);
  __pyx_4 = 0;
  __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1443; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1443; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_3, 0, __pyx_4);
  __pyx_4 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_5, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1443; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_2); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1443; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  if (__pyx_1) {
    __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1444; goto __pyx_L1;}
    Py_INCREF(__pyx_k96p);
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_k96p);
    __pyx_5 = PyObject_CallObject(PyExc_ValueError, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1444; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    __Pyx_Raise(__pyx_5, 0, 0);
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1444; goto __pyx_L1;}
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1445 */
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1445; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_any); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1445; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1445; goto __pyx_L1;}
  __pyx_5 = PyObject_GetAttr(__pyx_4, __pyx_n_less); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1445; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_3 = PyFloat_FromDouble(0.0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1445; goto __pyx_L1;}
  __pyx_4 = PyTuple_New(2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1445; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_ononc));
  PyTuple_SET_ITEM(__pyx_4, 0, ((PyObject *)__pyx_v_ononc));
  PyTuple_SET_ITEM(__pyx_4, 1, __pyx_3);
  __pyx_3 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_5, __pyx_4); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1445; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_5 = PyTuple_New(1); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1445; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_5, 0, __pyx_3);
  __pyx_3 = 0;
  __pyx_4 = PyObject_CallObject(__pyx_2, __pyx_5); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1445; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_4); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1445; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  if (__pyx_1) {
    __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1446; goto __pyx_L1;}
    Py_INCREF(__pyx_k97p);
    PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k97p);
    __pyx_2 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1446; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __Pyx_Raise(__pyx_2, 0, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1446; goto __pyx_L1;}
    goto __pyx_L8;
  }
  __pyx_L8:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1447 */
  __pyx_5 = __pyx_f_6mtrand_cont3_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_noncentral_f,__pyx_v_size,__pyx_v_odfnum,__pyx_v_odfden,__pyx_v_ononc); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1447; goto __pyx_L1;}
  __pyx_r = __pyx_5;
  __pyx_5 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.noncentral_f");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_odfnum);
  Py_DECREF(__pyx_v_odfden);
  Py_DECREF(__pyx_v_ononc);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_dfnum);
  Py_DECREF(__pyx_v_dfden);
  Py_DECREF(__pyx_v_nonc);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k98p;
static PyObject *__pyx_k99p;

static char __pyx_k98[] = "df <= 0";
static char __pyx_k99[] = "df <= 0";

static PyObject *__pyx_f_6mtrand_11RandomState_chisquare(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_chisquare[] = "\n        chisquare(df, size=None)\n\n        Draw samples from a chi-square distribution.\n\n        When `df` independent random variables, each with standard\n        normal distributions (mean 0, variance 1), are squared and summed,\n        the resulting distribution is chi-square (see Notes).  This\n        distribution is often used in hypothesis testing.\n\n        Parameters\n        ----------\n        df : int\n             Number of degrees of freedom.\n        size : tuple of ints, int, optional\n             Size of the returned array.  By default, a scalar is\n             returned.\n\n        Returns\n        -------\n        output : ndarray\n            Samples drawn from the distribution, packed in a `size`-shaped\n            array.\n\n        Raises\n        ------\n        ValueError\n            When `df` <= 0 or when an inappropriate `size` (e.g. ``size=-1``)\n            is given.\n\n        Notes\n        -----\n        The variable obtained by summing the squares of `df` independent,\n        standard normally distributed random variables:\n\n        .. math:: Q = \\sum_{i=0}^{\\mathtt{df}} X^2_i\n\n        is chi-square distributed, denoted\n\n        .. math:: Q \\sim \\chi^2_k.\n\n        The probability density function of the chi-squared distribution is\n\n        .. math:: p(x) = \\frac{(1/2)^{k/2}}{\\Gamma(k/2)}\n                         x^{k/2 - 1} e^{-x/2},\n\n        where :math:`\\Gamma` is the gamma function,\n\n        .. math:: \\Gamma(x) = \\int_0^{-\\infty} t^{x - 1} e^{-t} dt.\n\n        References\n        ----------\n        .. [1] NIST/SEMATECH e-Handbook of Statistical Methods,\n               http://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm\n        .. [2] Wikipedia, \"Chi-square distribution\",\n               http://en.wikipedia.org/wiki/Chi-square_distribution\n\n        Examples\n        --------\n        >>> np.random.chisquare(2,4)\n        array([ 1.89920014,  9.00867716,  3.13710533,  5.62318272])\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_chisquare(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_df = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_odf;
  double __pyx_v_fdf;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"df","size",0};
  __pyx_v_size = __pyx_k26;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O|O", __pyx_argnames, &__pyx_v_df, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_df);
  Py_INCREF(__pyx_v_size);
  __pyx_v_odf = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1517 */
  __pyx_v_fdf = PyFloat_AsDouble(__pyx_v_df);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1518 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1519 */
    __pyx_1 = (__pyx_v_fdf <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1520; goto __pyx_L1;}
      Py_INCREF(__pyx_k98p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k98p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1520; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1520; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1521 */
    __pyx_2 = __pyx_f_6mtrand_cont1_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_chisquare,__pyx_v_size,__pyx_v_fdf); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1521; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1523 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1525 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_df,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1525; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_odf));
  __pyx_v_odf = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1526 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1526; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_any); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1526; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1526; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1526; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = PyFloat_FromDouble(0.0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1526; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1526; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_odf));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_odf));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_2);
  __pyx_2 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1526; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1526; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_2);
  __pyx_2 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_3, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1526; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1526; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1527; goto __pyx_L1;}
    Py_INCREF(__pyx_k99p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k99p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1527; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1527; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1528 */
  __pyx_4 = __pyx_f_6mtrand_cont1_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_chisquare,__pyx_v_size,__pyx_v_odf); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1528; goto __pyx_L1;}
  __pyx_r = __pyx_4;
  __pyx_4 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.chisquare");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_odf);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_df);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k100p;
static PyObject *__pyx_k101p;
static PyObject *__pyx_k102p;
static PyObject *__pyx_k103p;

static char __pyx_k100[] = "df <= 0";
static char __pyx_k101[] = "nonc <= 0";
static char __pyx_k102[] = "df <= 1";
static char __pyx_k103[] = "nonc < 0";

static PyObject *__pyx_f_6mtrand_11RandomState_noncentral_chisquare(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_noncentral_chisquare[] = "\n        noncentral_chisquare(df, nonc, size=None)\n\n        Draw samples from a noncentral chi-square distribution.\n\n        The noncentral :math:`\\chi^2` distribution is a generalisation of\n        the :math:`\\chi^2` distribution.\n\n        Parameters\n        ----------\n        df : int\n            Degrees of freedom.\n        nonc : float\n            Non-centrality.\n        size : tuple of ints\n            Shape of the output.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_noncentral_chisquare(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_df = 0;
  PyObject *__pyx_v_nonc = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_odf;
  PyArrayObject *__pyx_v_ononc;
  double __pyx_v_fdf;
  double __pyx_v_fnonc;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"df","nonc","size",0};
  __pyx_v_size = __pyx_k27;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "OO|O", __pyx_argnames, &__pyx_v_df, &__pyx_v_nonc, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_df);
  Py_INCREF(__pyx_v_nonc);
  Py_INCREF(__pyx_v_size);
  __pyx_v_odf = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_ononc = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1551 */
  __pyx_v_fdf = PyFloat_AsDouble(__pyx_v_df);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1552 */
  __pyx_v_fnonc = PyFloat_AsDouble(__pyx_v_nonc);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1553 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1554 */
    __pyx_1 = (__pyx_v_fdf <= 1);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1555; goto __pyx_L1;}
      Py_INCREF(__pyx_k100p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k100p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1555; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1555; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1556 */
    __pyx_1 = (__pyx_v_fnonc <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1557; goto __pyx_L1;}
      Py_INCREF(__pyx_k101p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k101p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1557; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1557; goto __pyx_L1;}
      goto __pyx_L4;
    }
    __pyx_L4:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1558 */
    __pyx_2 = __pyx_f_6mtrand_cont2_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_noncentral_chisquare,__pyx_v_size,__pyx_v_fdf,__pyx_v_fnonc); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1558; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1561 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1563 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_df,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1563; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_odf));
  __pyx_v_odf = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1564 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_nonc,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1564; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
  Py_DECREF(((PyObject *)__pyx_v_ononc));
  __pyx_v_ononc = ((PyArrayObject *)__pyx_2);
  Py_DECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1565 */
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1565; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_any); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1565; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1565; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_3, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1565; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyFloat_FromDouble(0.0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1565; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1565; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_odf));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_odf));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_3);
  __pyx_3 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1565; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1565; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_3);
  __pyx_3 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1565; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1565; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1566; goto __pyx_L1;}
    Py_INCREF(__pyx_k102p);
    PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k102p);
    __pyx_2 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1566; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __Pyx_Raise(__pyx_2, 0, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1566; goto __pyx_L1;}
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1567 */
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1567; goto __pyx_L1;}
  __pyx_5 = PyObject_GetAttr(__pyx_4, __pyx_n_any); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1567; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1567; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_less_equal); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1567; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_4 = PyFloat_FromDouble(0.0); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1567; goto __pyx_L1;}
  __pyx_3 = PyTuple_New(2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1567; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_ononc));
  PyTuple_SET_ITEM(__pyx_3, 0, ((PyObject *)__pyx_v_ononc));
  PyTuple_SET_ITEM(__pyx_3, 1, __pyx_4);
  __pyx_4 = 0;
  __pyx_4 = PyObject_CallObject(__pyx_2, __pyx_3); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1567; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1567; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_2, 0, __pyx_4);
  __pyx_4 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_5, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1567; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_3); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1567; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  if (__pyx_1) {
    __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1568; goto __pyx_L1;}
    Py_INCREF(__pyx_k103p);
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_k103p);
    __pyx_5 = PyObject_CallObject(PyExc_ValueError, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1568; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    __Pyx_Raise(__pyx_5, 0, 0);
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1568; goto __pyx_L1;}
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1569 */
  __pyx_2 = __pyx_f_6mtrand_cont2_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_noncentral_chisquare,__pyx_v_size,__pyx_v_odf,__pyx_v_ononc); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1569; goto __pyx_L1;}
  __pyx_r = __pyx_2;
  __pyx_2 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.noncentral_chisquare");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_odf);
  Py_DECREF(__pyx_v_ononc);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_df);
  Py_DECREF(__pyx_v_nonc);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_11RandomState_standard_cauchy(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_standard_cauchy[] = "\n        standard_cauchy(size=None)\n\n        Standard Cauchy with mode=0.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_standard_cauchy(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_size = 0;
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  static char *__pyx_argnames[] = {"size",0};
  __pyx_v_size = __pyx_k28;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "|O", __pyx_argnames, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_size);
  __pyx_1 = __pyx_f_6mtrand_cont0_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_standard_cauchy,__pyx_v_size); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1579; goto __pyx_L1;}
  __pyx_r = __pyx_1;
  __pyx_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  __Pyx_AddTraceback("mtrand.RandomState.standard_cauchy");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k104p;
static PyObject *__pyx_k105p;

static char __pyx_k104[] = "df <= 0";
static char __pyx_k105[] = "df <= 0";

static PyObject *__pyx_f_6mtrand_11RandomState_standard_t(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_standard_t[] = "\n        standard_t(df, size=None)\n\n        Standard Student\'s t distribution with df degrees of freedom.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_standard_t(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_df = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_odf;
  double __pyx_v_fdf;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"df","size",0};
  __pyx_v_size = __pyx_k29;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O|O", __pyx_argnames, &__pyx_v_df, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_df);
  Py_INCREF(__pyx_v_size);
  __pyx_v_odf = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1591 */
  __pyx_v_fdf = PyFloat_AsDouble(__pyx_v_df);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1592 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1593 */
    __pyx_1 = (__pyx_v_fdf <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1594; goto __pyx_L1;}
      Py_INCREF(__pyx_k104p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k104p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1594; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1594; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1595 */
    __pyx_2 = __pyx_f_6mtrand_cont1_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_standard_t,__pyx_v_size,__pyx_v_fdf); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1595; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1597 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1599 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_df,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1599; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_odf));
  __pyx_v_odf = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1600 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1600; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_any); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1600; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1600; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1600; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = PyFloat_FromDouble(0.0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1600; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1600; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_odf));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_odf));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_2);
  __pyx_2 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1600; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1600; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_2);
  __pyx_2 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_3, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1600; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1600; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1601; goto __pyx_L1;}
    Py_INCREF(__pyx_k105p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k105p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1601; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1601; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1602 */
  __pyx_4 = __pyx_f_6mtrand_cont1_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_standard_t,__pyx_v_size,__pyx_v_odf); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1602; goto __pyx_L1;}
  __pyx_r = __pyx_4;
  __pyx_4 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.standard_t");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_odf);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_df);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k106p;
static PyObject *__pyx_k107p;

static char __pyx_k106[] = "kappa < 0";
static char __pyx_k107[] = "kappa < 0";

static PyObject *__pyx_f_6mtrand_11RandomState_vonmises(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_vonmises[] = "\n        vonmises(mu=0.0, kappa=1.0, size=None)\n\n        Draw samples from a von Mises distribution.\n\n        Samples are drawn from a von Mises distribution with specified mode (mu)\n        and dispersion (kappa), on the interval [-pi, pi].\n\n        The von Mises distribution (also known as the circular normal\n        distribution) is a continuous probability distribution on the circle. It\n        may be thought of as the circular analogue of the normal distribution.\n\n        Parameters\n        ----------\n        mu : float\n            Mode (\"center\") of the distribution.\n        kappa : float, >= 0.\n            Dispersion of the distribution.\n        size : {tuple, int}\n            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then\n            ``m * n * k`` samples are drawn.\n\n        Returns\n        -------\n        samples : {ndarray, scalar}\n            The returned samples live on the unit circle [-\\pi, \\pi].\n\n        See Also\n        --------\n        scipy.stats.distributions.vonmises : probability density function,\n            distribution or cumulative density function, etc.\n\n        Notes\n        -----\n        The probability density for the von Mises distribution is\n\n        .. math:: p(x) = \\frac{e^{\\kappa cos(x-\\mu)}}{2\\pi I_0(\\kappa)},\n\n        where :math:`\\mu` is the mode and :math:`\\kappa` the dispersion,\n        and :math:`I_0(\\kappa)` is the modified Bessel function of order 0.\n\n        The von Mises, named for Richard Edler von Mises, born in\n        Austria-Hungary, in what is now the Ukraine. He fled to the United\n        States in 1939 and became a professor at Harvard. He worked in\n        probability theory, aerodynamics, fluid mechanics, and philosophy of\n        science.\n\n        References\n        ----------\n        .. [1] Abramowitz, M. and Stegun, I. A. (ed.), Handbook of Mathematical\n               Functions, National Bureau of Standards, 1964; reprinted Dover\n               Publications, 1965.\n        .. [2] von Mises, Richard, 1964, Mathematical Theory of Probability\n               and Statistics (New York: Academic Press).\n        .. [3] Wikipedia, \"Von Mises distribution\",\n               http://en.wikipedia.org/wiki/Von_Mises_distribution\n\n        Examples\n        --------\n        Draw samples from the distribution:\n\n        >>> mu, kappa = 0.0, 4.0 # mean and dispersion\n        >>> s = np.random.vonmises(mu, kappa, 1000)\n\n        Display the histogram of the samples, along with\n        the probability density function:\n\n        >>> import matplotlib.pyplot as plt\n        >>> import scipy.special as sps\n        >>> count, bins, ignored = plt.hist(s, 50, normed=True)\n        >>> x = arange(-pi, pi, 2*pi/50.)\n        >>> y = -np.exp(kappa*np.cos(x-mu))/(2*pi*sps.jn(0,kappa))\n        >>> plt.plot(x, y/max(y), linewidth=2, color=\'r\')\n        >>> plt.show()\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_vonmises(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_mu = 0;
  PyObject *__pyx_v_kappa = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_omu;
  PyArrayObject *__pyx_v_okappa;
  double __pyx_v_fmu;
  double __pyx_v_fkappa;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"mu","kappa","size",0};
  __pyx_v_size = __pyx_k30;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "OO|O", __pyx_argnames, &__pyx_v_mu, &__pyx_v_kappa, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_mu);
  Py_INCREF(__pyx_v_kappa);
  Py_INCREF(__pyx_v_size);
  __pyx_v_omu = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_okappa = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1684 */
  __pyx_v_fmu = PyFloat_AsDouble(__pyx_v_mu);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1685 */
  __pyx_v_fkappa = PyFloat_AsDouble(__pyx_v_kappa);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1686 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1687 */
    __pyx_1 = (__pyx_v_fkappa < 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1688; goto __pyx_L1;}
      Py_INCREF(__pyx_k106p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k106p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1688; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1688; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1689 */
    __pyx_2 = __pyx_f_6mtrand_cont2_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_vonmises,__pyx_v_size,__pyx_v_fmu,__pyx_v_fkappa); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1689; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1691 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1693 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_mu,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1693; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_omu));
  __pyx_v_omu = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1694 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_kappa,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1694; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
  Py_DECREF(((PyObject *)__pyx_v_okappa));
  __pyx_v_okappa = ((PyArrayObject *)__pyx_2);
  Py_DECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1695 */
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1695; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_any); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1695; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1695; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_3, __pyx_n_less); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1695; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyFloat_FromDouble(0.0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1695; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1695; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_okappa));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_okappa));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_3);
  __pyx_3 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1695; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1695; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_3);
  __pyx_3 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1695; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1695; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1696; goto __pyx_L1;}
    Py_INCREF(__pyx_k107p);
    PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k107p);
    __pyx_2 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1696; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __Pyx_Raise(__pyx_2, 0, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1696; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1697 */
  __pyx_4 = __pyx_f_6mtrand_cont2_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_vonmises,__pyx_v_size,__pyx_v_omu,__pyx_v_okappa); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1697; goto __pyx_L1;}
  __pyx_r = __pyx_4;
  __pyx_4 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.vonmises");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_omu);
  Py_DECREF(__pyx_v_okappa);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_mu);
  Py_DECREF(__pyx_v_kappa);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k108p;
static PyObject *__pyx_k109p;

static char __pyx_k108[] = "a <= 0";
static char __pyx_k109[] = "a <= 0";

static PyObject *__pyx_f_6mtrand_11RandomState_pareto(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_pareto[] = "\n        pareto(a, size=None)\n\n        Draw samples from a Pareto distribution with specified shape.\n\n        This is a simplified version of the Generalized Pareto distribution\n        (available in SciPy), with the scale set to one and the location set to\n        zero. Most authors default the location to one.\n\n        The Pareto distribution must be greater than zero, and is unbounded above.\n        It is also known as the \"80-20 rule\".  In this distribution, 80 percent of\n        the weights are in the lowest 20 percent of the range, while the other 20\n        percent fill the remaining 80 percent of the range.\n\n        Parameters\n        ----------\n        shape : float, > 0.\n            Shape of the distribution.\n        size : tuple of ints\n            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then\n            ``m * n * k`` samples are drawn.\n\n        See Also\n        --------\n        scipy.stats.distributions.genpareto.pdf : probability density function,\n            distribution or cumulative density function, etc.\n\n        Notes\n        -----\n        The probability density for the Pareto distribution is\n\n        .. math:: p(x) = \\frac{am^a}{x^{a+1}}\n\n        where :math:`a` is the shape and :math:`m` the location\n\n        The Pareto distribution, named after the Italian economist Vilfredo Pareto,\n        is a power law probability distribution useful in many real world problems.\n        Outside the field of economics it is generally referred to as the Bradford\n        distribution. Pareto developed the distribution to describe the\n        distribution of wealth in an economy.  It has also found use in insurance,\n        web page access statistics, oil field sizes, and many other problems,\n        including the download frequency for projects in Sourceforge [1].  It is\n        one of the so-called \"fat-tailed\" distributions.\n\n\n        References\n        ----------\n        .. [1] Francis Hunt and Paul Johnson, On the Pareto Distribution of\n               Sourceforge projects.\n        .. [2] Pareto, V. (1896). Course of Political Economy. Lausanne.\n        .. [3] Reiss, R.D., Thomas, M.(2001), Statistical Analysis of Extreme\n               Values, Birkhauser Verlag, Basel, pp 23-30.\n        .. [4] Wikipedia, \"Pareto distribution\",\n               http://en.wikipedia.org/wiki/Pareto_distribution\n\n        Examples\n        --------\n        Draw samples from the distribution:\n\n        >>> a, m = 3., 1. # shape and mode\n        >>> s = np.random.pareto(a, 1000) + m\n\n        Display the histogram of the samples, along with\n        the probability density function:\n\n        >>> import matplotlib.pyplot as plt\n        >>> count, bins, ignored = plt.hist(s, 100, normed=True, align=\'center\')\n        >>> fit = a*m**a/bins**(a+1)\n        >>> plt.plot(bins, max(count)*fit/max(fit),linewidth=2, color=\'r\')\n        >>> plt.show()\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_pareto(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_a = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_oa;
  double __pyx_v_fa;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"a","size",0};
  __pyx_v_size = __pyx_k31;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O|O", __pyx_argnames, &__pyx_v_a, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_a);
  Py_INCREF(__pyx_v_size);
  __pyx_v_oa = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1775 */
  __pyx_v_fa = PyFloat_AsDouble(__pyx_v_a);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1776 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1777 */
    __pyx_1 = (__pyx_v_fa <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1778; goto __pyx_L1;}
      Py_INCREF(__pyx_k108p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k108p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1778; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1778; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1779 */
    __pyx_2 = __pyx_f_6mtrand_cont1_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_pareto,__pyx_v_size,__pyx_v_fa); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1779; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1781 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1783 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_a,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1783; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_oa));
  __pyx_v_oa = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1784 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1784; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_any); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1784; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1784; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1784; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = PyFloat_FromDouble(0.0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1784; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1784; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_oa));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_oa));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_2);
  __pyx_2 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1784; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1784; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_2);
  __pyx_2 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_3, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1784; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1784; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1785; goto __pyx_L1;}
    Py_INCREF(__pyx_k109p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k109p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1785; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1785; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1786 */
  __pyx_4 = __pyx_f_6mtrand_cont1_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_pareto,__pyx_v_size,__pyx_v_oa); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1786; goto __pyx_L1;}
  __pyx_r = __pyx_4;
  __pyx_4 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.pareto");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_oa);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_a);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k110p;
static PyObject *__pyx_k111p;

static char __pyx_k110[] = "a <= 0";
static char __pyx_k111[] = "a <= 0";

static PyObject *__pyx_f_6mtrand_11RandomState_weibull(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_weibull[] = "\n        weibull(a, size=None)\n\n        Weibull distribution.\n\n        Draw samples from a 1-parameter Weibull distribution with the given\n        shape parameter.\n\n        .. math:: X = (-ln(U))^{1/a}\n\n        Here, U is drawn from the uniform distribution over (0,1].\n\n        The more common 2-parameter Weibull, including a scale parameter\n        :math:`\\lambda` is just :math:`X = \\lambda(-ln(U))^{1/a}`.\n\n        The Weibull (or Type III asymptotic extreme value distribution for smallest\n        values, SEV Type III, or Rosin-Rammler distribution) is one of a class of\n        Generalized Extreme Value (GEV) distributions used in modeling extreme\n        value problems.  This class includes the Gumbel and Frechet distributions.\n\n        Parameters\n        ----------\n        a : float\n            Shape of the distribution.\n        size : tuple of ints\n            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then\n            ``m * n * k`` samples are drawn.\n\n        See Also\n        --------\n        scipy.stats.distributions.weibull : probability density function,\n            distribution or cumulative density function, etc.\n\n        gumbel, scipy.stats.distributions.genextreme\n\n        Notes\n        -----\n        The probability density for the Weibull distribution is\n\n        .. math:: p(x) = \\frac{a}\n                         {\\lambda}(\\frac{x}{\\lambda})^{a-1}e^{-(x/\\lambda)^a},\n\n        where :math:`a` is the shape and :math:`\\lambda` the scale.\n\n        The function has its peak (the mode) at\n        :math:`\\lambda(\\frac{a-1}{a})^{1/a}`.\n\n        When ``a = 1``, the Weibull distribution reduces to the exponential\n        distribution.\n\n        References\n        ----------\n        .. [1] Waloddi Weibull, Professor, Royal Technical University, Stockholm,\n               1939 \"A Statistical Theory Of The Strength Of Materials\",\n               Ingeniorsvetenskapsakademiens Handlingar Nr 151, 1939,\n               Generalstabens Litografiska Anstalts Forlag, Stockholm.\n        .. [2] Waloddi Weibull, 1951 \"A Statistical Distribution Function of Wide\n               Applicability\",  Journal Of Applied Mechanics ASME Paper.\n        .. [3] Wikipedia, \"Weibull distribution\",\n               http://en.wikipedia.org/wiki/Weibull_distribution\n\n        Examples\n        --------\n        Draw samples from the distribution:\n\n        >>> a = 5. # shape\n        >>> s = np.random.weibull(a, 1000)\n\n        Display the histogram of the samples, along with\n        the probability density function:\n\n        >>> import matplotlib.pyplot as plt\n        >>> def weib(x,n,a):\n        ...     return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)\n\n        >>> count, bins, ignored = plt.hist(np.random.weibull(5.,1000))\n        >>> x = np.arange(1,100.)/50.\n        >>> scale = count.max()/weib(x, 1., 5.).max()\n        >>> plt.plot(x, weib(x, 1., 5.)*scale)\n        >>> plt.show()\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_weibull(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_a = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_oa;
  double __pyx_v_fa;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"a","size",0};
  __pyx_v_size = __pyx_k32;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O|O", __pyx_argnames, &__pyx_v_a, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_a);
  Py_INCREF(__pyx_v_size);
  __pyx_v_oa = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1874 */
  __pyx_v_fa = PyFloat_AsDouble(__pyx_v_a);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1875 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1876 */
    __pyx_1 = (__pyx_v_fa <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1877; goto __pyx_L1;}
      Py_INCREF(__pyx_k110p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k110p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1877; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1877; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1878 */
    __pyx_2 = __pyx_f_6mtrand_cont1_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_weibull,__pyx_v_size,__pyx_v_fa); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1878; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1880 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1882 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_a,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1882; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_oa));
  __pyx_v_oa = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1883 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1883; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_any); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1883; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1883; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1883; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = PyFloat_FromDouble(0.0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1883; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1883; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_oa));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_oa));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_2);
  __pyx_2 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1883; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1883; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_2);
  __pyx_2 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_3, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1883; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1883; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1884; goto __pyx_L1;}
    Py_INCREF(__pyx_k111p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k111p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1884; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1884; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1885 */
  __pyx_4 = __pyx_f_6mtrand_cont1_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_weibull,__pyx_v_size,__pyx_v_oa); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1885; goto __pyx_L1;}
  __pyx_r = __pyx_4;
  __pyx_4 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.weibull");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_oa);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_a);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k112p;
static PyObject *__pyx_k113p;

static char __pyx_k112[] = "a <= 0";
static char __pyx_k113[] = "a <= 0";

static PyObject *__pyx_f_6mtrand_11RandomState_power(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_power[] = "\n        power(a, size=None)\n\n        Power distribution.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_power(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_a = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_oa;
  double __pyx_v_fa;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"a","size",0};
  __pyx_v_size = __pyx_k33;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O|O", __pyx_argnames, &__pyx_v_a, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_a);
  Py_INCREF(__pyx_v_size);
  __pyx_v_oa = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1897 */
  __pyx_v_fa = PyFloat_AsDouble(__pyx_v_a);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1898 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1899 */
    __pyx_1 = (__pyx_v_fa <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1900; goto __pyx_L1;}
      Py_INCREF(__pyx_k112p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k112p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1900; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1900; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1901 */
    __pyx_2 = __pyx_f_6mtrand_cont1_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_power,__pyx_v_size,__pyx_v_fa); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1901; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1903 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1905 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_a,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1905; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_oa));
  __pyx_v_oa = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1906 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1906; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_any); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1906; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1906; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1906; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = PyFloat_FromDouble(0.0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1906; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1906; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_oa));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_oa));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_2);
  __pyx_2 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1906; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1906; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_2);
  __pyx_2 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_3, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1906; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1906; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1907; goto __pyx_L1;}
    Py_INCREF(__pyx_k113p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k113p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1907; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1907; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1908 */
  __pyx_4 = __pyx_f_6mtrand_cont1_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_power,__pyx_v_size,__pyx_v_oa); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1908; goto __pyx_L1;}
  __pyx_r = __pyx_4;
  __pyx_4 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.power");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_oa);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_a);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k114p;
static PyObject *__pyx_k115p;

static char __pyx_k114[] = "scale <= 0";
static char __pyx_k115[] = "scale <= 0";

static PyObject *__pyx_f_6mtrand_11RandomState_laplace(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_laplace[] = "\n        laplace(loc=0.0, scale=1.0, size=None)\n\n        Laplace or double exponential distribution.\n\n        It has the probability density function\n\n        .. math:: f(x; \\mu, \\lambda) = \\frac{1}{2\\lambda}\n                                       \\exp\\left(-\\frac{|x - \\mu|}{\\lambda}\\right).\n\n        The Laplace distribution is similar to the Gaussian/normal distribution,\n        but is sharper at the peak and has fatter tails.\n\n        Parameters\n        ----------\n        loc : float\n            The position, :math:`\\mu`, of the distribution peak.\n        scale : float\n            :math:`\\lambda`, the exponential decay.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_laplace(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_loc = 0;
  PyObject *__pyx_v_scale = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_oloc;
  PyArrayObject *__pyx_v_oscale;
  double __pyx_v_floc;
  double __pyx_v_fscale;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"loc","scale","size",0};
  __pyx_v_loc = __pyx_k34;
  __pyx_v_scale = __pyx_k35;
  __pyx_v_size = __pyx_k36;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "|OOO", __pyx_argnames, &__pyx_v_loc, &__pyx_v_scale, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_loc);
  Py_INCREF(__pyx_v_scale);
  Py_INCREF(__pyx_v_size);
  __pyx_v_oloc = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_oscale = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1935 */
  __pyx_v_floc = PyFloat_AsDouble(__pyx_v_loc);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1936 */
  __pyx_v_fscale = PyFloat_AsDouble(__pyx_v_scale);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1937 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1938 */
    __pyx_1 = (__pyx_v_fscale <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1939; goto __pyx_L1;}
      Py_INCREF(__pyx_k114p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k114p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1939; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1939; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1940 */
    __pyx_2 = __pyx_f_6mtrand_cont2_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_laplace,__pyx_v_size,__pyx_v_floc,__pyx_v_fscale); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1940; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1942 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1943 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_loc,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1943; goto __pyx_L1;}
  if (!__Pyx_TypeTest(__pyx_3, __pyx_ptype_6mtrand_ndarray)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1943; goto __pyx_L1;}
  Py_DECREF(((PyObject *)__pyx_v_oloc));
  __pyx_v_oloc = ((PyArrayObject *)__pyx_3);
  __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1944 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_scale,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1944; goto __pyx_L1;}
  if (!__Pyx_TypeTest(__pyx_2, __pyx_ptype_6mtrand_ndarray)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1944; goto __pyx_L1;}
  Py_DECREF(((PyObject *)__pyx_v_oscale));
  __pyx_v_oscale = ((PyArrayObject *)__pyx_2);
  __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1945 */
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1945; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_any); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1945; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1945; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_3, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1945; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyFloat_FromDouble(0.0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1945; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1945; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_oscale));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_oscale));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_3);
  __pyx_3 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1945; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1945; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_3);
  __pyx_3 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1945; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1945; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1946; goto __pyx_L1;}
    Py_INCREF(__pyx_k115p);
    PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k115p);
    __pyx_2 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1946; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __Pyx_Raise(__pyx_2, 0, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1946; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1947 */
  __pyx_4 = __pyx_f_6mtrand_cont2_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_laplace,__pyx_v_size,__pyx_v_oloc,__pyx_v_oscale); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1947; goto __pyx_L1;}
  __pyx_r = __pyx_4;
  __pyx_4 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.laplace");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_oloc);
  Py_DECREF(__pyx_v_oscale);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_loc);
  Py_DECREF(__pyx_v_scale);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k116p;
static PyObject *__pyx_k117p;

static char __pyx_k116[] = "scale <= 0";
static char __pyx_k117[] = "scale <= 0";

static PyObject *__pyx_f_6mtrand_11RandomState_gumbel(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_gumbel[] = "\n        gumbel(loc=0.0, scale=1.0, size=None)\n\n        Gumbel distribution.\n\n        Draw samples from a Gumbel distribution with specified location (or mean)\n        and scale (or standard deviation).\n\n        The Gumbel (or Smallest Extreme Value (SEV) or the Smallest Extreme Value\n        Type I) distribution is one of a class of Generalized Extreme Value (GEV)\n        distributions used in modeling extreme value problems.  The Gumbel is a\n        special case of the Extreme Value Type I distribution for maximums from\n        distributions with \"exponential-like\" tails, it may be derived by\n        considering a Gaussian process of measurements, and generating the pdf for\n        the maximum values from that set of measurements (see examples).\n\n        Parameters\n        ----------\n        loc : float\n            The location of the mode of the distribution.\n        scale : float\n            The scale parameter of the distribution.\n        size : tuple of ints\n            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then\n            ``m * n * k`` samples are drawn.\n\n        See Also\n        --------\n        scipy.stats.gumbel : probability density function,\n            distribution or cumulative density function, etc.\n        weibull, scipy.stats.genextreme\n\n        Notes\n        -----\n        The probability density for the Gumbel distribution is\n\n        .. math:: p(x) = \\frac{e^{-(x - \\mu)/ \\beta}}{\\beta} e^{ -e^{-(x - \\mu)/\n                  \\beta}},\n\n        where :math:`\\mu` is the mode, a location parameter, and :math:`\\beta`\n        is the scale parameter.\n\n        The Gumbel (named for German mathematician Emil Julius Gumbel) was used\n        very early in the hydrology literature, for modeling the occurrence of\n        flood events. It is also used for modeling maximum wind speed and rainfall\n        rates.  It is a \"fat-tailed\" distribution - the probability of an event in\n        the tail of the distribution is larger than if one used a Gaussian, hence\n        the surprisingly frequent occurrence of 100-year floods. Floods were\n        initially modeled as a Gaussian process, which underestimated the frequency\n        of extreme events.\n\n        It is one of a class of extreme value distributions, the Generalized\n        Extreme Value (GEV) distributions, which also includes the Weibull and\n        Frechet.\n\n        The function has a mean of :math:`\\mu + 0.57721\\beta` and a variance of\n        :math:`\\frac{\\pi^2}{6}\\beta^2`.\n\n        References\n        ----------\n        .. [1] Gumbel, E.J. (1958). Statistics of Extremes. Columbia University\n               Press.\n        .. [2] Reiss, R.-D. and Thomas M. (2001), Statistical Analysis of Extreme\n               Values, from Insurance, Finance, Hydrology and Other Fields,\n               Birkhauser Verlag, Basel: Boston : Berlin.\n        .. [3] Wikipedia, \"Gumbel distribution\",\n               http://en.wikipedia.org/wiki/Gumbel_distribution\n\n        Examples\n        --------\n        Draw samples from the distribution:\n\n        >>> mu, beta = 0, 0.1 # location and scale\n        >>> s = np.random.gumbel(mu, beta, 1000)\n\n        Display the histogram of the samples, along with\n        the probability density function:\n\n        >>> import matplotlib.pyplot as plt\n        >>> count, bins, ignored = plt.hist(s, 30, normed=True)\n        >>> plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta)\n        ...          * np.exp( -np.exp( -(bins - mu) /beta) ),\n        ...          linewidth=2, color=\'r\')\n        >>> plt.show()\n\n        Show how an extreme value distribution can arise from a Gaussian process\n        and compare to a Gaussian:\n\n        >>> means = []\n        >>> maxima = []\n        >>> for i in range(0,1000) :\n        ...    a = np.random.normal(mu, beta, 1000)\n        ...    means.append(a.mean())\n        ...    maxima.append(a.max())\n        >>> count, bins, ignored = plt.hist(maxima, 30, normed=True)\n        >>> beta = np.std(maxima)*np.pi/np.sqrt(6)\n        >>> mu = np.mean(maxima) - 0.57721*beta\n        >>> plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta)\n        ...          * np.exp(-np.exp(-(bins - mu)/beta)),\n        ...          linewidth=2, color=\'r\')\n        >>> plt.plot(bins, 1/(beta * np.sqrt(2 * np.pi))\n        ...          * np.exp(-(bins - mu)**2 / (2 * beta**2)),\n        ...          linewidth=2, color=\'g\')\n        >>> plt.show()\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_gumbel(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_loc = 0;
  PyObject *__pyx_v_scale = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_oloc;
  PyArrayObject *__pyx_v_oscale;
  double __pyx_v_floc;
  double __pyx_v_fscale;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"loc","scale","size",0};
  __pyx_v_loc = __pyx_k37;
  __pyx_v_scale = __pyx_k38;
  __pyx_v_size = __pyx_k39;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "|OOO", __pyx_argnames, &__pyx_v_loc, &__pyx_v_scale, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_loc);
  Py_INCREF(__pyx_v_scale);
  Py_INCREF(__pyx_v_size);
  __pyx_v_oloc = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_oscale = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2059 */
  __pyx_v_floc = PyFloat_AsDouble(__pyx_v_loc);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2060 */
  __pyx_v_fscale = PyFloat_AsDouble(__pyx_v_scale);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2061 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2062 */
    __pyx_1 = (__pyx_v_fscale <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2063; goto __pyx_L1;}
      Py_INCREF(__pyx_k116p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k116p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2063; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2063; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2064 */
    __pyx_2 = __pyx_f_6mtrand_cont2_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_gumbel,__pyx_v_size,__pyx_v_floc,__pyx_v_fscale); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2064; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2066 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2067 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_loc,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2067; goto __pyx_L1;}
  if (!__Pyx_TypeTest(__pyx_3, __pyx_ptype_6mtrand_ndarray)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2067; goto __pyx_L1;}
  Py_DECREF(((PyObject *)__pyx_v_oloc));
  __pyx_v_oloc = ((PyArrayObject *)__pyx_3);
  __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2068 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_scale,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2068; goto __pyx_L1;}
  if (!__Pyx_TypeTest(__pyx_2, __pyx_ptype_6mtrand_ndarray)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2068; goto __pyx_L1;}
  Py_DECREF(((PyObject *)__pyx_v_oscale));
  __pyx_v_oscale = ((PyArrayObject *)__pyx_2);
  __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2069 */
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2069; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_any); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2069; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2069; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_3, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2069; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyFloat_FromDouble(0.0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2069; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2069; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_oscale));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_oscale));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_3);
  __pyx_3 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2069; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2069; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_3);
  __pyx_3 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2069; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2069; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2070; goto __pyx_L1;}
    Py_INCREF(__pyx_k117p);
    PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k117p);
    __pyx_2 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2070; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __Pyx_Raise(__pyx_2, 0, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2070; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2071 */
  __pyx_4 = __pyx_f_6mtrand_cont2_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_gumbel,__pyx_v_size,__pyx_v_oloc,__pyx_v_oscale); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2071; goto __pyx_L1;}
  __pyx_r = __pyx_4;
  __pyx_4 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.gumbel");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_oloc);
  Py_DECREF(__pyx_v_oscale);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_loc);
  Py_DECREF(__pyx_v_scale);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k118p;
static PyObject *__pyx_k119p;

static char __pyx_k118[] = "scale <= 0";
static char __pyx_k119[] = "scale <= 0";

static PyObject *__pyx_f_6mtrand_11RandomState_logistic(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_logistic[] = "\n        logistic(loc=0.0, scale=1.0, size=None)\n\n        Draw samples from a Logistic distribution.\n\n        Samples are drawn from a Logistic distribution with specified\n        parameters, loc (location or mean, also median), and scale (>0).\n\n        Parameters\n        ----------\n        loc : float\n\n        scale : float > 0.\n\n        size : {tuple, int}\n            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then\n            ``m * n * k`` samples are drawn.\n\n        Returns\n        -------\n        samples : {ndarray, scalar}\n                  where the values are all integers in  [0, n].\n\n        See Also\n        --------\n        scipy.stats.distributions.logistic : probability density function,\n            distribution or cumulative density function, etc.\n\n        Notes\n        -----\n        The probability density for the Logistic distribution is\n\n        .. math:: P(x) = P(x) = \\frac{e^{-(x-\\mu)/s}}{s(1+e^{-(x-\\mu)/s})^2},\n\n        where :math:`\\mu` = location and :math:`s` = scale.\n\n        The Logistic distribution is used in Extreme Value problems where it\n        can act as a mixture of Gumbel distributions, in Epidemiology, and by\n        the World Chess Federation (FIDE) where it is used in the Elo ranking\n        system, assuming the performance of each player is a logistically\n        distributed random variable.\n\n        References\n        ----------\n        .. [1] Reiss, R.-D. and Thomas M. (2001), Statistical Analysis of Extreme\n               Values, from Insurance, Finance, Hydrology and Other Fields,\n               Birkhauser Verlag, Basel, pp 132-133.\n        .. [2] Weisstein, Eric W. \"Logistic Distribution.\" From\n               MathWorld--A Wolfram Web Resource.\n               http://mathworld.wolfram.com/LogisticDistribution.html\n        .. [3] Wikipedia, \"Logistic-distribution\",\n               http://en.wikipedia.org/wiki/Logistic-distribution\n\n        Examples\n        --------\n        Draw samples from the distribution:\n\n        >>> loc, scale = 10, 1\n        >>> s = np.random.logistic(loc, scale, 10000)\n        >>> count, bins, ignored = plt.hist(s, bins=50)\n\n        #   plot against distribution\n\n        >>> def logist(x, loc, scale):\n        ...     return exp((loc-x)/scale)/(scale*(1+exp((loc-x)/scale))**2)\n        >>> plt.plot(bins, logist(bins, loc, scale)*count.max()/\\\n        ... logist(bins, loc, scale).max())\n        >>> plt.show()\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_logistic(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_loc = 0;
  PyObject *__pyx_v_scale = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_oloc;
  PyArrayObject *__pyx_v_oscale;
  double __pyx_v_floc;
  double __pyx_v_fscale;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"loc","scale","size",0};
  __pyx_v_loc = __pyx_k40;
  __pyx_v_scale = __pyx_k41;
  __pyx_v_size = __pyx_k42;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "|OOO", __pyx_argnames, &__pyx_v_loc, &__pyx_v_scale, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_loc);
  Py_INCREF(__pyx_v_scale);
  Py_INCREF(__pyx_v_size);
  __pyx_v_oloc = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_oscale = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2147 */
  __pyx_v_floc = PyFloat_AsDouble(__pyx_v_loc);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2148 */
  __pyx_v_fscale = PyFloat_AsDouble(__pyx_v_scale);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2149 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2150 */
    __pyx_1 = (__pyx_v_fscale <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2151; goto __pyx_L1;}
      Py_INCREF(__pyx_k118p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k118p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2151; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2151; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2152 */
    __pyx_2 = __pyx_f_6mtrand_cont2_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_logistic,__pyx_v_size,__pyx_v_floc,__pyx_v_fscale); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2152; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2154 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2155 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_loc,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2155; goto __pyx_L1;}
  if (!__Pyx_TypeTest(__pyx_3, __pyx_ptype_6mtrand_ndarray)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2155; goto __pyx_L1;}
  Py_DECREF(((PyObject *)__pyx_v_oloc));
  __pyx_v_oloc = ((PyArrayObject *)__pyx_3);
  __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2156 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_scale,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2156; goto __pyx_L1;}
  if (!__Pyx_TypeTest(__pyx_2, __pyx_ptype_6mtrand_ndarray)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2156; goto __pyx_L1;}
  Py_DECREF(((PyObject *)__pyx_v_oscale));
  __pyx_v_oscale = ((PyArrayObject *)__pyx_2);
  __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2157 */
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2157; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_any); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2157; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2157; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_3, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2157; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyFloat_FromDouble(0.0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2157; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2157; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_oscale));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_oscale));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_3);
  __pyx_3 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2157; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2157; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_3);
  __pyx_3 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2157; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2157; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2158; goto __pyx_L1;}
    Py_INCREF(__pyx_k119p);
    PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k119p);
    __pyx_2 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2158; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __Pyx_Raise(__pyx_2, 0, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2158; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2159 */
  __pyx_4 = __pyx_f_6mtrand_cont2_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_logistic,__pyx_v_size,__pyx_v_oloc,__pyx_v_oscale); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2159; goto __pyx_L1;}
  __pyx_r = __pyx_4;
  __pyx_4 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.logistic");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_oloc);
  Py_DECREF(__pyx_v_oscale);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_loc);
  Py_DECREF(__pyx_v_scale);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k120p;
static PyObject *__pyx_k121p;

static char __pyx_k120[] = "sigma <= 0";
static char __pyx_k121[] = "sigma <= 0.0";

static PyObject *__pyx_f_6mtrand_11RandomState_lognormal(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_lognormal[] = "\n        lognormal(mean=0.0, sigma=1.0, size=None)\n\n        Return samples drawn from a log-normal distribution.\n\n        Draw samples from a log-normal distribution with specified mean, standard\n        deviation, and shape. Note that the mean and standard deviation are not the\n        values for the distribution itself, but of the underlying normal\n        distribution it is derived from.\n\n\n        Parameters\n        ----------\n        mean : float\n            Mean value of the underlying normal distribution\n        sigma : float, >0.\n            Standard deviation of the underlying normal distribution\n        size : tuple of ints\n            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then\n            ``m * n * k`` samples are drawn.\n\n        See Also\n        --------\n        scipy.stats.lognorm : probability density function, distribution,\n            cumulative density function, etc.\n\n        Notes\n        -----\n        A variable `x` has a log-normal distribution if `log(x)` is normally\n        distributed.\n\n        The probability density function for the log-normal distribution is\n\n        .. math:: p(x) = \\frac{1}{\\sigma x \\sqrt{2\\pi}}\n                         e^{(-\\frac{(ln(x)-\\mu)^2}{2\\sigma^2})}\n\n        where :math:`\\mu` is the mean and :math:`\\sigma` is the standard deviation\n        of the normally distributed logarithm of the variable.\n\n        A log-normal distribution results if a random variable is the *product* of\n        a large number of independent, identically-distributed variables in the\n        same way that a normal distribution results if the variable is the *sum*\n        of a large number of independent, identically-distributed variables\n        (see the last example). It is one of the so-called \"fat-tailed\"\n        distributions.\n\n        The log-normal distribution is commonly used to model the lifespan of units\n        with fatigue-stress failure modes. Since this includes\n        most mechanical systems, the log-normal distribution has widespread\n        application.\n\n        It is also commonly used to model oil field sizes, species abundance, and\n        latent periods of infectious diseases.\n\n        References\n        ----------\n        .. [1] Eckhard Limpert, Werner A. Stahel, and Markus Abbt, \"Log-normal\n               Distributions across the Sciences: Keys and Clues\", May 2001\n               Vol. 51 No. 5 BioScience\n               http://stat.ethz.ch/~stahel/lognormal/bioscience.pdf\n        .. [2] Reiss, R.D., Thomas, M.(2001), Statistical Analysis of Extreme\n               Values, Birkhauser Verlag, Basel, pp 31-32.\n        .. [3] Wikipedia, \"Lognormal distribution\",\n               http://en.wikipedia.org/wiki/Lognormal_distribution\n\n        Examples\n        --------\n        Draw samples from the distribution:\n\n        >>> mu, sigma = 3., 1. # mean and standard deviation\n        >>> s = np.random.lognormal(mu, sigma, 1000)\n\n        Display the histogram of the samples, along with\n        the probability density function:\n\n        >>> import matplotlib.pyplot as plt\n        >>> count, bins, ignored = plt.hist(s, 100, normed=True, align=\'center\')\n\n        >>> x = np.linspace(min(bins), max(bins), 10000)\n        >>> pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))\n        ...        / (x * sigma * np.sqrt(2 * np.pi)))\n\n        >>> plt.plot(x, pdf, linewidth=2, color=\'r\')\n        >>> plt.axis(\'tight\')\n        >>> plt.show()\n\n        Demonstrate that taking the products of random samples from a uniform\n        distribution can be fit well by a log-normal probability density function.\n\n        >>> # Generate a thousand samples: each is the product of 100 random\n        >>> # values, drawn from a normal distribution.\n        >>> b = []\n        >>> for i in range(1000):\n        ...    a = 10. + np.random.random(100)\n        ...    b.append(np.product(a))\n\n        >>> b = np.array(b) / np.min(b) # scale values to be positive\n\n        >>> count, bins, ignored = plt.hist(b, 100, normed=True, align=\'center\')\n\n        >>> sigma = np.std(np.log(b))\n        >>> mu = np.mean(np.log(b))\n\n        >>> x = np.linspace(min(bins), max(bins), 10000)\n        >>> pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))\n        ...        / (x * sigma * np.sqrt(2 * np.pi)))\n\n        >>> plt.plot(x, pdf, color=\'r\', linewidth=2)\n        >>> plt.show()\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_lognormal(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_mean = 0;
  PyObject *__pyx_v_sigma = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_omean;
  PyArrayObject *__pyx_v_osigma;
  double __pyx_v_fmean;
  double __pyx_v_fsigma;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"mean","sigma","size",0};
  __pyx_v_mean = __pyx_k43;
  __pyx_v_sigma = __pyx_k44;
  __pyx_v_size = __pyx_k45;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "|OOO", __pyx_argnames, &__pyx_v_mean, &__pyx_v_sigma, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_mean);
  Py_INCREF(__pyx_v_sigma);
  Py_INCREF(__pyx_v_size);
  __pyx_v_omean = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_osigma = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2276 */
  __pyx_v_fmean = PyFloat_AsDouble(__pyx_v_mean);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2277 */
  __pyx_v_fsigma = PyFloat_AsDouble(__pyx_v_sigma);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2279 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2280 */
    __pyx_1 = (__pyx_v_fsigma <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2281; goto __pyx_L1;}
      Py_INCREF(__pyx_k120p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k120p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2281; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2281; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2282 */
    __pyx_2 = __pyx_f_6mtrand_cont2_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_lognormal,__pyx_v_size,__pyx_v_fmean,__pyx_v_fsigma); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2282; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2284 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2286 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_mean,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2286; goto __pyx_L1;}
  if (!__Pyx_TypeTest(__pyx_3, __pyx_ptype_6mtrand_ndarray)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2286; goto __pyx_L1;}
  Py_DECREF(((PyObject *)__pyx_v_omean));
  __pyx_v_omean = ((PyArrayObject *)__pyx_3);
  __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2287 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_sigma,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2287; goto __pyx_L1;}
  if (!__Pyx_TypeTest(__pyx_2, __pyx_ptype_6mtrand_ndarray)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2287; goto __pyx_L1;}
  Py_DECREF(((PyObject *)__pyx_v_osigma));
  __pyx_v_osigma = ((PyArrayObject *)__pyx_2);
  __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2288 */
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2288; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_any); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2288; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2288; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_3, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2288; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyFloat_FromDouble(0.0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2288; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2288; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_osigma));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_osigma));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_3);
  __pyx_3 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2288; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2288; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_3);
  __pyx_3 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2288; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2288; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2289; goto __pyx_L1;}
    Py_INCREF(__pyx_k121p);
    PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k121p);
    __pyx_2 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2289; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __Pyx_Raise(__pyx_2, 0, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2289; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2290 */
  __pyx_4 = __pyx_f_6mtrand_cont2_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_lognormal,__pyx_v_size,__pyx_v_omean,__pyx_v_osigma); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2290; goto __pyx_L1;}
  __pyx_r = __pyx_4;
  __pyx_4 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.lognormal");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_omean);
  Py_DECREF(__pyx_v_osigma);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_mean);
  Py_DECREF(__pyx_v_sigma);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k122p;
static PyObject *__pyx_k123p;

static char __pyx_k122[] = "scale <= 0";
static char __pyx_k123[] = "scale <= 0.0";

static PyObject *__pyx_f_6mtrand_11RandomState_rayleigh(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_rayleigh[] = "\n        rayleigh(scale=1.0, size=None)\n\n        Rayleigh distribution.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_rayleigh(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_scale = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_oscale;
  double __pyx_v_fscale;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"scale","size",0};
  __pyx_v_scale = __pyx_k46;
  __pyx_v_size = __pyx_k47;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "|OO", __pyx_argnames, &__pyx_v_scale, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_scale);
  Py_INCREF(__pyx_v_size);
  __pyx_v_oscale = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2302 */
  __pyx_v_fscale = PyFloat_AsDouble(__pyx_v_scale);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2304 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2305 */
    __pyx_1 = (__pyx_v_fscale <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2306; goto __pyx_L1;}
      Py_INCREF(__pyx_k122p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k122p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2306; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2306; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2307 */
    __pyx_2 = __pyx_f_6mtrand_cont1_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_rayleigh,__pyx_v_size,__pyx_v_fscale); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2307; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2309 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2311 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_scale,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2311; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_oscale));
  __pyx_v_oscale = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2312 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2312; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_any); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2312; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2312; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2312; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = PyFloat_FromDouble(0.0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2312; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2312; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_oscale));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_oscale));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_2);
  __pyx_2 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2312; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2312; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_2);
  __pyx_2 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_3, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2312; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2312; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2313; goto __pyx_L1;}
    Py_INCREF(__pyx_k123p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k123p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2313; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2313; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2314 */
  __pyx_4 = __pyx_f_6mtrand_cont1_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_rayleigh,__pyx_v_size,__pyx_v_oscale); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2314; goto __pyx_L1;}
  __pyx_r = __pyx_4;
  __pyx_4 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.rayleigh");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_oscale);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_scale);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k124p;
static PyObject *__pyx_k125p;
static PyObject *__pyx_k126p;
static PyObject *__pyx_k127p;

static char __pyx_k124[] = "mean <= 0";
static char __pyx_k125[] = "scale <= 0";
static char __pyx_k126[] = "mean <= 0.0";
static char __pyx_k127[] = "scale <= 0.0";

static PyObject *__pyx_f_6mtrand_11RandomState_wald(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_wald[] = "\n        wald(mean, scale, size=None)\n\n        Wald (inverse Gaussian) distribution.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_wald(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_mean = 0;
  PyObject *__pyx_v_scale = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_omean;
  PyArrayObject *__pyx_v_oscale;
  double __pyx_v_fmean;
  double __pyx_v_fscale;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"mean","scale","size",0};
  __pyx_v_size = __pyx_k48;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "OO|O", __pyx_argnames, &__pyx_v_mean, &__pyx_v_scale, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_mean);
  Py_INCREF(__pyx_v_scale);
  Py_INCREF(__pyx_v_size);
  __pyx_v_omean = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_oscale = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2326 */
  __pyx_v_fmean = PyFloat_AsDouble(__pyx_v_mean);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2327 */
  __pyx_v_fscale = PyFloat_AsDouble(__pyx_v_scale);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2328 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2329 */
    __pyx_1 = (__pyx_v_fmean <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2330; goto __pyx_L1;}
      Py_INCREF(__pyx_k124p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k124p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2330; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2330; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2331 */
    __pyx_1 = (__pyx_v_fscale <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2332; goto __pyx_L1;}
      Py_INCREF(__pyx_k125p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k125p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2332; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2332; goto __pyx_L1;}
      goto __pyx_L4;
    }
    __pyx_L4:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2333 */
    __pyx_2 = __pyx_f_6mtrand_cont2_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_wald,__pyx_v_size,__pyx_v_fmean,__pyx_v_fscale); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2333; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2335 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2336 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_mean,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2336; goto __pyx_L1;}
  if (!__Pyx_TypeTest(__pyx_3, __pyx_ptype_6mtrand_ndarray)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2336; goto __pyx_L1;}
  Py_DECREF(((PyObject *)__pyx_v_omean));
  __pyx_v_omean = ((PyArrayObject *)__pyx_3);
  __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2337 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_scale,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2337; goto __pyx_L1;}
  if (!__Pyx_TypeTest(__pyx_2, __pyx_ptype_6mtrand_ndarray)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2337; goto __pyx_L1;}
  Py_DECREF(((PyObject *)__pyx_v_oscale));
  __pyx_v_oscale = ((PyArrayObject *)__pyx_2);
  __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2338 */
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2338; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_any); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2338; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2338; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_3, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2338; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyFloat_FromDouble(0.0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2338; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2338; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_omean));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_omean));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_3);
  __pyx_3 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2338; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2338; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_3);
  __pyx_3 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2338; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2338; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2339; goto __pyx_L1;}
    Py_INCREF(__pyx_k126p);
    PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k126p);
    __pyx_2 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2339; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __Pyx_Raise(__pyx_2, 0, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2339; goto __pyx_L1;}
    goto __pyx_L5;
  }
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2340; goto __pyx_L1;}
  __pyx_5 = PyObject_GetAttr(__pyx_4, __pyx_n_any); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2340; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2340; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_less_equal); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2340; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_4 = PyFloat_FromDouble(0.0); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2340; goto __pyx_L1;}
  __pyx_3 = PyTuple_New(2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2340; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_oscale));
  PyTuple_SET_ITEM(__pyx_3, 0, ((PyObject *)__pyx_v_oscale));
  PyTuple_SET_ITEM(__pyx_3, 1, __pyx_4);
  __pyx_4 = 0;
  __pyx_4 = PyObject_CallObject(__pyx_2, __pyx_3); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2340; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2340; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_2, 0, __pyx_4);
  __pyx_4 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_5, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2340; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_3); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2340; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  if (__pyx_1) {
    __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2341; goto __pyx_L1;}
    Py_INCREF(__pyx_k127p);
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_k127p);
    __pyx_5 = PyObject_CallObject(PyExc_ValueError, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2341; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    __Pyx_Raise(__pyx_5, 0, 0);
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2341; goto __pyx_L1;}
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2342 */
  __pyx_2 = __pyx_f_6mtrand_cont2_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_wald,__pyx_v_size,__pyx_v_omean,__pyx_v_oscale); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2342; goto __pyx_L1;}
  __pyx_r = __pyx_2;
  __pyx_2 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.wald");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_omean);
  Py_DECREF(__pyx_v_oscale);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_mean);
  Py_DECREF(__pyx_v_scale);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_n_greater;
static PyObject *__pyx_n_equal;

static PyObject *__pyx_k128p;
static PyObject *__pyx_k129p;
static PyObject *__pyx_k130p;
static PyObject *__pyx_k131p;
static PyObject *__pyx_k132p;
static PyObject *__pyx_k133p;

static char __pyx_k128[] = "left > mode";
static char __pyx_k129[] = "mode > right";
static char __pyx_k130[] = "left == right";
static char __pyx_k131[] = "left > mode";
static char __pyx_k132[] = "mode > right";
static char __pyx_k133[] = "left == right";

static PyObject *__pyx_f_6mtrand_11RandomState_triangular(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_triangular[] = "\n        triangular(left, mode, right, size=None)\n\n        Triangular distribution starting at left, peaking at mode, and\n        ending at right (left <= mode <= right).\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_triangular(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_left = 0;
  PyObject *__pyx_v_mode = 0;
  PyObject *__pyx_v_right = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_oleft;
  PyArrayObject *__pyx_v_omode;
  PyArrayObject *__pyx_v_oright;
  double __pyx_v_fleft;
  double __pyx_v_fmode;
  double __pyx_v_fright;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"left","mode","right","size",0};
  __pyx_v_size = __pyx_k49;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "OOO|O", __pyx_argnames, &__pyx_v_left, &__pyx_v_mode, &__pyx_v_right, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_left);
  Py_INCREF(__pyx_v_mode);
  Py_INCREF(__pyx_v_right);
  Py_INCREF(__pyx_v_size);
  __pyx_v_oleft = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_omode = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_oright = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2357 */
  __pyx_v_fleft = PyFloat_AsDouble(__pyx_v_left);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2358 */
  __pyx_v_fright = PyFloat_AsDouble(__pyx_v_right);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2359 */
  __pyx_v_fmode = PyFloat_AsDouble(__pyx_v_mode);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2360 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2361 */
    __pyx_1 = (__pyx_v_fleft > __pyx_v_fmode);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2362; goto __pyx_L1;}
      Py_INCREF(__pyx_k128p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k128p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2362; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2362; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2363 */
    __pyx_1 = (__pyx_v_fmode > __pyx_v_fright);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2364; goto __pyx_L1;}
      Py_INCREF(__pyx_k129p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k129p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2364; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2364; goto __pyx_L1;}
      goto __pyx_L4;
    }
    __pyx_L4:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2365 */
    __pyx_1 = (__pyx_v_fleft == __pyx_v_fright);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2366; goto __pyx_L1;}
      Py_INCREF(__pyx_k130p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k130p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2366; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2366; goto __pyx_L1;}
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2367 */
    __pyx_2 = __pyx_f_6mtrand_cont3_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_triangular,__pyx_v_size,__pyx_v_fleft,__pyx_v_fmode,__pyx_v_fright); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2367; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2370 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2371 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_left,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2371; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_oleft));
  __pyx_v_oleft = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2372 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_mode,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2372; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
  Py_DECREF(((PyObject *)__pyx_v_omode));
  __pyx_v_omode = ((PyArrayObject *)__pyx_2);
  Py_DECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2373 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_right,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2373; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_oright));
  __pyx_v_oright = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2375 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2375; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_any); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2375; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2375; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_greater); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2375; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2375; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_oleft));
  PyTuple_SET_ITEM(__pyx_2, 0, ((PyObject *)__pyx_v_oleft));
  Py_INCREF(((PyObject *)__pyx_v_omode));
  PyTuple_SET_ITEM(__pyx_2, 1, ((PyObject *)__pyx_v_omode));
  __pyx_5 = PyObject_CallObject(__pyx_4, __pyx_2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2375; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2375; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_5);
  __pyx_5 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_3, __pyx_4); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2375; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_2); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2375; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  if (__pyx_1) {
    __pyx_5 = PyTuple_New(1); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2376; goto __pyx_L1;}
    Py_INCREF(__pyx_k131p);
    PyTuple_SET_ITEM(__pyx_5, 0, __pyx_k131p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_5); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2376; goto __pyx_L1;}
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2376; goto __pyx_L1;}
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2377 */
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2377; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_4, __pyx_n_any); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2377; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_5 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2377; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_5, __pyx_n_greater); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2377; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2377; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_omode));
  PyTuple_SET_ITEM(__pyx_4, 0, ((PyObject *)__pyx_v_omode));
  Py_INCREF(((PyObject *)__pyx_v_oright));
  PyTuple_SET_ITEM(__pyx_4, 1, ((PyObject *)__pyx_v_oright));
  __pyx_5 = PyObject_CallObject(__pyx_3, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2377; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2377; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_3, 0, __pyx_5);
  __pyx_5 = 0;
  __pyx_4 = PyObject_CallObject(__pyx_2, __pyx_3); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2377; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_4); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2377; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  if (__pyx_1) {
    __pyx_5 = PyTuple_New(1); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2378; goto __pyx_L1;}
    Py_INCREF(__pyx_k132p);
    PyTuple_SET_ITEM(__pyx_5, 0, __pyx_k132p);
    __pyx_2 = PyObject_CallObject(PyExc_ValueError, __pyx_5); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2378; goto __pyx_L1;}
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    __Pyx_Raise(__pyx_2, 0, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2378; goto __pyx_L1;}
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2379 */
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2379; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_3, __pyx_n_any); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2379; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_5 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2379; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_5, __pyx_n_equal); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2379; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_3 = PyTuple_New(2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2379; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_oleft));
  PyTuple_SET_ITEM(__pyx_3, 0, ((PyObject *)__pyx_v_oleft));
  Py_INCREF(((PyObject *)__pyx_v_oright));
  PyTuple_SET_ITEM(__pyx_3, 1, ((PyObject *)__pyx_v_oright));
  __pyx_5 = PyObject_CallObject(__pyx_2, __pyx_3); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2379; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2379; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_2, 0, __pyx_5);
  __pyx_5 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_4, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2379; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_3); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2379; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  if (__pyx_1) {
    __pyx_5 = PyTuple_New(1); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2380; goto __pyx_L1;}
    Py_INCREF(__pyx_k133p);
    PyTuple_SET_ITEM(__pyx_5, 0, __pyx_k133p);
    __pyx_4 = PyObject_CallObject(PyExc_ValueError, __pyx_5); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2380; goto __pyx_L1;}
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    __Pyx_Raise(__pyx_4, 0, 0);
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2380; goto __pyx_L1;}
    goto __pyx_L8;
  }
  __pyx_L8:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2381 */
  __pyx_2 = __pyx_f_6mtrand_cont3_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_triangular,__pyx_v_size,__pyx_v_oleft,__pyx_v_omode,__pyx_v_oright); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2381; goto __pyx_L1;}
  __pyx_r = __pyx_2;
  __pyx_2 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.triangular");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_oleft);
  Py_DECREF(__pyx_v_omode);
  Py_DECREF(__pyx_v_oright);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_left);
  Py_DECREF(__pyx_v_mode);
  Py_DECREF(__pyx_v_right);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k134p;
static PyObject *__pyx_k135p;
static PyObject *__pyx_k136p;
static PyObject *__pyx_k137p;
static PyObject *__pyx_k138p;
static PyObject *__pyx_k139p;

static char __pyx_k134[] = "n <= 0";
static char __pyx_k135[] = "p < 0";
static char __pyx_k136[] = "p > 1";
static char __pyx_k137[] = "n <= 0";
static char __pyx_k138[] = "p < 0";
static char __pyx_k139[] = "p > 1";

static PyObject *__pyx_f_6mtrand_11RandomState_binomial(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_binomial[] = "\n        binomial(n, p, size=None)\n\n        Draw samples from a binomial distribution.\n\n        Samples are drawn from a Binomial distribution with specified\n        parameters, n trials and p probability of success where\n        n an integer > 0 and p is in the interval [0,1]. (n may be\n        input as a float, but it is truncated to an integer in use)\n\n        Parameters\n        ----------\n        n : float (but truncated to an integer)\n                parameter, > 0.\n        p : float\n                parameter, >= 0 and <=1.\n        size : {tuple, int}\n            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then\n            ``m * n * k`` samples are drawn.\n\n        Returns\n        -------\n        samples : {ndarray, scalar}\n                  where the values are all integers in  [0, n].\n\n        See Also\n        --------\n        scipy.stats.distributions.binom : probability density function,\n            distribution or cumulative density function, etc.\n\n        Notes\n        -----\n        The probability density for the Binomial distribution is\n\n        .. math:: P(N) = \\binom{n}{N}p^N(1-p)^{n-N},\n\n        where :math:`n` is the number of trials, :math:`p` is the probability\n        of success, and :math:`N` is the number of successes.\n\n        When estimating the standard error of a proportion in a population by\n        using a random sample, the normal distribution works well unless the\n        product p*n <=5, where p = population proportion estimate, and n =\n        number of samples, in which case the binomial distribution is used\n        instead. For example, a sample of 15 people shows 4 who are left\n        handed, and 11 who are right handed. Then p = 4/15 = 27%. 0.27*15 = 4,\n        so the binomial distribution should be used in this case.\n\n        References\n        ----------\n        .. [1] Dalgaard, Peter, \"Introductory Statistics with R\",\n               Springer-Verlag, 2002.\n        .. [2] Glantz, Stanton A. \"Primer of Biostatistics.\", McGraw-Hill,\n               Fifth Edition, 2002.\n        .. [3] Lentner, Marvin, \"Elementary Applied Statistics\", Bogden\n               and Quigley, 1972.\n        .. [4] Weisstein, Eric W. \"Binomial Distribution.\" From MathWorld--A\n               Wolfram Web Resource.\n               http://mathworld.wolfram.com/BinomialDistribution.html\n        .. [5] Wikipedia, \"Binomial-distribution\",\n               http://en.wikipedia.org/wiki/Binomial_distribution\n\n        Examples\n        --------\n        Draw samples from the distribution:\n\n        >>> n, p = 10, .5 # number of trials, probability of each trial\n        >>> s = np.random.binomial(n, p, 1000)\n        # result of flipping a coin 10 times, tested 1000 times.\n\n        A real world example. A company drills 9 wild-cat oil exploration\n        wells, each with an estimated probability of success of 0.1. All nine\n        wells fail. What is the probability of that happening?\n\n        Let\'s do 20,000 trials of the model, and count the number that\n        generate zero positive results.\n\n        >>> sum(np.random.binomial(9,0.1,20000)==0)/20000.\n        answer = 0.38885, or 38%.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_binomial(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_n = 0;
  PyObject *__pyx_v_p = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_on;
  PyArrayObject *__pyx_v_op;
  long __pyx_v_ln;
  double __pyx_v_fp;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"n","p","size",0};
  __pyx_v_size = __pyx_k50;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "OO|O", __pyx_argnames, &__pyx_v_n, &__pyx_v_p, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_n);
  Py_INCREF(__pyx_v_p);
  Py_INCREF(__pyx_v_size);
  __pyx_v_on = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_op = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2470 */
  __pyx_v_fp = PyFloat_AsDouble(__pyx_v_p);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2471 */
  __pyx_v_ln = PyInt_AsLong(__pyx_v_n);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2472 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2473 */
    __pyx_1 = (__pyx_v_ln <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2474; goto __pyx_L1;}
      Py_INCREF(__pyx_k134p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k134p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2474; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2474; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2475 */
    __pyx_1 = (__pyx_v_fp < 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2476; goto __pyx_L1;}
      Py_INCREF(__pyx_k135p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k135p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2476; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2476; goto __pyx_L1;}
      goto __pyx_L4;
    }
    __pyx_1 = (__pyx_v_fp > 1);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2478; goto __pyx_L1;}
      Py_INCREF(__pyx_k136p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k136p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2478; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2478; goto __pyx_L1;}
      goto __pyx_L4;
    }
    __pyx_L4:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2479 */
    __pyx_2 = __pyx_f_6mtrand_discnp_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_binomial,__pyx_v_size,__pyx_v_ln,__pyx_v_fp); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2479; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2481 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2483 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_n,NPY_LONG,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2483; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_on));
  __pyx_v_on = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2484 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_p,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2484; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
  Py_DECREF(((PyObject *)__pyx_v_op));
  __pyx_v_op = ((PyArrayObject *)__pyx_2);
  Py_DECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2485 */
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2485; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_any); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2485; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2485; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_3, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2485; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyInt_FromLong(0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2485; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2485; goto __pyx_L1;}
  Py_INCREF(__pyx_v_n);
  PyTuple_SET_ITEM(__pyx_5, 0, __pyx_v_n);
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_3);
  __pyx_3 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2485; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2485; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_3);
  __pyx_3 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2485; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2485; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2486; goto __pyx_L1;}
    Py_INCREF(__pyx_k137p);
    PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k137p);
    __pyx_2 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2486; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __Pyx_Raise(__pyx_2, 0, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2486; goto __pyx_L1;}
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2487 */
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2487; goto __pyx_L1;}
  __pyx_5 = PyObject_GetAttr(__pyx_4, __pyx_n_any); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2487; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2487; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_less); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2487; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_4 = PyInt_FromLong(0); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2487; goto __pyx_L1;}
  __pyx_3 = PyTuple_New(2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2487; goto __pyx_L1;}
  Py_INCREF(__pyx_v_p);
  PyTuple_SET_ITEM(__pyx_3, 0, __pyx_v_p);
  PyTuple_SET_ITEM(__pyx_3, 1, __pyx_4);
  __pyx_4 = 0;
  __pyx_4 = PyObject_CallObject(__pyx_2, __pyx_3); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2487; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2487; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_2, 0, __pyx_4);
  __pyx_4 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_5, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2487; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_3); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2487; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  if (__pyx_1) {
    __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2488; goto __pyx_L1;}
    Py_INCREF(__pyx_k138p);
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_k138p);
    __pyx_5 = PyObject_CallObject(PyExc_ValueError, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2488; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    __Pyx_Raise(__pyx_5, 0, 0);
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2488; goto __pyx_L1;}
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2489 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2489; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_any); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2489; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2489; goto __pyx_L1;}
  __pyx_5 = PyObject_GetAttr(__pyx_4, __pyx_n_greater); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2489; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_2 = PyInt_FromLong(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2489; goto __pyx_L1;}
  __pyx_4 = PyTuple_New(2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2489; goto __pyx_L1;}
  Py_INCREF(__pyx_v_p);
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_v_p);
  PyTuple_SET_ITEM(__pyx_4, 1, __pyx_2);
  __pyx_2 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_5, __pyx_4); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2489; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_5 = PyTuple_New(1); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2489; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_5, 0, __pyx_2);
  __pyx_2 = 0;
  __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_5); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2489; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_4); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2489; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  if (__pyx_1) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2490; goto __pyx_L1;}
    Py_INCREF(__pyx_k139p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k139p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2490; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2490; goto __pyx_L1;}
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2491 */
  __pyx_5 = __pyx_f_6mtrand_discnp_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_binomial,__pyx_v_size,__pyx_v_on,__pyx_v_op); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2491; goto __pyx_L1;}
  __pyx_r = __pyx_5;
  __pyx_5 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.binomial");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_on);
  Py_DECREF(__pyx_v_op);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_n);
  Py_DECREF(__pyx_v_p);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k140p;
static PyObject *__pyx_k141p;
static PyObject *__pyx_k142p;
static PyObject *__pyx_k143p;
static PyObject *__pyx_k144p;
static PyObject *__pyx_k145p;

static char __pyx_k140[] = "n <= 0";
static char __pyx_k141[] = "p < 0";
static char __pyx_k142[] = "p > 1";
static char __pyx_k143[] = "n <= 0";
static char __pyx_k144[] = "p < 0";
static char __pyx_k145[] = "p > 1";

static PyObject *__pyx_f_6mtrand_11RandomState_negative_binomial(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_negative_binomial[] = "\n        negative_binomial(n, p, size=None)\n\n        Negative Binomial distribution.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_negative_binomial(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_n = 0;
  PyObject *__pyx_v_p = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_on;
  PyArrayObject *__pyx_v_op;
  double __pyx_v_fn;
  double __pyx_v_fp;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"n","p","size",0};
  __pyx_v_size = __pyx_k51;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "OO|O", __pyx_argnames, &__pyx_v_n, &__pyx_v_p, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_n);
  Py_INCREF(__pyx_v_p);
  Py_INCREF(__pyx_v_size);
  __pyx_v_on = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_op = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2505 */
  __pyx_v_fp = PyFloat_AsDouble(__pyx_v_p);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2506 */
  __pyx_v_fn = PyFloat_AsDouble(__pyx_v_n);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2507 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2508 */
    __pyx_1 = (__pyx_v_fn <= 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2509; goto __pyx_L1;}
      Py_INCREF(__pyx_k140p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k140p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2509; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2509; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2510 */
    __pyx_1 = (__pyx_v_fp < 0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2511; goto __pyx_L1;}
      Py_INCREF(__pyx_k141p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k141p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2511; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2511; goto __pyx_L1;}
      goto __pyx_L4;
    }
    __pyx_1 = (__pyx_v_fp > 1);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2513; goto __pyx_L1;}
      Py_INCREF(__pyx_k142p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k142p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2513; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2513; goto __pyx_L1;}
      goto __pyx_L4;
    }
    __pyx_L4:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2514 */
    __pyx_2 = __pyx_f_6mtrand_discdd_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_negative_binomial,__pyx_v_size,__pyx_v_fn,__pyx_v_fp); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2514; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2517 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2519 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_n,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2519; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_on));
  __pyx_v_on = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2520 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_p,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2520; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
  Py_DECREF(((PyObject *)__pyx_v_op));
  __pyx_v_op = ((PyArrayObject *)__pyx_2);
  Py_DECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2521 */
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2521; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_any); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2521; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2521; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_3, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2521; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyInt_FromLong(0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2521; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2521; goto __pyx_L1;}
  Py_INCREF(__pyx_v_n);
  PyTuple_SET_ITEM(__pyx_5, 0, __pyx_v_n);
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_3);
  __pyx_3 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2521; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2521; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_3);
  __pyx_3 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2521; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2521; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2522; goto __pyx_L1;}
    Py_INCREF(__pyx_k143p);
    PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k143p);
    __pyx_2 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2522; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __Pyx_Raise(__pyx_2, 0, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2522; goto __pyx_L1;}
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2523 */
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2523; goto __pyx_L1;}
  __pyx_5 = PyObject_GetAttr(__pyx_4, __pyx_n_any); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2523; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2523; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_less); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2523; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_4 = PyInt_FromLong(0); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2523; goto __pyx_L1;}
  __pyx_3 = PyTuple_New(2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2523; goto __pyx_L1;}
  Py_INCREF(__pyx_v_p);
  PyTuple_SET_ITEM(__pyx_3, 0, __pyx_v_p);
  PyTuple_SET_ITEM(__pyx_3, 1, __pyx_4);
  __pyx_4 = 0;
  __pyx_4 = PyObject_CallObject(__pyx_2, __pyx_3); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2523; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2523; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_2, 0, __pyx_4);
  __pyx_4 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_5, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2523; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_3); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2523; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  if (__pyx_1) {
    __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2524; goto __pyx_L1;}
    Py_INCREF(__pyx_k144p);
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_k144p);
    __pyx_5 = PyObject_CallObject(PyExc_ValueError, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2524; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    __Pyx_Raise(__pyx_5, 0, 0);
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2524; goto __pyx_L1;}
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2525 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2525; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_any); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2525; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2525; goto __pyx_L1;}
  __pyx_5 = PyObject_GetAttr(__pyx_4, __pyx_n_greater); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2525; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_2 = PyInt_FromLong(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2525; goto __pyx_L1;}
  __pyx_4 = PyTuple_New(2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2525; goto __pyx_L1;}
  Py_INCREF(__pyx_v_p);
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_v_p);
  PyTuple_SET_ITEM(__pyx_4, 1, __pyx_2);
  __pyx_2 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_5, __pyx_4); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2525; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_5 = PyTuple_New(1); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2525; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_5, 0, __pyx_2);
  __pyx_2 = 0;
  __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_5); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2525; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_4); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2525; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  if (__pyx_1) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2526; goto __pyx_L1;}
    Py_INCREF(__pyx_k145p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k145p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2526; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2526; goto __pyx_L1;}
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2527 */
  __pyx_5 = __pyx_f_6mtrand_discdd_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_negative_binomial,__pyx_v_size,__pyx_v_on,__pyx_v_op); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2527; goto __pyx_L1;}
  __pyx_r = __pyx_5;
  __pyx_5 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.negative_binomial");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_on);
  Py_DECREF(__pyx_v_op);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_n);
  Py_DECREF(__pyx_v_p);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k146p;
static PyObject *__pyx_k147p;

static char __pyx_k146[] = "lam < 0";
static char __pyx_k147[] = "lam < 0";

static PyObject *__pyx_f_6mtrand_11RandomState_poisson(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_poisson[] = "\n        poisson(lam=1.0, size=None)\n\n        Poisson distribution.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_poisson(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_lam = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_olam;
  double __pyx_v_flam;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"lam","size",0};
  __pyx_v_lam = __pyx_k52;
  __pyx_v_size = __pyx_k53;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "|OO", __pyx_argnames, &__pyx_v_lam, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_lam);
  Py_INCREF(__pyx_v_size);
  __pyx_v_olam = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2539 */
  __pyx_v_flam = PyFloat_AsDouble(__pyx_v_lam);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2540 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2541 */
    __pyx_2 = PyInt_FromLong(0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2541; goto __pyx_L1;}
    if (PyObject_Cmp(__pyx_v_lam, __pyx_2, &__pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2541; goto __pyx_L1;}
    __pyx_1 = __pyx_1 < 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2542; goto __pyx_L1;}
      Py_INCREF(__pyx_k146p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k146p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2542; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2542; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2543 */
    __pyx_2 = __pyx_f_6mtrand_discd_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_poisson,__pyx_v_size,__pyx_v_flam); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2543; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2545 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2547 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_lam,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2547; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_olam));
  __pyx_v_olam = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2548 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2548; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_any); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2548; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2548; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_less); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2548; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = PyInt_FromLong(0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2548; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2548; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_olam));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_olam));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_2);
  __pyx_2 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2548; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2548; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_2);
  __pyx_2 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_3, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2548; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2548; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2549; goto __pyx_L1;}
    Py_INCREF(__pyx_k147p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k147p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2549; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2549; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2550 */
  __pyx_4 = __pyx_f_6mtrand_discd_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_poisson,__pyx_v_size,__pyx_v_olam); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2550; goto __pyx_L1;}
  __pyx_r = __pyx_4;
  __pyx_4 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.poisson");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_olam);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_lam);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k148p;
static PyObject *__pyx_k149p;

static char __pyx_k148[] = "a <= 1.0";
static char __pyx_k149[] = "a <= 1.0";

static PyObject *__pyx_f_6mtrand_11RandomState_zipf(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_zipf[] = "\n        zipf(a, size=None)\n\n        Draw samples from a Zipf distribution.\n\n        Samples are drawn from a Zipf distribution with specified parameter (a),\n        where a > 1.\n\n        The zipf distribution (also known as the zeta\n        distribution) is a continuous probability distribution that satisfies\n        Zipf\'s law, where the frequency of an item is inversely proportional to\n        its rank in a frequency table.\n\n        Parameters\n        ----------\n        a : float\n            parameter, > 1.\n        size : {tuple, int}\n            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then\n            ``m * n * k`` samples are drawn.\n\n        Returns\n        -------\n        samples : {ndarray, scalar}\n            The returned samples are greater than or equal to one.\n\n        See Also\n        --------\n        scipy.stats.distributions.zipf : probability density function,\n            distribution or cumulative density function, etc.\n\n        Notes\n        -----\n        The probability density for the Zipf distribution is\n\n        .. math:: p(x) = \\frac{x^{-a}}{\\zeta(a)},\n\n        where :math:`\\zeta` is the Riemann Zeta function.\n\n        Named after the American linguist George Kingsley Zipf, who noted that\n        the frequency of any word in a sample of a language is inversely\n        proportional to its rank in the frequency table.\n\n\n        References\n        ----------\n        .. [1] Weisstein, Eric W. \"Zipf Distribution.\" From MathWorld--A Wolfram\n               Web Resource. http://mathworld.wolfram.com/ZipfDistribution.html\n        .. [2] Wikipedia, \"Zeta distribution\",\n               http://en.wikipedia.org/wiki/Zeta_distribution\n        .. [3] Wikipedia, \"Zipf\'s Law\",\n               http://en.wikipedia.org/wiki/Zipf%27s_law\n        .. [4] Zipf, George Kingsley (1932): Selected Studies of the Principle\n               of Relative Frequency in Language. Cambridge (Mass.).\n\n        Examples\n        --------\n        Draw samples from the distribution:\n\n        >>> a = 2. # parameter\n        >>> s = np.random.zipf(a, 1000)\n\n        Display the histogram of the samples, along with\n        the probability density function:\n\n        >>> import matplotlib.pyplot as plt\n        >>> import scipy.special as sps\n        Truncate s values at 50 so plot is interesting\n        >>> count, bins, ignored = plt.hist(s[s<50], 50, normed=True)\n        >>> x = arange(1., 50.)\n        >>> y = x**(-a)/sps.zetac(a)\n        >>> plt.plot(x, y/max(y), linewidth=2, color=\'r\')\n        >>> plt.show()\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_zipf(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_a = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_oa;
  double __pyx_v_fa;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"a","size",0};
  __pyx_v_size = __pyx_k54;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O|O", __pyx_argnames, &__pyx_v_a, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_a);
  Py_INCREF(__pyx_v_size);
  __pyx_v_oa = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2631 */
  __pyx_v_fa = PyFloat_AsDouble(__pyx_v_a);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2632 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2633 */
    __pyx_1 = (__pyx_v_fa <= 1.0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2634; goto __pyx_L1;}
      Py_INCREF(__pyx_k148p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k148p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2634; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2634; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2635 */
    __pyx_2 = __pyx_f_6mtrand_discd_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_zipf,__pyx_v_size,__pyx_v_fa); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2635; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2637 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2639 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_a,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2639; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_oa));
  __pyx_v_oa = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2640 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2640; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_any); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2640; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2640; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2640; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = PyFloat_FromDouble(1.0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2640; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2640; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_oa));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_oa));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_2);
  __pyx_2 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2640; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2640; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_2);
  __pyx_2 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_3, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2640; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2640; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2641; goto __pyx_L1;}
    Py_INCREF(__pyx_k149p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k149p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2641; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2641; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2642 */
  __pyx_4 = __pyx_f_6mtrand_discd_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_zipf,__pyx_v_size,__pyx_v_oa); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2642; goto __pyx_L1;}
  __pyx_r = __pyx_4;
  __pyx_4 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.zipf");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_oa);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_a);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_k150p;
static PyObject *__pyx_k151p;
static PyObject *__pyx_k152p;
static PyObject *__pyx_k153p;

static char __pyx_k150[] = "p < 0.0";
static char __pyx_k151[] = "p > 1.0";
static char __pyx_k152[] = "p < 0.0";
static char __pyx_k153[] = "p > 1.0";

static PyObject *__pyx_f_6mtrand_11RandomState_geometric(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_geometric[] = "\n        geometric(p, size=None)\n\n        Draw samples from the geometric distribution.\n\n        Bernoulli trials are experiments with one of two outcomes:\n        success or failure (an example of such an experiment is flipping\n        a coin).  The geometric distribution models the number of trials\n        that must be run in order to achieve success.  It is therefore\n        supported on the positive integers, ``k = 1, 2, ...``.\n\n        The probability mass function of the geometric distribution is\n\n        .. math:: f(k) = (1 - p)^{k - 1} p\n\n        where `p` is the probability of success of an individual trial.\n\n        Parameters\n        ----------\n        p : float\n            The probability of success of an individual trial.\n        size : tuple of ints\n            Number of values to draw from the distribution.  The output\n            is shaped according to `size`.\n\n        Returns\n        -------\n        out : ndarray\n            Samples from the geometric distribution, shaped according to\n            `size`.\n\n        Examples\n        --------\n        Draw ten thousand values from the geometric distribution,\n        with the probability of an individual success equal to 0.35:\n\n        >>> z = np.random.geometric(p=0.35, size=10000)\n\n        How many trials succeeded after a single run?\n\n        >>> (z == 1).sum() / 10000.\n        0.34889999999999999 #random\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_geometric(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_p = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_op;
  double __pyx_v_fp;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"p","size",0};
  __pyx_v_size = __pyx_k55;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O|O", __pyx_argnames, &__pyx_v_p, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_p);
  Py_INCREF(__pyx_v_size);
  __pyx_v_op = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2692 */
  __pyx_v_fp = PyFloat_AsDouble(__pyx_v_p);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2693 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2694 */
    __pyx_1 = (__pyx_v_fp < 0.0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2695; goto __pyx_L1;}
      Py_INCREF(__pyx_k150p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k150p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2695; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2695; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2696 */
    __pyx_1 = (__pyx_v_fp > 1.0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2697; goto __pyx_L1;}
      Py_INCREF(__pyx_k151p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k151p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2697; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2697; goto __pyx_L1;}
      goto __pyx_L4;
    }
    __pyx_L4:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2698 */
    __pyx_2 = __pyx_f_6mtrand_discd_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_geometric,__pyx_v_size,__pyx_v_fp); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2698; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2700 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2703 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_p,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2703; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_op));
  __pyx_v_op = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2704 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2704; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_any); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2704; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2704; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_less); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2704; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = PyFloat_FromDouble(0.0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2704; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2704; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_op));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_op));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_2);
  __pyx_2 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2704; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2704; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_2);
  __pyx_2 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_3, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2704; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2704; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2705; goto __pyx_L1;}
    Py_INCREF(__pyx_k152p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k152p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2705; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2705; goto __pyx_L1;}
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2706 */
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2706; goto __pyx_L1;}
  __pyx_5 = PyObject_GetAttr(__pyx_4, __pyx_n_any); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2706; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2706; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_greater); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2706; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_4 = PyFloat_FromDouble(1.0); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2706; goto __pyx_L1;}
  __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2706; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_op));
  PyTuple_SET_ITEM(__pyx_2, 0, ((PyObject *)__pyx_v_op));
  PyTuple_SET_ITEM(__pyx_2, 1, __pyx_4);
  __pyx_4 = 0;
  __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2706; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2706; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_3, 0, __pyx_4);
  __pyx_4 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_5, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2706; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_2); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2706; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  if (__pyx_1) {
    __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2707; goto __pyx_L1;}
    Py_INCREF(__pyx_k153p);
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_k153p);
    __pyx_5 = PyObject_CallObject(PyExc_ValueError, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2707; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    __Pyx_Raise(__pyx_5, 0, 0);
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2707; goto __pyx_L1;}
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2708 */
  __pyx_3 = __pyx_f_6mtrand_discd_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_geometric,__pyx_v_size,__pyx_v_op); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2708; goto __pyx_L1;}
  __pyx_r = __pyx_3;
  __pyx_3 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.geometric");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_op);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_p);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_n_add;

static PyObject *__pyx_k154p;
static PyObject *__pyx_k155p;
static PyObject *__pyx_k156p;
static PyObject *__pyx_k157p;
static PyObject *__pyx_k158p;
static PyObject *__pyx_k159p;
static PyObject *__pyx_k160p;
static PyObject *__pyx_k161p;

static char __pyx_k154[] = "ngood < 1";
static char __pyx_k155[] = "nbad < 1";
static char __pyx_k156[] = "nsample < 1";
static char __pyx_k157[] = "ngood + nbad < nsample";
static char __pyx_k158[] = "ngood < 1";
static char __pyx_k159[] = "nbad < 1";
static char __pyx_k160[] = "nsample < 1";
static char __pyx_k161[] = "ngood + nbad < nsample";

static PyObject *__pyx_f_6mtrand_11RandomState_hypergeometric(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_hypergeometric[] = "\n        hypergeometric(ngood, nbad, nsample, size=None)\n\n        Draw samples from a Hypergeometric distribution.\n\n        Samples are drawn from a Hypergeometric distribution with specified\n        parameters, ngood (ways to make a good selection), nbad (ways to make\n        a bad selection), and nsample = number of items sampled, which is less\n        than or equal to the sum ngood + nbad.\n\n        Parameters\n        ----------\n        ngood : float (but truncated to an integer)\n                parameter, > 0.\n        nbad  : float\n                parameter, >= 0.\n        nsample  : float\n                   parameter, > 0 and <= ngood+nbad\n        size : {tuple, int}\n            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then\n            ``m * n * k`` samples are drawn.\n\n        Returns\n        -------\n        samples : {ndarray, scalar}\n                  where the values are all integers in  [0, n].\n\n        See Also\n        --------\n        scipy.stats.distributions.hypergeom : probability density function,\n            distribution or cumulative density function, etc.\n\n        Notes\n        -----\n        The probability density for the Hypergeometric distribution is\n\n        .. math:: P(x) = \\frac{\\binom{m}{n}\\binom{N-m}{n-x}}{\\binom{N}{n}},\n\n        where :math:`0 \\le x \\le m` and :math:`n+m-N \\le x \\le n`\n\n        for P(x) the probability of x successes, n = ngood, m = nbad, and\n        N = number of samples.\n\n        Consider an urn with black and white marbles in it, ngood of them\n        black and nbad are white. If you draw nsample balls without\n        replacement, then the Hypergeometric distribution describes the\n        distribution of black balls in the drawn sample.\n\n        Note that this distribution is very similar to the Binomial\n        distribution, except that in this case, samples are drawn without\n        replacement, whereas in the Binomial case samples are drawn with\n        replacement (or the sample space is infinite). As the sample space\n        becomes large, this distribution approaches the Binomial.\n\n        References\n        ----------\n        .. [1] Lentner, Marvin, \"Elementary Applied Statistics\", Bogden\n               and Quigley, 1972.\n        .. [2] Weisstein, Eric W. \"Hypergeometric Distribution.\" From\n               MathWorld--A Wolfram Web Resource.\n               http://mathworld.wolfram.com/HypergeometricDistribution.html\n        .. [3] Wikipedia, \"Hypergeometric-distribution\",\n               http://en.wikipedia.org/wiki/Hypergeometric-distribution\n\n        Examples\n        --------\n        Draw samples from the distribution:\n\n        >>> ngood, nbad, nsamp = 100, 2, 10\n        # number of good, number of bad, and number of samples\n        >>> s = np.random.hypergeometric(ngood, nbad, nsamp, 1000)\n        >>> hist(s)\n        #   note that it is very unlikely to grab both bad items\n\n        Suppose you have an urn with 15 white and 15 black marbles.\n        If you pull 15 marbles at random, how likely is it that\n        12 or more of them are one color?\n\n        >>> s = np.random.hypergeometric(15, 15, 15, 100000)\n        >>> sum(s>=12)/100000. + sum(s<=3)/100000.\n        #   answer = 0.003 ... pretty unlikely!\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_hypergeometric(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_ngood = 0;
  PyObject *__pyx_v_nbad = 0;
  PyObject *__pyx_v_nsample = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_ongood;
  PyArrayObject *__pyx_v_onbad;
  PyArrayObject *__pyx_v_onsample;
  long __pyx_v_lngood;
  long __pyx_v_lnbad;
  long __pyx_v_lnsample;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  PyObject *__pyx_6 = 0;
  static char *__pyx_argnames[] = {"ngood","nbad","nsample","size",0};
  __pyx_v_size = __pyx_k56;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "OOO|O", __pyx_argnames, &__pyx_v_ngood, &__pyx_v_nbad, &__pyx_v_nsample, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_ngood);
  Py_INCREF(__pyx_v_nbad);
  Py_INCREF(__pyx_v_nsample);
  Py_INCREF(__pyx_v_size);
  __pyx_v_ongood = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_onbad = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_onsample = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2797 */
  __pyx_v_lngood = PyInt_AsLong(__pyx_v_ngood);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2798 */
  __pyx_v_lnbad = PyInt_AsLong(__pyx_v_nbad);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2799 */
  __pyx_v_lnsample = PyInt_AsLong(__pyx_v_nsample);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2800 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2801 */
    __pyx_2 = PyInt_FromLong(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2801; goto __pyx_L1;}
    if (PyObject_Cmp(__pyx_v_ngood, __pyx_2, &__pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2801; goto __pyx_L1;}
    __pyx_1 = __pyx_1 < 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2802; goto __pyx_L1;}
      Py_INCREF(__pyx_k154p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k154p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2802; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2802; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2803 */
    __pyx_2 = PyInt_FromLong(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2803; goto __pyx_L1;}
    if (PyObject_Cmp(__pyx_v_nbad, __pyx_2, &__pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2803; goto __pyx_L1;}
    __pyx_1 = __pyx_1 < 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    if (__pyx_1) {
      __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2804; goto __pyx_L1;}
      Py_INCREF(__pyx_k155p);
      PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k155p);
      __pyx_2 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2804; goto __pyx_L1;}
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      __Pyx_Raise(__pyx_2, 0, 0);
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2804; goto __pyx_L1;}
      goto __pyx_L4;
    }
    __pyx_L4:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2805 */
    __pyx_3 = PyInt_FromLong(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2805; goto __pyx_L1;}
    if (PyObject_Cmp(__pyx_v_nsample, __pyx_3, &__pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2805; goto __pyx_L1;}
    __pyx_1 = __pyx_1 < 0;
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2806; goto __pyx_L1;}
      Py_INCREF(__pyx_k156p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k156p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2806; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2806; goto __pyx_L1;}
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2807 */
    __pyx_2 = PyNumber_Add(__pyx_v_ngood, __pyx_v_nbad); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2807; goto __pyx_L1;}
    if (PyObject_Cmp(__pyx_2, __pyx_v_nsample, &__pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2807; goto __pyx_L1;}
    __pyx_1 = __pyx_1 < 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    if (__pyx_1) {
      __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2808; goto __pyx_L1;}
      Py_INCREF(__pyx_k157p);
      PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k157p);
      __pyx_2 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2808; goto __pyx_L1;}
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      __Pyx_Raise(__pyx_2, 0, 0);
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2808; goto __pyx_L1;}
      goto __pyx_L6;
    }
    __pyx_L6:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2809 */
    __pyx_3 = __pyx_f_6mtrand_discnmN_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_hypergeometric,__pyx_v_size,__pyx_v_lngood,__pyx_v_lnbad,__pyx_v_lnsample); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2809; goto __pyx_L1;}
    __pyx_r = __pyx_3;
    __pyx_3 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2813 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2815 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_ngood,NPY_LONG,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2815; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
  Py_DECREF(((PyObject *)__pyx_v_ongood));
  __pyx_v_ongood = ((PyArrayObject *)__pyx_2);
  Py_DECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2816 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_nbad,NPY_LONG,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2816; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_onbad));
  __pyx_v_onbad = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2817 */
  __pyx_2 = PyArray_FROM_OTF(__pyx_v_nsample,NPY_LONG,NPY_ALIGNED); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2817; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
  Py_DECREF(((PyObject *)__pyx_v_onsample));
  __pyx_v_onsample = ((PyArrayObject *)__pyx_2);
  Py_DECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2818 */
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2818; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_any); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2818; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2818; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_3, __pyx_n_less); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2818; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyInt_FromLong(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2818; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2818; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_ongood));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_ongood));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_3);
  __pyx_3 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2818; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2818; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_3);
  __pyx_3 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2818; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2818; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2819; goto __pyx_L1;}
    Py_INCREF(__pyx_k158p);
    PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k158p);
    __pyx_2 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2819; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __Pyx_Raise(__pyx_2, 0, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2819; goto __pyx_L1;}
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2820 */
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2820; goto __pyx_L1;}
  __pyx_5 = PyObject_GetAttr(__pyx_4, __pyx_n_any); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2820; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2820; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_less); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2820; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_4 = PyInt_FromLong(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2820; goto __pyx_L1;}
  __pyx_3 = PyTuple_New(2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2820; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_onbad));
  PyTuple_SET_ITEM(__pyx_3, 0, ((PyObject *)__pyx_v_onbad));
  PyTuple_SET_ITEM(__pyx_3, 1, __pyx_4);
  __pyx_4 = 0;
  __pyx_4 = PyObject_CallObject(__pyx_2, __pyx_3); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2820; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2820; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_2, 0, __pyx_4);
  __pyx_4 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_5, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2820; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_3); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2820; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  if (__pyx_1) {
    __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2821; goto __pyx_L1;}
    Py_INCREF(__pyx_k159p);
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_k159p);
    __pyx_5 = PyObject_CallObject(PyExc_ValueError, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2821; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    __Pyx_Raise(__pyx_5, 0, 0);
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2821; goto __pyx_L1;}
    goto __pyx_L8;
  }
  __pyx_L8:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2822 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2822; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_any); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2822; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2822; goto __pyx_L1;}
  __pyx_5 = PyObject_GetAttr(__pyx_4, __pyx_n_less); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2822; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_2 = PyInt_FromLong(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2822; goto __pyx_L1;}
  __pyx_4 = PyTuple_New(2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2822; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_onsample));
  PyTuple_SET_ITEM(__pyx_4, 0, ((PyObject *)__pyx_v_onsample));
  PyTuple_SET_ITEM(__pyx_4, 1, __pyx_2);
  __pyx_2 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_5, __pyx_4); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2822; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_5 = PyTuple_New(1); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2822; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_5, 0, __pyx_2);
  __pyx_2 = 0;
  __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_5); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2822; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_4); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2822; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  if (__pyx_1) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2823; goto __pyx_L1;}
    Py_INCREF(__pyx_k160p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k160p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2823; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2823; goto __pyx_L1;}
    goto __pyx_L9;
  }
  __pyx_L9:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2824 */
  __pyx_5 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2824; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_5, __pyx_n_any); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2824; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2824; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_less); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2824; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_5 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2824; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_5, __pyx_n_add); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2824; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2824; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_ongood));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_ongood));
  Py_INCREF(((PyObject *)__pyx_v_onbad));
  PyTuple_SET_ITEM(__pyx_5, 1, ((PyObject *)__pyx_v_onbad));
  __pyx_6 = PyObject_CallObject(__pyx_2, __pyx_5); if (!__pyx_6) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2824; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2824; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_2, 0, __pyx_6);
  Py_INCREF(((PyObject *)__pyx_v_onsample));
  PyTuple_SET_ITEM(__pyx_2, 1, ((PyObject *)__pyx_v_onsample));
  __pyx_6 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_3, __pyx_2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2824; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_6 = PyTuple_New(1); if (!__pyx_6) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2824; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_6, 0, __pyx_5);
  __pyx_5 = 0;
  __pyx_3 = PyObject_CallObject(__pyx_4, __pyx_6); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2824; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_6); __pyx_6 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_3); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2824; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  if (__pyx_1) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2825; goto __pyx_L1;}
    Py_INCREF(__pyx_k161p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k161p);
    __pyx_5 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2825; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_5, 0, 0);
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2825; goto __pyx_L1;}
    goto __pyx_L10;
  }
  __pyx_L10:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2826 */
  __pyx_4 = __pyx_f_6mtrand_discnmN_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_hypergeometric,__pyx_v_size,__pyx_v_ongood,__pyx_v_onbad,__pyx_v_onsample); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2826; goto __pyx_L1;}
  __pyx_r = __pyx_4;
  __pyx_4 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  Py_XDECREF(__pyx_6);
  __Pyx_AddTraceback("mtrand.RandomState.hypergeometric");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_ongood);
  Py_DECREF(__pyx_v_onbad);
  Py_DECREF(__pyx_v_onsample);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_ngood);
  Py_DECREF(__pyx_v_nbad);
  Py_DECREF(__pyx_v_nsample);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_n_greater_equal;

static PyObject *__pyx_k162p;
static PyObject *__pyx_k163p;
static PyObject *__pyx_k164p;
static PyObject *__pyx_k165p;

static char __pyx_k162[] = "p <= 0.0";
static char __pyx_k163[] = "p >= 1.0";
static char __pyx_k164[] = "p <= 0.0";
static char __pyx_k165[] = "p >= 1.0";

static PyObject *__pyx_f_6mtrand_11RandomState_logseries(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_logseries[] = "\n        logseries(p, size=None)\n\n        Draw samples from a Logarithmic Series distribution.\n\n        Samples are drawn from a Log Series distribution with specified\n        parameter, p (probability, 0 < p < 1).\n\n        Parameters\n        ----------\n        loc : float\n\n        scale : float > 0.\n\n        size : {tuple, int}\n            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then\n            ``m * n * k`` samples are drawn.\n\n        Returns\n        -------\n        samples : {ndarray, scalar}\n                  where the values are all integers in  [0, n].\n\n        See Also\n        --------\n        scipy.stats.distributions.logser : probability density function,\n            distribution or cumulative density function, etc.\n\n        Notes\n        -----\n        The probability density for the Log Series distribution is\n\n        .. math:: P(k) = \\frac{-p^k}{k \\ln(1-p)},\n\n        where p = probability.\n\n        The Log Series distribution is frequently used to represent species\n        richness and occurrence, first proposed by Fisher, Corbet, and\n        Williams in 1943 [2].  It may also be used to model the numbers of\n        occupants seen in cars [3].\n\n        References\n        ----------\n        .. [1] Buzas, Martin A.; Culver, Stephen J.,  Understanding regional\n               species diversity through the log series distribution of\n               occurrences: BIODIVERSITY RESEARCH Diversity & Distributions,\n               Volume 5, Number 5, September 1999 , pp. 187-195(9).\n        .. [2] Fisher, R.A,, A.S. Corbet, and C.B. Williams. 1943. The\n               relation between the number of species and the number of\n               individuals in a random sample of an animal population.\n               Journal of Animal Ecology, 12:42-58.\n        .. [3] D. J. Hand, F. Daly, D. Lunn, E. Ostrowski, A Handbook of Small\n               Data Sets, CRC Press, 1994.\n        .. [4] Wikipedia, \"Logarithmic-distribution\",\n               http://en.wikipedia.org/wiki/Logarithmic-distribution\n\n        Examples\n        --------\n        Draw samples from the distribution:\n\n        >>> a = .6\n        >>> s = np.random.logseries(a, 10000)\n        >>> count, bins, ignored = plt.hist(s)\n\n        #   plot against distribution\n\n        >>> def logseries(k, p):\n        ...     return -p**k/(k*log(1-p))\n        >>> plt.plot(bins, logseries(bins, a)*count.max()/\\\n            logseries(bins, a).max(),\'r\')\n        >>> plt.show()\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_logseries(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_p = 0;
  PyObject *__pyx_v_size = 0;
  PyArrayObject *__pyx_v_op;
  double __pyx_v_fp;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"p","size",0};
  __pyx_v_size = __pyx_k57;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O|O", __pyx_argnames, &__pyx_v_p, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_p);
  Py_INCREF(__pyx_v_size);
  __pyx_v_op = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2906 */
  __pyx_v_fp = PyFloat_AsDouble(__pyx_v_p);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2907 */
  __pyx_1 = (!PyErr_Occurred());
  if (__pyx_1) {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2908 */
    __pyx_1 = (__pyx_v_fp <= 0.0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2909; goto __pyx_L1;}
      Py_INCREF(__pyx_k162p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k162p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2909; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2909; goto __pyx_L1;}
      goto __pyx_L3;
    }
    __pyx_L3:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2910 */
    __pyx_1 = (__pyx_v_fp >= 1.0);
    if (__pyx_1) {
      __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2911; goto __pyx_L1;}
      Py_INCREF(__pyx_k163p);
      PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k163p);
      __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2911; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      __Pyx_Raise(__pyx_3, 0, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2911; goto __pyx_L1;}
      goto __pyx_L4;
    }
    __pyx_L4:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2912 */
    __pyx_2 = __pyx_f_6mtrand_discd_array_sc(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_logseries,__pyx_v_size,__pyx_v_fp); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2912; goto __pyx_L1;}
    __pyx_r = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2914 */
  PyErr_Clear();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2916 */
  __pyx_3 = PyArray_FROM_OTF(__pyx_v_p,NPY_DOUBLE,NPY_ALIGNED); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2916; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_3)));
  Py_DECREF(((PyObject *)__pyx_v_op));
  __pyx_v_op = ((PyArrayObject *)__pyx_3);
  Py_DECREF(__pyx_3); __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2917 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2917; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_any); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2917; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2917; goto __pyx_L1;}
  __pyx_4 = PyObject_GetAttr(__pyx_2, __pyx_n_less_equal); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2917; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = PyFloat_FromDouble(0.0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2917; goto __pyx_L1;}
  __pyx_5 = PyTuple_New(2); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2917; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_op));
  PyTuple_SET_ITEM(__pyx_5, 0, ((PyObject *)__pyx_v_op));
  PyTuple_SET_ITEM(__pyx_5, 1, __pyx_2);
  __pyx_2 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_4, __pyx_5); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2917; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2917; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_2);
  __pyx_2 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_3, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2917; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_5); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2917; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  if (__pyx_1) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2918; goto __pyx_L1;}
    Py_INCREF(__pyx_k164p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k164p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2918; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2918; goto __pyx_L1;}
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2919 */
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2919; goto __pyx_L1;}
  __pyx_5 = PyObject_GetAttr(__pyx_4, __pyx_n_any); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2919; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2919; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_greater_equal); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2919; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_4 = PyFloat_FromDouble(1.0); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2919; goto __pyx_L1;}
  __pyx_2 = PyTuple_New(2); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2919; goto __pyx_L1;}
  Py_INCREF(((PyObject *)__pyx_v_op));
  PyTuple_SET_ITEM(__pyx_2, 0, ((PyObject *)__pyx_v_op));
  PyTuple_SET_ITEM(__pyx_2, 1, __pyx_4);
  __pyx_4 = 0;
  __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2919; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2919; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_3, 0, __pyx_4);
  __pyx_4 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_5, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2919; goto __pyx_L1;}
  Py_DECREF(__pyx_5); __pyx_5 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_1 = PyObject_IsTrue(__pyx_2); if (__pyx_1 < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2919; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  if (__pyx_1) {
    __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2920; goto __pyx_L1;}
    Py_INCREF(__pyx_k165p);
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_k165p);
    __pyx_5 = PyObject_CallObject(PyExc_ValueError, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2920; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    __Pyx_Raise(__pyx_5, 0, 0);
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2920; goto __pyx_L1;}
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2921 */
  __pyx_3 = __pyx_f_6mtrand_discd_array(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,rk_logseries,__pyx_v_size,__pyx_v_op); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2921; goto __pyx_L1;}
  __pyx_r = __pyx_3;
  __pyx_3 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.logseries");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_op);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_p);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_n_array;
static PyObject *__pyx_n_shape;
static PyObject *__pyx_n_append;
static PyObject *__pyx_n_multiply;
static PyObject *__pyx_n_reduce;
static PyObject *__pyx_n_svd;
static PyObject *__pyx_n_dot;
static PyObject *__pyx_n_sqrt;

static PyObject *__pyx_k166p;
static PyObject *__pyx_k167p;
static PyObject *__pyx_k168p;
static PyObject *__pyx_k169p;

static char __pyx_k166[] = "mean must be 1 dimensional";
static char __pyx_k167[] = "cov must be 2 dimensional and square";
static char __pyx_k168[] = "mean and cov must have same length";
static char __pyx_k169[] = "numpy.dual";

static PyObject *__pyx_f_6mtrand_11RandomState_multivariate_normal(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_multivariate_normal[] = "\n        multivariate_normal(mean, cov[, size])\n\n        Draw random samples from a multivariate normal distribution.\n\n        The multivariate normal, multinormal or Gaussian distribution is a\n        generalisation of the one-dimensional normal distribution to higher\n        dimensions.\n\n        Such a distribution is specified by its mean and covariance matrix,\n        which are analogous to the mean (average or \"centre\") and variance\n        (standard deviation squared or \"width\") of the one-dimensional normal\n        distribution.\n\n        Parameters\n        ----------\n        mean : (N,) ndarray\n            Mean of the N-dimensional distribution.\n        cov : (N,N) ndarray\n            Covariance matrix of the distribution.\n        size : tuple of ints, optional\n            Given a shape of, for example, (m,n,k), m*n*k samples are\n            generated, and packed in an m-by-n-by-k arrangement.  Because each\n            sample is N-dimensional, the output shape is (m,n,k,N).  If no\n            shape is specified, a single sample is returned.\n\n        Returns\n        -------\n        out : ndarray\n            The drawn samples, arranged according to `size`.  If the\n            shape given is (m,n,...), then the shape of `out` is is\n            (m,n,...,N).\n\n            In other words, each entry ``out[i,j,...,:]`` is an N-dimensional\n            value drawn from the distribution.\n\n        Notes\n        -----\n        The mean is a coordinate in N-dimensional space, which represents the\n        location where samples are most likely to be generated.  This is\n        analogous to the peak of the bell curve for the one-dimensional or\n        univariate normal distribution.\n\n        Covariance indicates the level to which two variables vary together.\n        From the multivariate normal distribution, we draw N-dimensional\n        samples, :math:`X = [x_1, x_2, ... x_N]`.  The covariance matrix\n        element :math:`C_{ij}` is the covariance of :math:`x_i` and :math:`x_j`.\n        The element :math:`C_{ii}` is the variance of :math:`x_i` (i.e. its\n        \"spread\").\n\n        Instead of specifying the full covariance matrix, popular\n        approximations include:\n\n          - Spherical covariance (`cov` is a multiple of the identity matrix)\n          - Diagonal covariance (`cov` has non-negative elements, and only on\n            the diagonal)\n\n        This geometrical property can be seen in two dimensions by plotting\n        generated data-points:\n\n        >>> mean = [0,0]\n        >>> cov = [[1,0],[0,100]] # diagonal covariance, points lie on x or y-axis\n\n        >>> import matplotlib.pyplot as plt\n        >>> x,y = np.random.multivariate_normal(mean,cov,5000).T\n        >>> plt.plot(x,y,\'x\'); plt.axis(\'equal\'); plt.show()\n\n        Note that the covariance matrix must be non-negative definite.\n\n        References\n        ----------\n        .. [1] A. Papoulis, \"Probability, Random Variables, and Stochastic\n               Processes,\" 3rd ed., McGraw-Hill Companies, 1991\n        .. [2] R.O. Duda, P.E. Hart, and D.G. Stork, \"Pattern Classification,\"\n               2nd ed., Wiley, 2001.\n\n        Examples\n        --------\n        >>> mean = (1,2)\n        >>> cov = [[1,0],[1,0]]\n        >>> x = np.random.multivariate_normal(mean,cov,(3,3))\n        >>> x.shape\n        (3, 3, 2)\n\n        The following is probably true, given that 0.6 is roughly twice the\n        standard deviation:\n\n        >>> print list( (x[0,0,:] - mean) < 0.6 )\n        [True, True]\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_multivariate_normal(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_mean = 0;
  PyObject *__pyx_v_cov = 0;
  PyObject *__pyx_v_size = 0;
  PyObject *__pyx_v_shape;
  PyObject *__pyx_v_final_shape;
  PyObject *__pyx_v_x;
  PyObject *__pyx_v_svd;
  PyObject *__pyx_v_u;
  PyObject *__pyx_v_s;
  PyObject *__pyx_v_v;
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  int __pyx_4;
  Py_ssize_t __pyx_5;
  PyObject *__pyx_6 = 0;
  static char *__pyx_argnames[] = {"mean","cov","size",0};
  __pyx_v_size = __pyx_k58;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "OO|O", __pyx_argnames, &__pyx_v_mean, &__pyx_v_cov, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_mean);
  Py_INCREF(__pyx_v_cov);
  Py_INCREF(__pyx_v_size);
  __pyx_v_shape = Py_None; Py_INCREF(Py_None);
  __pyx_v_final_shape = Py_None; Py_INCREF(Py_None);
  __pyx_v_x = Py_None; Py_INCREF(Py_None);
  __pyx_v_svd = Py_None; Py_INCREF(Py_None);
  __pyx_v_u = Py_None; Py_INCREF(Py_None);
  __pyx_v_s = Py_None; Py_INCREF(Py_None);
  __pyx_v_v = Py_None; Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3017 */
  __pyx_1 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3017; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_1, __pyx_n_array); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3017; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  __pyx_1 = PyTuple_New(1); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3017; goto __pyx_L1;}
  Py_INCREF(__pyx_v_mean);
  PyTuple_SET_ITEM(__pyx_1, 0, __pyx_v_mean);
  __pyx_3 = PyObject_CallObject(__pyx_2, __pyx_1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3017; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  Py_DECREF(__pyx_v_mean);
  __pyx_v_mean = __pyx_3;
  __pyx_3 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3018 */
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3018; goto __pyx_L1;}
  __pyx_1 = PyObject_GetAttr(__pyx_2, __pyx_n_array); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3018; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3018; goto __pyx_L1;}
  Py_INCREF(__pyx_v_cov);
  PyTuple_SET_ITEM(__pyx_3, 0, __pyx_v_cov);
  __pyx_2 = PyObject_CallObject(__pyx_1, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3018; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_v_cov);
  __pyx_v_cov = __pyx_2;
  __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3019 */
  __pyx_4 = __pyx_v_size == Py_None;
  if (__pyx_4) {
    __pyx_1 = PyList_New(0); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3020; goto __pyx_L1;}
    Py_DECREF(__pyx_v_shape);
    __pyx_v_shape = __pyx_1;
    __pyx_1 = 0;
    goto __pyx_L2;
  }
  /*else*/ {
    Py_INCREF(__pyx_v_size);
    Py_DECREF(__pyx_v_shape);
    __pyx_v_shape = __pyx_v_size;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3023 */
  __pyx_3 = PyObject_GetAttr(__pyx_v_mean, __pyx_n_shape); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3023; goto __pyx_L1;}
  __pyx_5 = PyObject_Length(__pyx_3); if (__pyx_5 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3023; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_4 = (__pyx_5 != 1);
  if (__pyx_4) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3024; goto __pyx_L1;}
    Py_INCREF(__pyx_k166p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k166p);
    __pyx_1 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3024; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_1, 0, 0);
    Py_DECREF(__pyx_1); __pyx_1 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3024; goto __pyx_L1;}
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3025 */
  __pyx_3 = PyObject_GetAttr(__pyx_v_cov, __pyx_n_shape); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3025; goto __pyx_L1;}
  __pyx_5 = PyObject_Length(__pyx_3); if (__pyx_5 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3025; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_4 = (__pyx_5 != 2);
  if (!__pyx_4) {
    __pyx_2 = PyObject_GetAttr(__pyx_v_cov, __pyx_n_shape); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3025; goto __pyx_L1;}
    __pyx_1 = __Pyx_GetItemInt(__pyx_2, 0); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3025; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_3 = PyObject_GetAttr(__pyx_v_cov, __pyx_n_shape); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3025; goto __pyx_L1;}
    __pyx_2 = __Pyx_GetItemInt(__pyx_3, 1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3025; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    if (PyObject_Cmp(__pyx_1, __pyx_2, &__pyx_4) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3025; goto __pyx_L1;}
    __pyx_4 = __pyx_4 != 0;
    Py_DECREF(__pyx_1); __pyx_1 = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
  }
  if (__pyx_4) {
    __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3026; goto __pyx_L1;}
    Py_INCREF(__pyx_k167p);
    PyTuple_SET_ITEM(__pyx_3, 0, __pyx_k167p);
    __pyx_1 = PyObject_CallObject(PyExc_ValueError, __pyx_3); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3026; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    __Pyx_Raise(__pyx_1, 0, 0);
    Py_DECREF(__pyx_1); __pyx_1 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3026; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3027 */
  __pyx_2 = PyObject_GetAttr(__pyx_v_mean, __pyx_n_shape); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3027; goto __pyx_L1;}
  __pyx_3 = __Pyx_GetItemInt(__pyx_2, 0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3027; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_1 = PyObject_GetAttr(__pyx_v_cov, __pyx_n_shape); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3027; goto __pyx_L1;}
  __pyx_2 = __Pyx_GetItemInt(__pyx_1, 0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3027; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  if (PyObject_Cmp(__pyx_3, __pyx_2, &__pyx_4) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3027; goto __pyx_L1;}
  __pyx_4 = __pyx_4 != 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  if (__pyx_4) {
    __pyx_1 = PyTuple_New(1); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3028; goto __pyx_L1;}
    Py_INCREF(__pyx_k168p);
    PyTuple_SET_ITEM(__pyx_1, 0, __pyx_k168p);
    __pyx_3 = PyObject_CallObject(PyExc_ValueError, __pyx_1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3028; goto __pyx_L1;}
    Py_DECREF(__pyx_1); __pyx_1 = 0;
    __Pyx_Raise(__pyx_3, 0, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3028; goto __pyx_L1;}
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3030 */
  __pyx_4 = PyObject_IsInstance(__pyx_v_shape,((PyObject *)(&PyInt_Type))); if (__pyx_4 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3030; goto __pyx_L1;}
  if (__pyx_4) {
    __pyx_2 = PyList_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3031; goto __pyx_L1;}
    Py_INCREF(__pyx_v_shape);
    PyList_SET_ITEM(__pyx_2, 0, __pyx_v_shape);
    Py_DECREF(__pyx_v_shape);
    __pyx_v_shape = __pyx_2;
    __pyx_2 = 0;
    goto __pyx_L6;
  }
  __pyx_L6:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3032 */
  __pyx_1 = PySequence_GetSlice(__pyx_v_shape, 0, PY_SSIZE_T_MAX); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3032; goto __pyx_L1;}
  __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3032; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_3, 0, __pyx_1);
  __pyx_1 = 0;
  __pyx_2 = PyObject_CallObject(((PyObject *)(&PyList_Type)), __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3032; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_v_final_shape);
  __pyx_v_final_shape = __pyx_2;
  __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3033 */
  __pyx_1 = PyObject_GetAttr(__pyx_v_final_shape, __pyx_n_append); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3033; goto __pyx_L1;}
  __pyx_3 = PyObject_GetAttr(__pyx_v_mean, __pyx_n_shape); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3033; goto __pyx_L1;}
  __pyx_2 = __Pyx_GetItemInt(__pyx_3, 0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3033; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3033; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_3, 0, __pyx_2);
  __pyx_2 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_1, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3033; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3037 */
  __pyx_1 = PyObject_GetAttr(__pyx_v_self, __pyx_n_standard_normal); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3037; goto __pyx_L1;}
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3037; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_3, __pyx_n_multiply); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3037; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_reduce); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3037; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3037; goto __pyx_L1;}
  Py_INCREF(__pyx_v_final_shape);
  PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_final_shape);
  __pyx_6 = PyObject_CallObject(__pyx_3, __pyx_2); if (!__pyx_6) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3037; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3037; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_3, 0, __pyx_6);
  __pyx_6 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_1, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3037; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_v_x);
  __pyx_v_x = __pyx_2;
  __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3038 */
  __pyx_6 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_6) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3038; goto __pyx_L1;}
  __pyx_1 = PyObject_GetAttr(__pyx_6, __pyx_n_multiply); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3038; goto __pyx_L1;}
  Py_DECREF(__pyx_6); __pyx_6 = 0;
  __pyx_3 = PyObject_GetAttr(__pyx_1, __pyx_n_reduce); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3038; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  __pyx_5 = PyObject_Length(__pyx_v_final_shape); if (__pyx_5 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3038; goto __pyx_L1;}
  __pyx_2 = PySequence_GetSlice(__pyx_v_final_shape, 0, (__pyx_5 - 1)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3038; goto __pyx_L1;}
  __pyx_6 = PyTuple_New(1); if (!__pyx_6) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3038; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_6, 0, __pyx_2);
  __pyx_2 = 0;
  __pyx_1 = PyObject_CallObject(__pyx_3, __pyx_6); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3038; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_6); __pyx_6 = 0;
  __pyx_2 = PyObject_GetAttr(__pyx_v_mean, __pyx_n_shape); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3039; goto __pyx_L1;}
  __pyx_3 = __Pyx_GetItemInt(__pyx_2, 0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3039; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_6 = PyTuple_New(2); if (!__pyx_6) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3038; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_6, 0, __pyx_1);
  PyTuple_SET_ITEM(__pyx_6, 1, __pyx_3);
  __pyx_1 = 0;
  __pyx_3 = 0;
  if (PyObject_SetAttr(__pyx_v_x, __pyx_n_shape, __pyx_6) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3038; goto __pyx_L1;}
  Py_DECREF(__pyx_6); __pyx_6 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3047 */
  __pyx_2 = PyList_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3047; goto __pyx_L1;}
  Py_INCREF(__pyx_n_svd);
  PyList_SET_ITEM(__pyx_2, 0, __pyx_n_svd);
  __pyx_1 = __Pyx_Import(__pyx_k169p, __pyx_2); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3047; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_3 = PyObject_GetAttr(__pyx_1, __pyx_n_svd); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3047; goto __pyx_L1;}
  Py_DECREF(__pyx_v_svd);
  __pyx_v_svd = __pyx_3;
  __pyx_3 = 0;
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3049 */
  __pyx_6 = PyTuple_New(1); if (!__pyx_6) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3049; goto __pyx_L1;}
  Py_INCREF(__pyx_v_cov);
  PyTuple_SET_ITEM(__pyx_6, 0, __pyx_v_cov);
  __pyx_2 = PyObject_CallObject(__pyx_v_svd, __pyx_6); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3049; goto __pyx_L1;}
  Py_DECREF(__pyx_6); __pyx_6 = 0;
  __pyx_1 = PyObject_GetIter(__pyx_2); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3049; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_3 = __Pyx_UnpackItem(__pyx_1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3049; goto __pyx_L1;}
  Py_DECREF(__pyx_v_u);
  __pyx_v_u = __pyx_3;
  __pyx_3 = 0;
  __pyx_6 = __Pyx_UnpackItem(__pyx_1); if (!__pyx_6) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3049; goto __pyx_L1;}
  Py_DECREF(__pyx_v_s);
  __pyx_v_s = __pyx_6;
  __pyx_6 = 0;
  __pyx_2 = __Pyx_UnpackItem(__pyx_1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3049; goto __pyx_L1;}
  Py_DECREF(__pyx_v_v);
  __pyx_v_v = __pyx_2;
  __pyx_2 = 0;
  if (__Pyx_EndUnpack(__pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3049; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3050 */
  __pyx_3 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3050; goto __pyx_L1;}
  __pyx_6 = PyObject_GetAttr(__pyx_3, __pyx_n_dot); if (!__pyx_6) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3050; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3050; goto __pyx_L1;}
  __pyx_1 = PyObject_GetAttr(__pyx_2, __pyx_n_sqrt); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3050; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_3 = PyTuple_New(1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3050; goto __pyx_L1;}
  Py_INCREF(__pyx_v_s);
  PyTuple_SET_ITEM(__pyx_3, 0, __pyx_v_s);
  __pyx_2 = PyObject_CallObject(__pyx_1, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3050; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_1 = PyNumber_Multiply(__pyx_v_x, __pyx_2); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3050; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_3 = PyTuple_New(2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3050; goto __pyx_L1;}
  PyTuple_SET_ITEM(__pyx_3, 0, __pyx_1);
  Py_INCREF(__pyx_v_v);
  PyTuple_SET_ITEM(__pyx_3, 1, __pyx_v_v);
  __pyx_1 = 0;
  __pyx_2 = PyObject_CallObject(__pyx_6, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3050; goto __pyx_L1;}
  Py_DECREF(__pyx_6); __pyx_6 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_v_x);
  __pyx_v_x = __pyx_2;
  __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3053 */
  __pyx_1 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3053; goto __pyx_L1;}
  __pyx_6 = PyObject_GetAttr(__pyx_1, __pyx_n_add); if (!__pyx_6) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3053; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  __pyx_3 = PyTuple_New(3); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3053; goto __pyx_L1;}
  Py_INCREF(__pyx_v_mean);
  PyTuple_SET_ITEM(__pyx_3, 0, __pyx_v_mean);
  Py_INCREF(__pyx_v_x);
  PyTuple_SET_ITEM(__pyx_3, 1, __pyx_v_x);
  Py_INCREF(__pyx_v_x);
  PyTuple_SET_ITEM(__pyx_3, 2, __pyx_v_x);
  __pyx_2 = PyObject_CallObject(__pyx_6, __pyx_3); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3053; goto __pyx_L1;}
  Py_DECREF(__pyx_6); __pyx_6 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3054 */
  __pyx_1 = PyTuple_New(1); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3054; goto __pyx_L1;}
  Py_INCREF(__pyx_v_final_shape);
  PyTuple_SET_ITEM(__pyx_1, 0, __pyx_v_final_shape);
  __pyx_6 = PyObject_CallObject(((PyObject *)(&PyTuple_Type)), __pyx_1); if (!__pyx_6) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3054; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  if (PyObject_SetAttr(__pyx_v_x, __pyx_n_shape, __pyx_6) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3054; goto __pyx_L1;}
  Py_DECREF(__pyx_6); __pyx_6 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3055 */
  Py_INCREF(__pyx_v_x);
  __pyx_r = __pyx_v_x;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_6);
  __Pyx_AddTraceback("mtrand.RandomState.multivariate_normal");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_shape);
  Py_DECREF(__pyx_v_final_shape);
  Py_DECREF(__pyx_v_x);
  Py_DECREF(__pyx_v_svd);
  Py_DECREF(__pyx_v_u);
  Py_DECREF(__pyx_v_s);
  Py_DECREF(__pyx_v_v);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_mean);
  Py_DECREF(__pyx_v_cov);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_n_zeros;

static PyObject *__pyx_k171p;

static char __pyx_k171[] = "sum(pvals[:-1]) > 1.0";

static PyObject *__pyx_f_6mtrand_11RandomState_multinomial(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_multinomial[] = "\n        multinomial(n, pvals, size=None)\n\n        Draw samples from a multinomial distribution.\n\n        The multinomial distribution is a multivariate generalisation of the\n        binomial distribution.  Take an experiment with one of ``p``\n        possible outcomes.  An example of such an experiment is throwing a dice,\n        where the outcome can be 1 through 6.  Each sample drawn from the\n        distribution represents `n` such experiments.  Its values,\n        ``X_i = [X_0, X_1, ..., X_p]``, represent the number of times the outcome\n        was ``i``.\n\n        Parameters\n        ----------\n        n : int\n            Number of experiments.\n        pvals : sequence of floats, length p\n            Probabilities of each of the ``p`` different outcomes.  These\n            should sum to 1 (however, the last element is always assumed to\n            account for the remaining probability, as long as\n            ``sum(pvals[:-1]) <= 1)``.\n        size : tuple of ints\n            Given a `size` of ``(M, N, K)``, then ``M*N*K`` samples are drawn,\n            and the output shape becomes ``(M, N, K, p)``, since each sample\n            has shape ``(p,)``.\n\n        Examples\n        --------\n        Throw a dice 20 times:\n\n        >>> np.random.multinomial(20, [1/6.]*6, size=1)\n        array([[4, 1, 7, 5, 2, 1]])\n\n        It landed 4 times on 1, once on 2, etc.\n\n        Now, throw the dice 20 times, and 20 times again:\n\n        >>> np.random.multinomial(20, [1/6.]*6, size=2)\n        array([[3, 4, 3, 3, 4, 3],\n               [2, 4, 3, 4, 0, 7]])\n\n        For the first run, we threw 3 times 1, 4 times 2, etc.  For the second,\n        we threw 2 times 1, 4 times 2, etc.\n\n        A loaded dice is more likely to land on number 6:\n\n        >>> np.random.multinomial(100, [1/7.]*5)\n        array([13, 16, 13, 16, 42])\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_multinomial(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  long __pyx_v_n;
  PyObject *__pyx_v_pvals = 0;
  PyObject *__pyx_v_size = 0;
  long __pyx_v_d;
  PyArrayObject *arrayObject_parr;
  PyArrayObject *arrayObject_mnarr;
  double *__pyx_v_pix;
  long *__pyx_v_mnix;
  long __pyx_v_i;
  long __pyx_v_j;
  long __pyx_v_dn;
  double __pyx_v_Sum;
  PyObject *__pyx_v_shape;
  PyObject *__pyx_v_multin;
  PyObject *__pyx_r;
  Py_ssize_t __pyx_1;
  PyObject *__pyx_2 = 0;
  int __pyx_3;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  long __pyx_6;
  static char *__pyx_argnames[] = {"n","pvals","size",0};
  __pyx_v_size = __pyx_k59;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "lO|O", __pyx_argnames, &__pyx_v_n, &__pyx_v_pvals, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_pvals);
  Py_INCREF(__pyx_v_size);
  arrayObject_parr = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  arrayObject_mnarr = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_shape = Py_None; Py_INCREF(Py_None);
  __pyx_v_multin = Py_None; Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3116 */
  __pyx_1 = PyObject_Length(__pyx_v_pvals); if (__pyx_1 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3116; goto __pyx_L1;}
  __pyx_v_d = __pyx_1;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3117 */
  __pyx_2 = PyArray_ContiguousFromObject(__pyx_v_pvals,NPY_DOUBLE,1,1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3117; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
  Py_DECREF(((PyObject *)arrayObject_parr));
  arrayObject_parr = ((PyArrayObject *)__pyx_2);
  Py_DECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3118 */
  __pyx_v_pix = ((double *)arrayObject_parr->data);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3120 */
  __pyx_3 = (__pyx_f_6mtrand_kahan_sum(__pyx_v_pix,(__pyx_v_d - 1)) > (1.0 + 1e-12));
  if (__pyx_3) {
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3121; goto __pyx_L1;}
    Py_INCREF(__pyx_k171p);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_k171p);
    __pyx_4 = PyObject_CallObject(PyExc_ValueError, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3121; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __Pyx_Raise(__pyx_4, 0, 0);
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3121; goto __pyx_L1;}
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3123 */
  __pyx_3 = __pyx_v_size == Py_None;
  if (__pyx_3) {
    __pyx_2 = PyInt_FromLong(__pyx_v_d); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3124; goto __pyx_L1;}
    __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3124; goto __pyx_L1;}
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_2);
    __pyx_2 = 0;
    Py_DECREF(__pyx_v_shape);
    __pyx_v_shape = __pyx_4;
    __pyx_4 = 0;
    goto __pyx_L3;
  }
  __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3125; goto __pyx_L1;}
  Py_INCREF(__pyx_v_size);
  PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_size);
  __pyx_4 = PyObject_CallObject(((PyObject *)(&PyType_Type)), __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3125; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_3 = __pyx_4 == ((PyObject *)(&PyInt_Type));
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  if (__pyx_3) {
    __pyx_2 = PyInt_FromLong(__pyx_v_d); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3126; goto __pyx_L1;}
    __pyx_4 = PyTuple_New(2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3126; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_v_size);
    PyTuple_SET_ITEM(__pyx_4, 1, __pyx_2);
    __pyx_2 = 0;
    Py_DECREF(__pyx_v_shape);
    __pyx_v_shape = __pyx_4;
    __pyx_4 = 0;
    goto __pyx_L3;
  }
  /*else*/ {
    __pyx_2 = PyInt_FromLong(__pyx_v_d); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3128; goto __pyx_L1;}
    __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3128; goto __pyx_L1;}
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_2);
    __pyx_2 = 0;
    __pyx_2 = PyNumber_Add(__pyx_v_size, __pyx_4); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3128; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    Py_DECREF(__pyx_v_shape);
    __pyx_v_shape = __pyx_2;
    __pyx_2 = 0;
  }
  __pyx_L3:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3130 */
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3130; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_4, __pyx_n_zeros); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3130; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_4 = PyTuple_New(2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3130; goto __pyx_L1;}
  Py_INCREF(__pyx_v_shape);
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_v_shape);
  Py_INCREF(((PyObject *)(&PyInt_Type)));
  PyTuple_SET_ITEM(__pyx_4, 1, ((PyObject *)(&PyInt_Type)));
  __pyx_5 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3130; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_v_multin);
  __pyx_v_multin = __pyx_5;
  __pyx_5 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3131 */
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_v_multin)));
  Py_DECREF(((PyObject *)arrayObject_mnarr));
  arrayObject_mnarr = ((PyArrayObject *)__pyx_v_multin);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3132 */
  __pyx_v_mnix = ((long *)arrayObject_mnarr->data);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3133 */
  __pyx_v_i = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3134 */
  while (1) {
    __pyx_3 = (__pyx_v_i < PyArray_SIZE(arrayObject_mnarr));
    if (!__pyx_3) break;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3135 */
    __pyx_v_Sum = 1.0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3136 */
    __pyx_v_dn = __pyx_v_n;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3137 */
    __pyx_6 = (__pyx_v_d - 1);
    for (__pyx_v_j = 0; __pyx_v_j < __pyx_6; ++__pyx_v_j) {

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3138 */
      (__pyx_v_mnix[(__pyx_v_i + __pyx_v_j)]) = rk_binomial(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,__pyx_v_dn,((__pyx_v_pix[__pyx_v_j]) / __pyx_v_Sum));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3139 */
      __pyx_v_dn = (__pyx_v_dn - (__pyx_v_mnix[(__pyx_v_i + __pyx_v_j)]));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3140 */
      __pyx_3 = (__pyx_v_dn <= 0);
      if (__pyx_3) {
        goto __pyx_L7;
        goto __pyx_L8;
      }
      __pyx_L8:;

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3142 */
      __pyx_v_Sum = (__pyx_v_Sum - (__pyx_v_pix[__pyx_v_j]));
    }
    __pyx_L7:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3143 */
    __pyx_3 = (__pyx_v_dn > 0);
    if (__pyx_3) {
      (__pyx_v_mnix[((__pyx_v_i + __pyx_v_d) - 1)]) = __pyx_v_dn;
      goto __pyx_L9;
    }
    __pyx_L9:;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3146 */
    __pyx_v_i = (__pyx_v_i + __pyx_v_d);
  }

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3148 */
  Py_INCREF(__pyx_v_multin);
  __pyx_r = __pyx_v_multin;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.multinomial");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(arrayObject_parr);
  Py_DECREF(arrayObject_mnarr);
  Py_DECREF(__pyx_v_shape);
  Py_DECREF(__pyx_v_multin);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_pvals);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_f_6mtrand_11RandomState_dirichlet(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_dirichlet[] = "\n        dirichlet(alpha, size=None)\n\n        Draw samples from the Dirichlet distribution.\n\n        Draw `size` samples of dimension k from a Dirichlet distribution. A\n        Dirichlet-distributed random variable can be seen as a multivariate\n        generalization of a Beta distribution. Dirichlet pdf is the conjugate\n        prior of a multinomial in Bayesian inference.\n\n        Parameters\n        ----------\n        alpha : array\n            Parameter of the distribution (k dimension for sample of\n            dimension k).\n        size : array\n            Number of samples to draw.\n\n        Notes\n        -----\n        .. math:: X \\approx \\prod_{i=1}^{k}{x^{\\alpha_i-1}_i}\n\n        Uses the following property for computation: for each dimension,\n        draw a random sample y_i from a standard gamma generator of shape\n        `alpha_i`, then\n        :math:`X = \\frac{1}{\\sum_{i=1}^k{y_i}} (y_1, \\ldots, y_n)` is\n        Dirichlet distributed.\n\n        References\n        ----------\n        .. [1] David McKay, \"Information Theory, Inference and Learning\n               Algorithms,\" chapter 23,\n               http://www.inference.phy.cam.ac.uk/mackay/\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_dirichlet(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_alpha = 0;
  PyObject *__pyx_v_size = 0;
  long __pyx_v_k;
  long __pyx_v_totsize;
  PyArrayObject *__pyx_v_alpha_arr;
  PyArrayObject *__pyx_v_val_arr;
  double *__pyx_v_alpha_data;
  double *__pyx_v_val_data;
  long __pyx_v_i;
  long __pyx_v_j;
  double __pyx_v_acc;
  double __pyx_v_invacc;
  PyObject *__pyx_v_shape;
  PyObject *__pyx_v_diric;
  PyObject *__pyx_r;
  Py_ssize_t __pyx_1;
  PyObject *__pyx_2 = 0;
  int __pyx_3;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  static char *__pyx_argnames[] = {"alpha","size",0};
  __pyx_v_size = __pyx_k60;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O|O", __pyx_argnames, &__pyx_v_alpha, &__pyx_v_size)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_alpha);
  Py_INCREF(__pyx_v_size);
  __pyx_v_alpha_arr = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_val_arr = ((PyArrayObject *)Py_None); Py_INCREF(Py_None);
  __pyx_v_shape = Py_None; Py_INCREF(Py_None);
  __pyx_v_diric = Py_None; Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3214 */
  __pyx_1 = PyObject_Length(__pyx_v_alpha); if (__pyx_1 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3214; goto __pyx_L1;}
  __pyx_v_k = __pyx_1;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3215 */
  __pyx_2 = PyArray_ContiguousFromObject(__pyx_v_alpha,NPY_DOUBLE,1,1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3215; goto __pyx_L1;}
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_2)));
  Py_DECREF(((PyObject *)__pyx_v_alpha_arr));
  __pyx_v_alpha_arr = ((PyArrayObject *)__pyx_2);
  Py_DECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3216 */
  __pyx_v_alpha_data = ((double *)__pyx_v_alpha_arr->data);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3218 */
  __pyx_3 = __pyx_v_size == Py_None;
  if (__pyx_3) {
    __pyx_2 = PyInt_FromLong(__pyx_v_k); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3219; goto __pyx_L1;}
    __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3219; goto __pyx_L1;}
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_2);
    __pyx_2 = 0;
    Py_DECREF(__pyx_v_shape);
    __pyx_v_shape = __pyx_4;
    __pyx_4 = 0;
    goto __pyx_L2;
  }
  __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3220; goto __pyx_L1;}
  Py_INCREF(__pyx_v_size);
  PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_size);
  __pyx_4 = PyObject_CallObject(((PyObject *)(&PyType_Type)), __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3220; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_3 = __pyx_4 == ((PyObject *)(&PyInt_Type));
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  if (__pyx_3) {
    __pyx_2 = PyInt_FromLong(__pyx_v_k); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3221; goto __pyx_L1;}
    __pyx_4 = PyTuple_New(2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3221; goto __pyx_L1;}
    Py_INCREF(__pyx_v_size);
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_v_size);
    PyTuple_SET_ITEM(__pyx_4, 1, __pyx_2);
    __pyx_2 = 0;
    Py_DECREF(__pyx_v_shape);
    __pyx_v_shape = __pyx_4;
    __pyx_4 = 0;
    goto __pyx_L2;
  }
  /*else*/ {
    __pyx_2 = PyInt_FromLong(__pyx_v_k); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3223; goto __pyx_L1;}
    __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3223; goto __pyx_L1;}
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_2);
    __pyx_2 = 0;
    __pyx_2 = PyNumber_Add(__pyx_v_size, __pyx_4); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3223; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    Py_DECREF(__pyx_v_shape);
    __pyx_v_shape = __pyx_2;
    __pyx_2 = 0;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3225 */
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3225; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_4, __pyx_n_zeros); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3225; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_4 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3225; goto __pyx_L1;}
  __pyx_5 = PyObject_GetAttr(__pyx_4, __pyx_n_float64); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3225; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_4 = PyTuple_New(2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3225; goto __pyx_L1;}
  Py_INCREF(__pyx_v_shape);
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_v_shape);
  PyTuple_SET_ITEM(__pyx_4, 1, __pyx_5);
  __pyx_5 = 0;
  __pyx_5 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3225; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_v_diric);
  __pyx_v_diric = __pyx_5;
  __pyx_5 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3226 */
  Py_INCREF(((PyObject *)((PyArrayObject *)__pyx_v_diric)));
  Py_DECREF(((PyObject *)__pyx_v_val_arr));
  __pyx_v_val_arr = ((PyArrayObject *)__pyx_v_diric);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3227 */
  __pyx_v_val_data = ((double *)__pyx_v_val_arr->data);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3229 */
  __pyx_v_i = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3230 */
  __pyx_v_totsize = PyArray_SIZE(__pyx_v_val_arr);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3231 */
  while (1) {
    __pyx_3 = (__pyx_v_i < __pyx_v_totsize);
    if (!__pyx_3) break;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3232 */
    __pyx_v_acc = 0.0;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3233 */
    for (__pyx_v_j = 0; __pyx_v_j < __pyx_v_k; ++__pyx_v_j) {

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3234 */
      (__pyx_v_val_data[(__pyx_v_i + __pyx_v_j)]) = rk_standard_gamma(((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state,(__pyx_v_alpha_data[__pyx_v_j]));

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3235 */
      __pyx_v_acc = (__pyx_v_acc + (__pyx_v_val_data[(__pyx_v_i + __pyx_v_j)]));
    }

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3236 */
    __pyx_v_invacc = (1 / __pyx_v_acc);

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3237 */
    for (__pyx_v_j = 0; __pyx_v_j < __pyx_v_k; ++__pyx_v_j) {
      (__pyx_v_val_data[(__pyx_v_i + __pyx_v_j)]) = ((__pyx_v_val_data[(__pyx_v_i + __pyx_v_j)]) * __pyx_v_invacc);
    }

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3239 */
    __pyx_v_i = (__pyx_v_i + __pyx_v_k);
  }

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3241 */
  Py_INCREF(__pyx_v_diric);
  __pyx_r = __pyx_v_diric;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  __Pyx_AddTraceback("mtrand.RandomState.dirichlet");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_alpha_arr);
  Py_DECREF(__pyx_v_val_arr);
  Py_DECREF(__pyx_v_shape);
  Py_DECREF(__pyx_v_diric);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_alpha);
  Py_DECREF(__pyx_v_size);
  return __pyx_r;
}

static PyObject *__pyx_n_copy;


static PyObject *__pyx_f_6mtrand_11RandomState_shuffle(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_shuffle[] = "\n        shuffle(x)\n\n        Modify a sequence in-place by shuffling its contents.\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_shuffle(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_x = 0;
  long __pyx_v_i;
  long __pyx_v_j;
  int __pyx_v_copy;
  PyObject *__pyx_r;
  Py_ssize_t __pyx_1;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  int __pyx_5;
  static char *__pyx_argnames[] = {"x",0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O", __pyx_argnames, &__pyx_v_x)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_x);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3254 */
  __pyx_1 = PyObject_Length(__pyx_v_x); if (__pyx_1 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3254; goto __pyx_L1;}
  __pyx_v_i = (__pyx_1 - 1);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3255 */
  /*try:*/ {
    __pyx_2 = __Pyx_GetItemInt(__pyx_v_x, 0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3256; goto __pyx_L2;}
    __pyx_1 = PyObject_Length(__pyx_2); if (__pyx_1 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3256; goto __pyx_L2;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_v_j = __pyx_1;
  }
  goto __pyx_L3;
  __pyx_L2:;
  Py_XDECREF(__pyx_2); __pyx_2 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3257 */
  /*except:*/ {
    __Pyx_AddTraceback("mtrand.shuffle");
    if (__Pyx_GetException(&__pyx_2, &__pyx_3, &__pyx_4) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3257; goto __pyx_L1;}
    __pyx_v_j = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3260 */
  __pyx_5 = (__pyx_v_j == 0);
  if (__pyx_5) {
    while (1) {
      __pyx_5 = (__pyx_v_i > 0);
      if (!__pyx_5) break;

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3263 */
      __pyx_v_j = rk_interval(__pyx_v_i,((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state);

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3264 */
      __pyx_2 = __Pyx_GetItemInt(__pyx_v_x, __pyx_v_j); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3264; goto __pyx_L1;}
      __pyx_3 = __Pyx_GetItemInt(__pyx_v_x, __pyx_v_i); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3264; goto __pyx_L1;}
      if (__Pyx_SetItemInt(__pyx_v_x, __pyx_v_i, __pyx_2) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3264; goto __pyx_L1;}
      Py_DECREF(__pyx_2); __pyx_2 = 0;
      if (__Pyx_SetItemInt(__pyx_v_x, __pyx_v_j, __pyx_3) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3264; goto __pyx_L1;}
      Py_DECREF(__pyx_3); __pyx_3 = 0;

      /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3265 */
      __pyx_v_i = (__pyx_v_i - 1);
    }
    goto __pyx_L4;
  }
  /*else*/ {

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3268 */
    __pyx_4 = __Pyx_GetItemInt(__pyx_v_x, 0); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3268; goto __pyx_L1;}
    __pyx_5 = PyObject_HasAttr(__pyx_4,__pyx_n_copy); if (__pyx_5 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3268; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    __pyx_v_copy = __pyx_5;

    /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3269 */
    __pyx_5 = __pyx_v_copy;
    if (__pyx_5) {
      while (1) {
        __pyx_5 = (__pyx_v_i > 0);
        if (!__pyx_5) break;

        /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3271 */
        __pyx_v_j = rk_interval(__pyx_v_i,((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state);

        /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3272 */
        __pyx_2 = __Pyx_GetItemInt(__pyx_v_x, __pyx_v_j); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3272; goto __pyx_L1;}
        __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_copy); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3272; goto __pyx_L1;}
        Py_DECREF(__pyx_2); __pyx_2 = 0;
        __pyx_4 = PyObject_CallObject(__pyx_3, 0); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3272; goto __pyx_L1;}
        Py_DECREF(__pyx_3); __pyx_3 = 0;
        __pyx_2 = __Pyx_GetItemInt(__pyx_v_x, __pyx_v_i); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3272; goto __pyx_L1;}
        __pyx_3 = PyObject_GetAttr(__pyx_2, __pyx_n_copy); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3272; goto __pyx_L1;}
        Py_DECREF(__pyx_2); __pyx_2 = 0;
        __pyx_2 = PyObject_CallObject(__pyx_3, 0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3272; goto __pyx_L1;}
        Py_DECREF(__pyx_3); __pyx_3 = 0;
        if (__Pyx_SetItemInt(__pyx_v_x, __pyx_v_i, __pyx_4) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3272; goto __pyx_L1;}
        Py_DECREF(__pyx_4); __pyx_4 = 0;
        if (__Pyx_SetItemInt(__pyx_v_x, __pyx_v_j, __pyx_2) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3272; goto __pyx_L1;}
        Py_DECREF(__pyx_2); __pyx_2 = 0;

        /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3273 */
        __pyx_v_i = (__pyx_v_i - 1);
      }
      goto __pyx_L7;
    }
    /*else*/ {
      while (1) {
        __pyx_5 = (__pyx_v_i > 0);
        if (!__pyx_5) break;

        /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3276 */
        __pyx_v_j = rk_interval(__pyx_v_i,((struct __pyx_obj_6mtrand_RandomState *)__pyx_v_self)->internal_state);

        /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3277 */
        __pyx_3 = __Pyx_GetItemInt(__pyx_v_x, __pyx_v_j); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3277; goto __pyx_L1;}
        __pyx_4 = PySequence_GetSlice(__pyx_3, 0, PY_SSIZE_T_MAX); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3277; goto __pyx_L1;}
        Py_DECREF(__pyx_3); __pyx_3 = 0;
        __pyx_2 = __Pyx_GetItemInt(__pyx_v_x, __pyx_v_i); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3277; goto __pyx_L1;}
        __pyx_3 = PySequence_GetSlice(__pyx_2, 0, PY_SSIZE_T_MAX); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3277; goto __pyx_L1;}
        Py_DECREF(__pyx_2); __pyx_2 = 0;
        if (__Pyx_SetItemInt(__pyx_v_x, __pyx_v_i, __pyx_4) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3277; goto __pyx_L1;}
        Py_DECREF(__pyx_4); __pyx_4 = 0;
        if (__Pyx_SetItemInt(__pyx_v_x, __pyx_v_j, __pyx_3) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3277; goto __pyx_L1;}
        Py_DECREF(__pyx_3); __pyx_3 = 0;

        /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3278 */
        __pyx_v_i = (__pyx_v_i - 1);
      }
    }
    __pyx_L7:;
  }
  __pyx_L4:;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("mtrand.RandomState.shuffle");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_x);
  return __pyx_r;
}

static PyObject *__pyx_n_arange;

static PyObject *__pyx_f_6mtrand_11RandomState_permutation(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_6mtrand_11RandomState_permutation[] = "\n        permutation(x)\n\n        Randomly permute a sequence, or return a permuted range.\n\n        Parameters\n        ----------\n        x : int or array_like\n            If `x` is an integer, randomly permute ``np.arange(x)``.\n            If `x` is an array, make a copy and shuffle the elements\n            randomly.\n\n        Returns\n        -------\n        out : ndarray\n            Permuted sequence or array range.\n\n        Examples\n        --------\n        >>> np.random.permutation(10)\n        array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])\n\n        >>> np.random.permutation([1, 4, 9, 12, 15])\n        array([15,  1,  9,  4, 12])\n\n        ";
static PyObject *__pyx_f_6mtrand_11RandomState_permutation(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_x = 0;
  PyObject *__pyx_v_arr;
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  PyObject *__pyx_2 = 0;
  int __pyx_3;
  PyObject *__pyx_4 = 0;
  static char *__pyx_argnames[] = {"x",0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O", __pyx_argnames, &__pyx_v_x)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_x);
  __pyx_v_arr = Py_None; Py_INCREF(Py_None);

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3307 */
  __pyx_1 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3307; goto __pyx_L1;}
  __pyx_2 = PyObject_GetAttr(__pyx_1, __pyx_n_integer); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3307; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  __pyx_1 = PyTuple_New(2); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3307; goto __pyx_L1;}
  Py_INCREF(((PyObject *)(&PyInt_Type)));
  PyTuple_SET_ITEM(__pyx_1, 0, ((PyObject *)(&PyInt_Type)));
  PyTuple_SET_ITEM(__pyx_1, 1, __pyx_2);
  __pyx_2 = 0;
  __pyx_3 = PyObject_IsInstance(__pyx_v_x,__pyx_1); if (__pyx_3 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3307; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  if (__pyx_3) {
    __pyx_2 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3308; goto __pyx_L1;}
    __pyx_1 = PyObject_GetAttr(__pyx_2, __pyx_n_arange); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3308; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    __pyx_2 = PyTuple_New(1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3308; goto __pyx_L1;}
    Py_INCREF(__pyx_v_x);
    PyTuple_SET_ITEM(__pyx_2, 0, __pyx_v_x);
    __pyx_4 = PyObject_CallObject(__pyx_1, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3308; goto __pyx_L1;}
    Py_DECREF(__pyx_1); __pyx_1 = 0;
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_DECREF(__pyx_v_arr);
    __pyx_v_arr = __pyx_4;
    __pyx_4 = 0;
    goto __pyx_L2;
  }
  /*else*/ {
    __pyx_1 = __Pyx_GetName(__pyx_m, __pyx_n_np); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3310; goto __pyx_L1;}
    __pyx_2 = PyObject_GetAttr(__pyx_1, __pyx_n_array); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3310; goto __pyx_L1;}
    Py_DECREF(__pyx_1); __pyx_1 = 0;
    __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3310; goto __pyx_L1;}
    Py_INCREF(__pyx_v_x);
    PyTuple_SET_ITEM(__pyx_4, 0, __pyx_v_x);
    __pyx_1 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3310; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    Py_DECREF(__pyx_v_arr);
    __pyx_v_arr = __pyx_1;
    __pyx_1 = 0;
  }
  __pyx_L2:;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3311 */
  __pyx_2 = PyObject_GetAttr(__pyx_v_self, __pyx_n_shuffle); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3311; goto __pyx_L1;}
  __pyx_4 = PyTuple_New(1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3311; goto __pyx_L1;}
  Py_INCREF(__pyx_v_arr);
  PyTuple_SET_ITEM(__pyx_4, 0, __pyx_v_arr);
  __pyx_1 = PyObject_CallObject(__pyx_2, __pyx_4); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3311; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3312 */
  Py_INCREF(__pyx_v_arr);
  __pyx_r = __pyx_v_arr;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("mtrand.RandomState.permutation");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_arr);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_x);
  return __pyx_r;
}

static __Pyx_InternTabEntry __pyx_intern_tab[] = {
  {&__pyx_n_MT19937, "MT19937"},
  {&__pyx_n___RandomState_ctor, "__RandomState_ctor"},
  {&__pyx_n__rand, "_rand"},
  {&__pyx_n_add, "add"},
  {&__pyx_n_any, "any"},
  {&__pyx_n_append, "append"},
  {&__pyx_n_arange, "arange"},
  {&__pyx_n_array, "array"},
  {&__pyx_n_asarray, "asarray"},
  {&__pyx_n_beta, "beta"},
  {&__pyx_n_binomial, "binomial"},
  {&__pyx_n_bytes, "bytes"},
  {&__pyx_n_chisquare, "chisquare"},
  {&__pyx_n_copy, "copy"},
  {&__pyx_n_dirichlet, "dirichlet"},
  {&__pyx_n_dot, "dot"},
  {&__pyx_n_empty, "empty"},
  {&__pyx_n_equal, "equal"},
  {&__pyx_n_exponential, "exponential"},
  {&__pyx_n_f, "f"},
  {&__pyx_n_float64, "float64"},
  {&__pyx_n_gamma, "gamma"},
  {&__pyx_n_geometric, "geometric"},
  {&__pyx_n_get_state, "get_state"},
  {&__pyx_n_greater, "greater"},
  {&__pyx_n_greater_equal, "greater_equal"},
  {&__pyx_n_gumbel, "gumbel"},
  {&__pyx_n_hypergeometric, "hypergeometric"},
  {&__pyx_n_integer, "integer"},
  {&__pyx_n_laplace, "laplace"},
  {&__pyx_n_less, "less"},
  {&__pyx_n_less_equal, "less_equal"},
  {&__pyx_n_logistic, "logistic"},
  {&__pyx_n_lognormal, "lognormal"},
  {&__pyx_n_logseries, "logseries"},
  {&__pyx_n_multinomial, "multinomial"},
  {&__pyx_n_multiply, "multiply"},
  {&__pyx_n_multivariate_normal, "multivariate_normal"},
  {&__pyx_n_negative_binomial, "negative_binomial"},
  {&__pyx_n_noncentral_chisquare, "noncentral_chisquare"},
  {&__pyx_n_noncentral_f, "noncentral_f"},
  {&__pyx_n_normal, "normal"},
  {&__pyx_n_np, "np"},
  {&__pyx_n_numpy, "numpy"},
  {&__pyx_n_pareto, "pareto"},
  {&__pyx_n_permutation, "permutation"},
  {&__pyx_n_poisson, "poisson"},
  {&__pyx_n_power, "power"},
  {&__pyx_n_rand, "rand"},
  {&__pyx_n_randint, "randint"},
  {&__pyx_n_randn, "randn"},
  {&__pyx_n_random, "random"},
  {&__pyx_n_random_integers, "random_integers"},
  {&__pyx_n_random_sample, "random_sample"},
  {&__pyx_n_rayleigh, "rayleigh"},
  {&__pyx_n_reduce, "reduce"},
  {&__pyx_n_seed, "seed"},
  {&__pyx_n_set_state, "set_state"},
  {&__pyx_n_shape, "shape"},
  {&__pyx_n_shuffle, "shuffle"},
  {&__pyx_n_size, "size"},
  {&__pyx_n_sqrt, "sqrt"},
  {&__pyx_n_standard_cauchy, "standard_cauchy"},
  {&__pyx_n_standard_exponential, "standard_exponential"},
  {&__pyx_n_standard_gamma, "standard_gamma"},
  {&__pyx_n_standard_normal, "standard_normal"},
  {&__pyx_n_standard_t, "standard_t"},
  {&__pyx_n_subtract, "subtract"},
  {&__pyx_n_svd, "svd"},
  {&__pyx_n_triangular, "triangular"},
  {&__pyx_n_uint, "uint"},
  {&__pyx_n_uint32, "uint32"},
  {&__pyx_n_uniform, "uniform"},
  {&__pyx_n_vonmises, "vonmises"},
  {&__pyx_n_wald, "wald"},
  {&__pyx_n_weibull, "weibull"},
  {&__pyx_n_zeros, "zeros"},
  {&__pyx_n_zipf, "zipf"},
  {0, 0}
};

static __Pyx_StringTabEntry __pyx_string_tab[] = {
  {&__pyx_k61p, __pyx_k61, sizeof(__pyx_k61)},
  {&__pyx_k62p, __pyx_k62, sizeof(__pyx_k62)},
  {&__pyx_k63p, __pyx_k63, sizeof(__pyx_k63)},
  {&__pyx_k64p, __pyx_k64, sizeof(__pyx_k64)},
  {&__pyx_k65p, __pyx_k65, sizeof(__pyx_k65)},
  {&__pyx_k66p, __pyx_k66, sizeof(__pyx_k66)},
  {&__pyx_k67p, __pyx_k67, sizeof(__pyx_k67)},
  {&__pyx_k70p, __pyx_k70, sizeof(__pyx_k70)},
  {&__pyx_k71p, __pyx_k71, sizeof(__pyx_k71)},
  {&__pyx_k72p, __pyx_k72, sizeof(__pyx_k72)},
  {&__pyx_k74p, __pyx_k74, sizeof(__pyx_k74)},
  {&__pyx_k75p, __pyx_k75, sizeof(__pyx_k75)},
  {&__pyx_k76p, __pyx_k76, sizeof(__pyx_k76)},
  {&__pyx_k77p, __pyx_k77, sizeof(__pyx_k77)},
  {&__pyx_k78p, __pyx_k78, sizeof(__pyx_k78)},
  {&__pyx_k79p, __pyx_k79, sizeof(__pyx_k79)},
  {&__pyx_k80p, __pyx_k80, sizeof(__pyx_k80)},
  {&__pyx_k81p, __pyx_k81, sizeof(__pyx_k81)},
  {&__pyx_k82p, __pyx_k82, sizeof(__pyx_k82)},
  {&__pyx_k83p, __pyx_k83, sizeof(__pyx_k83)},
  {&__pyx_k84p, __pyx_k84, sizeof(__pyx_k84)},
  {&__pyx_k85p, __pyx_k85, sizeof(__pyx_k85)},
  {&__pyx_k86p, __pyx_k86, sizeof(__pyx_k86)},
  {&__pyx_k87p, __pyx_k87, sizeof(__pyx_k87)},
  {&__pyx_k88p, __pyx_k88, sizeof(__pyx_k88)},
  {&__pyx_k89p, __pyx_k89, sizeof(__pyx_k89)},
  {&__pyx_k90p, __pyx_k90, sizeof(__pyx_k90)},
  {&__pyx_k91p, __pyx_k91, sizeof(__pyx_k91)},
  {&__pyx_k92p, __pyx_k92, sizeof(__pyx_k92)},
  {&__pyx_k93p, __pyx_k93, sizeof(__pyx_k93)},
  {&__pyx_k94p, __pyx_k94, sizeof(__pyx_k94)},
  {&__pyx_k95p, __pyx_k95, sizeof(__pyx_k95)},
  {&__pyx_k96p, __pyx_k96, sizeof(__pyx_k96)},
  {&__pyx_k97p, __pyx_k97, sizeof(__pyx_k97)},
  {&__pyx_k98p, __pyx_k98, sizeof(__pyx_k98)},
  {&__pyx_k99p, __pyx_k99, sizeof(__pyx_k99)},
  {&__pyx_k100p, __pyx_k100, sizeof(__pyx_k100)},
  {&__pyx_k101p, __pyx_k101, sizeof(__pyx_k101)},
  {&__pyx_k102p, __pyx_k102, sizeof(__pyx_k102)},
  {&__pyx_k103p, __pyx_k103, sizeof(__pyx_k103)},
  {&__pyx_k104p, __pyx_k104, sizeof(__pyx_k104)},
  {&__pyx_k105p, __pyx_k105, sizeof(__pyx_k105)},
  {&__pyx_k106p, __pyx_k106, sizeof(__pyx_k106)},
  {&__pyx_k107p, __pyx_k107, sizeof(__pyx_k107)},
  {&__pyx_k108p, __pyx_k108, sizeof(__pyx_k108)},
  {&__pyx_k109p, __pyx_k109, sizeof(__pyx_k109)},
  {&__pyx_k110p, __pyx_k110, sizeof(__pyx_k110)},
  {&__pyx_k111p, __pyx_k111, sizeof(__pyx_k111)},
  {&__pyx_k112p, __pyx_k112, sizeof(__pyx_k112)},
  {&__pyx_k113p, __pyx_k113, sizeof(__pyx_k113)},
  {&__pyx_k114p, __pyx_k114, sizeof(__pyx_k114)},
  {&__pyx_k115p, __pyx_k115, sizeof(__pyx_k115)},
  {&__pyx_k116p, __pyx_k116, sizeof(__pyx_k116)},
  {&__pyx_k117p, __pyx_k117, sizeof(__pyx_k117)},
  {&__pyx_k118p, __pyx_k118, sizeof(__pyx_k118)},
  {&__pyx_k119p, __pyx_k119, sizeof(__pyx_k119)},
  {&__pyx_k120p, __pyx_k120, sizeof(__pyx_k120)},
  {&__pyx_k121p, __pyx_k121, sizeof(__pyx_k121)},
  {&__pyx_k122p, __pyx_k122, sizeof(__pyx_k122)},
  {&__pyx_k123p, __pyx_k123, sizeof(__pyx_k123)},
  {&__pyx_k124p, __pyx_k124, sizeof(__pyx_k124)},
  {&__pyx_k125p, __pyx_k125, sizeof(__pyx_k125)},
  {&__pyx_k126p, __pyx_k126, sizeof(__pyx_k126)},
  {&__pyx_k127p, __pyx_k127, sizeof(__pyx_k127)},
  {&__pyx_k128p, __pyx_k128, sizeof(__pyx_k128)},
  {&__pyx_k129p, __pyx_k129, sizeof(__pyx_k129)},
  {&__pyx_k130p, __pyx_k130, sizeof(__pyx_k130)},
  {&__pyx_k131p, __pyx_k131, sizeof(__pyx_k131)},
  {&__pyx_k132p, __pyx_k132, sizeof(__pyx_k132)},
  {&__pyx_k133p, __pyx_k133, sizeof(__pyx_k133)},
  {&__pyx_k134p, __pyx_k134, sizeof(__pyx_k134)},
  {&__pyx_k135p, __pyx_k135, sizeof(__pyx_k135)},
  {&__pyx_k136p, __pyx_k136, sizeof(__pyx_k136)},
  {&__pyx_k137p, __pyx_k137, sizeof(__pyx_k137)},
  {&__pyx_k138p, __pyx_k138, sizeof(__pyx_k138)},
  {&__pyx_k139p, __pyx_k139, sizeof(__pyx_k139)},
  {&__pyx_k140p, __pyx_k140, sizeof(__pyx_k140)},
  {&__pyx_k141p, __pyx_k141, sizeof(__pyx_k141)},
  {&__pyx_k142p, __pyx_k142, sizeof(__pyx_k142)},
  {&__pyx_k143p, __pyx_k143, sizeof(__pyx_k143)},
  {&__pyx_k144p, __pyx_k144, sizeof(__pyx_k144)},
  {&__pyx_k145p, __pyx_k145, sizeof(__pyx_k145)},
  {&__pyx_k146p, __pyx_k146, sizeof(__pyx_k146)},
  {&__pyx_k147p, __pyx_k147, sizeof(__pyx_k147)},
  {&__pyx_k148p, __pyx_k148, sizeof(__pyx_k148)},
  {&__pyx_k149p, __pyx_k149, sizeof(__pyx_k149)},
  {&__pyx_k150p, __pyx_k150, sizeof(__pyx_k150)},
  {&__pyx_k151p, __pyx_k151, sizeof(__pyx_k151)},
  {&__pyx_k152p, __pyx_k152, sizeof(__pyx_k152)},
  {&__pyx_k153p, __pyx_k153, sizeof(__pyx_k153)},
  {&__pyx_k154p, __pyx_k154, sizeof(__pyx_k154)},
  {&__pyx_k155p, __pyx_k155, sizeof(__pyx_k155)},
  {&__pyx_k156p, __pyx_k156, sizeof(__pyx_k156)},
  {&__pyx_k157p, __pyx_k157, sizeof(__pyx_k157)},
  {&__pyx_k158p, __pyx_k158, sizeof(__pyx_k158)},
  {&__pyx_k159p, __pyx_k159, sizeof(__pyx_k159)},
  {&__pyx_k160p, __pyx_k160, sizeof(__pyx_k160)},
  {&__pyx_k161p, __pyx_k161, sizeof(__pyx_k161)},
  {&__pyx_k162p, __pyx_k162, sizeof(__pyx_k162)},
  {&__pyx_k163p, __pyx_k163, sizeof(__pyx_k163)},
  {&__pyx_k164p, __pyx_k164, sizeof(__pyx_k164)},
  {&__pyx_k165p, __pyx_k165, sizeof(__pyx_k165)},
  {&__pyx_k166p, __pyx_k166, sizeof(__pyx_k166)},
  {&__pyx_k167p, __pyx_k167, sizeof(__pyx_k167)},
  {&__pyx_k168p, __pyx_k168, sizeof(__pyx_k168)},
  {&__pyx_k169p, __pyx_k169, sizeof(__pyx_k169)},
  {&__pyx_k171p, __pyx_k171, sizeof(__pyx_k171)},
  {0, 0, 0}
};

static PyObject *__pyx_tp_new_6mtrand_RandomState(PyTypeObject *t, PyObject *a, PyObject *k) {
  PyObject *o = (*t->tp_alloc)(t, 0);
  if (!o) return 0;
  return o;
}

static void __pyx_tp_dealloc_6mtrand_RandomState(PyObject *o) {
  {
    PyObject *etype, *eval, *etb;
    PyErr_Fetch(&etype, &eval, &etb);
    ++o->ob_refcnt;
    __pyx_f_6mtrand_11RandomState___dealloc__(o);
    if (PyErr_Occurred()) PyErr_WriteUnraisable(o);
    --o->ob_refcnt;
    PyErr_Restore(etype, eval, etb);
  }
  (*o->ob_type->tp_free)(o);
}

static struct PyMethodDef __pyx_methods_6mtrand_RandomState[] = {
  {"seed", (PyCFunction)__pyx_f_6mtrand_11RandomState_seed, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_seed},
  {"get_state", (PyCFunction)__pyx_f_6mtrand_11RandomState_get_state, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_get_state},
  {"set_state", (PyCFunction)__pyx_f_6mtrand_11RandomState_set_state, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_set_state},
  {"__getstate__", (PyCFunction)__pyx_f_6mtrand_11RandomState___getstate__, METH_VARARGS|METH_KEYWORDS, 0},
  {"__setstate__", (PyCFunction)__pyx_f_6mtrand_11RandomState___setstate__, METH_VARARGS|METH_KEYWORDS, 0},
  {"__reduce__", (PyCFunction)__pyx_f_6mtrand_11RandomState___reduce__, METH_VARARGS|METH_KEYWORDS, 0},
  {"random_sample", (PyCFunction)__pyx_f_6mtrand_11RandomState_random_sample, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_random_sample},
  {"tomaxint", (PyCFunction)__pyx_f_6mtrand_11RandomState_tomaxint, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_tomaxint},
  {"randint", (PyCFunction)__pyx_f_6mtrand_11RandomState_randint, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_randint},
  {"bytes", (PyCFunction)__pyx_f_6mtrand_11RandomState_bytes, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_bytes},
  {"uniform", (PyCFunction)__pyx_f_6mtrand_11RandomState_uniform, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_uniform},
  {"rand", (PyCFunction)__pyx_f_6mtrand_11RandomState_rand, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_rand},
  {"randn", (PyCFunction)__pyx_f_6mtrand_11RandomState_randn, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_randn},
  {"random_integers", (PyCFunction)__pyx_f_6mtrand_11RandomState_random_integers, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_random_integers},
  {"standard_normal", (PyCFunction)__pyx_f_6mtrand_11RandomState_standard_normal, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_standard_normal},
  {"normal", (PyCFunction)__pyx_f_6mtrand_11RandomState_normal, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_normal},
  {"beta", (PyCFunction)__pyx_f_6mtrand_11RandomState_beta, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_beta},
  {"exponential", (PyCFunction)__pyx_f_6mtrand_11RandomState_exponential, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_exponential},
  {"standard_exponential", (PyCFunction)__pyx_f_6mtrand_11RandomState_standard_exponential, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_standard_exponential},
  {"standard_gamma", (PyCFunction)__pyx_f_6mtrand_11RandomState_standard_gamma, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_standard_gamma},
  {"gamma", (PyCFunction)__pyx_f_6mtrand_11RandomState_gamma, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_gamma},
  {"f", (PyCFunction)__pyx_f_6mtrand_11RandomState_f, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_f},
  {"noncentral_f", (PyCFunction)__pyx_f_6mtrand_11RandomState_noncentral_f, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_noncentral_f},
  {"chisquare", (PyCFunction)__pyx_f_6mtrand_11RandomState_chisquare, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_chisquare},
  {"noncentral_chisquare", (PyCFunction)__pyx_f_6mtrand_11RandomState_noncentral_chisquare, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_noncentral_chisquare},
  {"standard_cauchy", (PyCFunction)__pyx_f_6mtrand_11RandomState_standard_cauchy, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_standard_cauchy},
  {"standard_t", (PyCFunction)__pyx_f_6mtrand_11RandomState_standard_t, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_standard_t},
  {"vonmises", (PyCFunction)__pyx_f_6mtrand_11RandomState_vonmises, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_vonmises},
  {"pareto", (PyCFunction)__pyx_f_6mtrand_11RandomState_pareto, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_pareto},
  {"weibull", (PyCFunction)__pyx_f_6mtrand_11RandomState_weibull, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_weibull},
  {"power", (PyCFunction)__pyx_f_6mtrand_11RandomState_power, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_power},
  {"laplace", (PyCFunction)__pyx_f_6mtrand_11RandomState_laplace, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_laplace},
  {"gumbel", (PyCFunction)__pyx_f_6mtrand_11RandomState_gumbel, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_gumbel},
  {"logistic", (PyCFunction)__pyx_f_6mtrand_11RandomState_logistic, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_logistic},
  {"lognormal", (PyCFunction)__pyx_f_6mtrand_11RandomState_lognormal, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_lognormal},
  {"rayleigh", (PyCFunction)__pyx_f_6mtrand_11RandomState_rayleigh, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_rayleigh},
  {"wald", (PyCFunction)__pyx_f_6mtrand_11RandomState_wald, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_wald},
  {"triangular", (PyCFunction)__pyx_f_6mtrand_11RandomState_triangular, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_triangular},
  {"binomial", (PyCFunction)__pyx_f_6mtrand_11RandomState_binomial, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_binomial},
  {"negative_binomial", (PyCFunction)__pyx_f_6mtrand_11RandomState_negative_binomial, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_negative_binomial},
  {"poisson", (PyCFunction)__pyx_f_6mtrand_11RandomState_poisson, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_poisson},
  {"zipf", (PyCFunction)__pyx_f_6mtrand_11RandomState_zipf, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_zipf},
  {"geometric", (PyCFunction)__pyx_f_6mtrand_11RandomState_geometric, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_geometric},
  {"hypergeometric", (PyCFunction)__pyx_f_6mtrand_11RandomState_hypergeometric, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_hypergeometric},
  {"logseries", (PyCFunction)__pyx_f_6mtrand_11RandomState_logseries, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_logseries},
  {"multivariate_normal", (PyCFunction)__pyx_f_6mtrand_11RandomState_multivariate_normal, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_multivariate_normal},
  {"multinomial", (PyCFunction)__pyx_f_6mtrand_11RandomState_multinomial, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_multinomial},
  {"dirichlet", (PyCFunction)__pyx_f_6mtrand_11RandomState_dirichlet, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_dirichlet},
  {"shuffle", (PyCFunction)__pyx_f_6mtrand_11RandomState_shuffle, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_shuffle},
  {"permutation", (PyCFunction)__pyx_f_6mtrand_11RandomState_permutation, METH_VARARGS|METH_KEYWORDS, __pyx_doc_6mtrand_11RandomState_permutation},
  {0, 0, 0, 0}
};

static PyNumberMethods __pyx_tp_as_number_RandomState = {
  0, /*nb_add*/
  0, /*nb_subtract*/
  0, /*nb_multiply*/
  0, /*nb_divide*/
  0, /*nb_remainder*/
  0, /*nb_divmod*/
  0, /*nb_power*/
  0, /*nb_negative*/
  0, /*nb_positive*/
  0, /*nb_absolute*/
  0, /*nb_nonzero*/
  0, /*nb_invert*/
  0, /*nb_lshift*/
  0, /*nb_rshift*/
  0, /*nb_and*/
  0, /*nb_xor*/
  0, /*nb_or*/
  0, /*nb_coerce*/
  0, /*nb_int*/
  0, /*nb_long*/
  0, /*nb_float*/
  0, /*nb_oct*/
  0, /*nb_hex*/
  0, /*nb_inplace_add*/
  0, /*nb_inplace_subtract*/
  0, /*nb_inplace_multiply*/
  0, /*nb_inplace_divide*/
  0, /*nb_inplace_remainder*/
  0, /*nb_inplace_power*/
  0, /*nb_inplace_lshift*/
  0, /*nb_inplace_rshift*/
  0, /*nb_inplace_and*/
  0, /*nb_inplace_xor*/
  0, /*nb_inplace_or*/
  0, /*nb_floor_divide*/
  0, /*nb_true_divide*/
  0, /*nb_inplace_floor_divide*/
  0, /*nb_inplace_true_divide*/
  #if Py_TPFLAGS_DEFAULT & Py_TPFLAGS_HAVE_INDEX
  0, /*nb_index*/
  #endif
};

static PySequenceMethods __pyx_tp_as_sequence_RandomState = {
  0, /*sq_length*/
  0, /*sq_concat*/
  0, /*sq_repeat*/
  0, /*sq_item*/
  0, /*sq_slice*/
  0, /*sq_ass_item*/
  0, /*sq_ass_slice*/
  0, /*sq_contains*/
  0, /*sq_inplace_concat*/
  0, /*sq_inplace_repeat*/
};

static PyMappingMethods __pyx_tp_as_mapping_RandomState = {
  0, /*mp_length*/
  0, /*mp_subscript*/
  0, /*mp_ass_subscript*/
};

static PyBufferProcs __pyx_tp_as_buffer_RandomState = {
  0, /*bf_getreadbuffer*/
  0, /*bf_getwritebuffer*/
  0, /*bf_getsegcount*/
  0, /*bf_getcharbuffer*/
};

PyTypeObject __pyx_type_6mtrand_RandomState = {
  PyObject_HEAD_INIT(0)
  0, /*ob_size*/
  "mtrand.RandomState", /*tp_name*/
  sizeof(struct __pyx_obj_6mtrand_RandomState), /*tp_basicsize*/
  0, /*tp_itemsize*/
  __pyx_tp_dealloc_6mtrand_RandomState, /*tp_dealloc*/
  0, /*tp_print*/
  0, /*tp_getattr*/
  0, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  &__pyx_tp_as_number_RandomState, /*tp_as_number*/
  &__pyx_tp_as_sequence_RandomState, /*tp_as_sequence*/
  &__pyx_tp_as_mapping_RandomState, /*tp_as_mapping*/
  0, /*tp_hash*/
  0, /*tp_call*/
  0, /*tp_str*/
  0, /*tp_getattro*/
  0, /*tp_setattro*/
  &__pyx_tp_as_buffer_RandomState, /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT|Py_TPFLAGS_CHECKTYPES|Py_TPFLAGS_BASETYPE, /*tp_flags*/
  "\n    RandomState(seed=None)\n\n    Container for the Mersenne Twister PRNG.\n\n    `RandomState` exposes a number of methods for generating random numbers\n    drawn from a variety of probability distributions. In addition to the\n    distribution-specific arguments, each method takes a keyword argument\n    `size` that defaults to ``None``. If `size` is ``None``, then a single\n    value is generated and returned. If `size` is an integer, then a 1-D\n    numpy array filled with generated values is returned. If size is a tuple,\n    then a numpy array with that shape is filled and returned.\n\n    Parameters\n    ----------\n    seed : array_like, int, optional\n        Random seed initializing the PRNG.\n        Can be an integer, an array (or other sequence) of integers of\n        any length, or ``None``.\n        If `seed` is ``None``, then `RandomState` will try to read data from\n        ``/dev/urandom`` (or the Windows analogue) if available or seed from\n        the clock otherwise.\n\n    ", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  __pyx_methods_6mtrand_RandomState, /*tp_methods*/
  0, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  __pyx_f_6mtrand_11RandomState___init__, /*tp_init*/
  0, /*tp_alloc*/
  __pyx_tp_new_6mtrand_RandomState, /*tp_new*/
  0, /*tp_free*/
  0, /*tp_is_gc*/
  0, /*tp_bases*/
  0, /*tp_mro*/
  0, /*tp_cache*/
  0, /*tp_subclasses*/
  0, /*tp_weaklist*/
};

static struct PyMethodDef __pyx_methods[] = {
  {0, 0, 0, 0}
};

static void __pyx_init_filenames(void); /*proto*/

PyMODINIT_FUNC initmtrand(void); /*proto*/
PyMODINIT_FUNC initmtrand(void) {
  PyObject *__pyx_1 = 0;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  PyObject *__pyx_6 = 0;
  PyObject *__pyx_7 = 0;
  PyObject *__pyx_8 = 0;
  PyObject *__pyx_9 = 0;
  PyObject *__pyx_10 = 0;
  PyObject *__pyx_11 = 0;
  PyObject *__pyx_12 = 0;
  PyObject *__pyx_13 = 0;
  PyObject *__pyx_14 = 0;
  PyObject *__pyx_15 = 0;
  PyObject *__pyx_16 = 0;
  PyObject *__pyx_17 = 0;
  PyObject *__pyx_18 = 0;
  __pyx_init_filenames();
  __pyx_m = Py_InitModule4("mtrand", __pyx_methods, 0, 0, PYTHON_API_VERSION);
  if (!__pyx_m) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 24; goto __pyx_L1;};
  Py_INCREF(__pyx_m);
  __pyx_b = PyImport_AddModule("__builtin__");
  if (!__pyx_b) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 24; goto __pyx_L1;};
  if (PyObject_SetAttrString(__pyx_m, "__builtins__", __pyx_b) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 24; goto __pyx_L1;};
  if (__Pyx_InternStrings(__pyx_intern_tab) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 24; goto __pyx_L1;};
  if (__Pyx_InitStrings(__pyx_string_tab) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 24; goto __pyx_L1;};
  __pyx_ptype_6mtrand_dtype = __Pyx_ImportType("numpy", "dtype", sizeof(PyArray_Descr)); if (!__pyx_ptype_6mtrand_dtype) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 74; goto __pyx_L1;}
  __pyx_ptype_6mtrand_ndarray = __Pyx_ImportType("numpy", "ndarray", sizeof(PyArrayObject)); if (!__pyx_ptype_6mtrand_ndarray) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 79; goto __pyx_L1;}
  __pyx_ptype_6mtrand_flatiter = __Pyx_ImportType("numpy", "flatiter", sizeof(PyArrayIterObject)); if (!__pyx_ptype_6mtrand_flatiter) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 88; goto __pyx_L1;}
  __pyx_ptype_6mtrand_broadcast = __Pyx_ImportType("numpy", "broadcast", sizeof(PyArrayMultiIterObject)); if (!__pyx_ptype_6mtrand_broadcast) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 94; goto __pyx_L1;}
  if (PyType_Ready(&__pyx_type_6mtrand_RandomState) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 519; goto __pyx_L1;}
  if (PyObject_SetAttrString(__pyx_m, "RandomState", (PyObject *)&__pyx_type_6mtrand_RandomState) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 519; goto __pyx_L1;}
  __pyx_ptype_6mtrand_RandomState = &__pyx_type_6mtrand_RandomState;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":121 */
  import_array();

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":123 */
  __pyx_1 = __Pyx_Import(__pyx_n_numpy, 0); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 123; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_np, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 123; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":546 */
  Py_INCREF(Py_None);
  __pyx_k2 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":556 */
  Py_INCREF(Py_None);
  __pyx_k3 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":679 */
  Py_INCREF(Py_None);
  __pyx_k4 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":698 */
  Py_INCREF(Py_None);
  __pyx_k5 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":726 */
  Py_INCREF(Py_None);
  __pyx_k6 = Py_None;
  Py_INCREF(Py_None);
  __pyx_k7 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":790 */
  __pyx_1 = PyFloat_FromDouble(0.0); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 790; goto __pyx_L1;}
  __pyx_k8 = __pyx_1;
  __pyx_1 = 0;
  __pyx_2 = PyFloat_FromDouble(1.0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 790; goto __pyx_L1;}
  __pyx_k9 = __pyx_2;
  __pyx_2 = 0;
  Py_INCREF(Py_None);
  __pyx_k10 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":938 */
  Py_INCREF(Py_None);
  __pyx_k11 = Py_None;
  Py_INCREF(Py_None);
  __pyx_k12 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":953 */
  Py_INCREF(Py_None);
  __pyx_k13 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":974 */
  __pyx_3 = PyFloat_FromDouble(0.0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 974; goto __pyx_L1;}
  __pyx_k14 = __pyx_3;
  __pyx_3 = 0;
  __pyx_4 = PyFloat_FromDouble(1.0); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 974; goto __pyx_L1;}
  __pyx_k15 = __pyx_4;
  __pyx_4 = 0;
  Py_INCREF(Py_None);
  __pyx_k16 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1074 */
  Py_INCREF(Py_None);
  __pyx_k17 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1133 */
  __pyx_5 = PyFloat_FromDouble(1.0); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1133; goto __pyx_L1;}
  __pyx_k18 = __pyx_5;
  __pyx_5 = 0;
  Py_INCREF(Py_None);
  __pyx_k19 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1187 */
  Py_INCREF(Py_None);
  __pyx_k20 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1196 */
  Py_INCREF(Py_None);
  __pyx_k21 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1218 */
  __pyx_6 = PyFloat_FromDouble(1.0); if (!__pyx_6) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1218; goto __pyx_L1;}
  __pyx_k22 = __pyx_6;
  __pyx_6 = 0;
  Py_INCREF(Py_None);
  __pyx_k23 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1309 */
  Py_INCREF(Py_None);
  __pyx_k24 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1412 */
  Py_INCREF(Py_None);
  __pyx_k25 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1450 */
  Py_INCREF(Py_None);
  __pyx_k26 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1530 */
  Py_INCREF(Py_None);
  __pyx_k27 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1572 */
  Py_INCREF(Py_None);
  __pyx_k28 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1581 */
  Py_INCREF(Py_None);
  __pyx_k29 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1604 */
  Py_INCREF(Py_None);
  __pyx_k30 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1699 */
  Py_INCREF(Py_None);
  __pyx_k31 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1788 */
  Py_INCREF(Py_None);
  __pyx_k32 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1887 */
  Py_INCREF(Py_None);
  __pyx_k33 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1910 */
  __pyx_7 = PyFloat_FromDouble(0.0); if (!__pyx_7) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1910; goto __pyx_L1;}
  __pyx_k34 = __pyx_7;
  __pyx_7 = 0;
  __pyx_8 = PyFloat_FromDouble(1.0); if (!__pyx_8) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1910; goto __pyx_L1;}
  __pyx_k35 = __pyx_8;
  __pyx_8 = 0;
  Py_INCREF(Py_None);
  __pyx_k36 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":1949 */
  __pyx_9 = PyFloat_FromDouble(0.0); if (!__pyx_9) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1949; goto __pyx_L1;}
  __pyx_k37 = __pyx_9;
  __pyx_9 = 0;
  __pyx_10 = PyFloat_FromDouble(1.0); if (!__pyx_10) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1949; goto __pyx_L1;}
  __pyx_k38 = __pyx_10;
  __pyx_10 = 0;
  Py_INCREF(Py_None);
  __pyx_k39 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2073 */
  __pyx_11 = PyFloat_FromDouble(0.0); if (!__pyx_11) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2073; goto __pyx_L1;}
  __pyx_k40 = __pyx_11;
  __pyx_11 = 0;
  __pyx_12 = PyFloat_FromDouble(1.0); if (!__pyx_12) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2073; goto __pyx_L1;}
  __pyx_k41 = __pyx_12;
  __pyx_12 = 0;
  Py_INCREF(Py_None);
  __pyx_k42 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2161 */
  __pyx_13 = PyFloat_FromDouble(0.0); if (!__pyx_13) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2161; goto __pyx_L1;}
  __pyx_k43 = __pyx_13;
  __pyx_13 = 0;
  __pyx_14 = PyFloat_FromDouble(1.0); if (!__pyx_14) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2161; goto __pyx_L1;}
  __pyx_k44 = __pyx_14;
  __pyx_14 = 0;
  Py_INCREF(Py_None);
  __pyx_k45 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2292 */
  __pyx_15 = PyFloat_FromDouble(1.0); if (!__pyx_15) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2292; goto __pyx_L1;}
  __pyx_k46 = __pyx_15;
  __pyx_15 = 0;
  Py_INCREF(Py_None);
  __pyx_k47 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2316 */
  Py_INCREF(Py_None);
  __pyx_k48 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2346 */
  Py_INCREF(Py_None);
  __pyx_k49 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2385 */
  Py_INCREF(Py_None);
  __pyx_k50 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2493 */
  Py_INCREF(Py_None);
  __pyx_k51 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2530 */
  __pyx_16 = PyFloat_FromDouble(1.0); if (!__pyx_16) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2530; goto __pyx_L1;}
  __pyx_k52 = __pyx_16;
  __pyx_16 = 0;
  Py_INCREF(Py_None);
  __pyx_k53 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2552 */
  Py_INCREF(Py_None);
  __pyx_k54 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2644 */
  Py_INCREF(Py_None);
  __pyx_k55 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2710 */
  Py_INCREF(Py_None);
  __pyx_k56 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2829 */
  Py_INCREF(Py_None);
  __pyx_k57 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":2924 */
  Py_INCREF(Py_None);
  __pyx_k58 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3057 */
  Py_INCREF(Py_None);
  __pyx_k59 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3150 */
  Py_INCREF(Py_None);
  __pyx_k60 = Py_None;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3314 */
  __pyx_17 = PyObject_CallObject(((PyObject *)__pyx_ptype_6mtrand_RandomState), 0); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3314; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n__rand, __pyx_17) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3314; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3315 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3315; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_seed); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3315; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_seed, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3315; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3316 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3316; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_get_state); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3316; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_get_state, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3316; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3317 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3317; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_set_state); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3317; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_set_state, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3317; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3318 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3318; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_random_sample); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3318; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_random_sample, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3318; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3319 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3319; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_randint); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3319; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_randint, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3319; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3320 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3320; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_bytes); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3320; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_bytes, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3320; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3321 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3321; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_uniform); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3321; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_uniform, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3321; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3322 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3322; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_rand); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3322; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_rand, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3322; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3323 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3323; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_randn); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3323; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_randn, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3323; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3324 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3324; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_random_integers); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3324; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_random_integers, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3324; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3325 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3325; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_standard_normal); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3325; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_standard_normal, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3325; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3326 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3326; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_normal); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3326; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_normal, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3326; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3327 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3327; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_beta); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3327; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_beta, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3327; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3328 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3328; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_exponential); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3328; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_exponential, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3328; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3329 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3329; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_standard_exponential); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3329; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_standard_exponential, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3329; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3330 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3330; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_standard_gamma); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3330; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_standard_gamma, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3330; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3331 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3331; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_gamma); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3331; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_gamma, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3331; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3332 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3332; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_f); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3332; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_f, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3332; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3333 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3333; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_noncentral_f); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3333; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_noncentral_f, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3333; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3334 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3334; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_chisquare); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3334; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_chisquare, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3334; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3335 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3335; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_noncentral_chisquare); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3335; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_noncentral_chisquare, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3335; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3336 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3336; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_standard_cauchy); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3336; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_standard_cauchy, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3336; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3337 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3337; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_standard_t); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3337; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_standard_t, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3337; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3338 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3338; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_vonmises); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3338; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_vonmises, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3338; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3339 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3339; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_pareto); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3339; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_pareto, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3339; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3340 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3340; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_weibull); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3340; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_weibull, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3340; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3341 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3341; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_power); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3341; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_power, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3341; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3342 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3342; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_laplace); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3342; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_laplace, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3342; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3343 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3343; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_gumbel); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3343; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_gumbel, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3343; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3344 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3344; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_logistic); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3344; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_logistic, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3344; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3345 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3345; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_lognormal); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3345; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_lognormal, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3345; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3346 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3346; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_rayleigh); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3346; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_rayleigh, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3346; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3347 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3347; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_wald); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3347; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_wald, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3347; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3348 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3348; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_triangular); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3348; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_triangular, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3348; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3350 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3350; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_binomial); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3350; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_binomial, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3350; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3351 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3351; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_negative_binomial); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3351; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_negative_binomial, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3351; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3352 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3352; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_poisson); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3352; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_poisson, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3352; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3353 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3353; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_zipf); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3353; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_zipf, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3353; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3354 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3354; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_geometric); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3354; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_geometric, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3354; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3355 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3355; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_hypergeometric); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3355; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_hypergeometric, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3355; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3356 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3356; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_logseries); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3356; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_logseries, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3356; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3358 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3358; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_multivariate_normal); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3358; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_multivariate_normal, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3358; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3359 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3359; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_multinomial); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3359; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_multinomial, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3359; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3360 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3360; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_dirichlet); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3360; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_dirichlet, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3360; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3362 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3362; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_shuffle); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3362; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_shuffle, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3362; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;

  /* "/home/pauli/koodi/proj/scipy/numpy.git/numpy/random/mtrand/mtrand.pyx":3363 */
  __pyx_17 = __Pyx_GetName(__pyx_m, __pyx_n__rand); if (!__pyx_17) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3363; goto __pyx_L1;}
  __pyx_18 = PyObject_GetAttr(__pyx_17, __pyx_n_permutation); if (!__pyx_18) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3363; goto __pyx_L1;}
  Py_DECREF(__pyx_17); __pyx_17 = 0;
  if (PyObject_SetAttr(__pyx_m, __pyx_n_permutation, __pyx_18) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 3363; goto __pyx_L1;}
  Py_DECREF(__pyx_18); __pyx_18 = 0;
  return;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  Py_XDECREF(__pyx_6);
  Py_XDECREF(__pyx_7);
  Py_XDECREF(__pyx_8);
  Py_XDECREF(__pyx_9);
  Py_XDECREF(__pyx_10);
  Py_XDECREF(__pyx_11);
  Py_XDECREF(__pyx_12);
  Py_XDECREF(__pyx_13);
  Py_XDECREF(__pyx_14);
  Py_XDECREF(__pyx_15);
  Py_XDECREF(__pyx_16);
  Py_XDECREF(__pyx_17);
  Py_XDECREF(__pyx_18);
  __Pyx_AddTraceback("mtrand");
}

static char *__pyx_filenames[] = {
  "mtrand.pyx",
  "numpy.pxi",
};

/* Runtime support code */

static void __pyx_init_filenames(void) {
  __pyx_f = __pyx_filenames;
}

static int __Pyx_GetStarArgs(
    PyObject **args, 
    PyObject **kwds,
    char *kwd_list[], 
    Py_ssize_t nargs,
    PyObject **args2, 
    PyObject **kwds2,
    char rqd_kwds[])
{
    PyObject *x = 0, *args1 = 0, *kwds1 = 0;
    int i;
    char **p;
    
    if (args2)
        *args2 = 0;
    if (kwds2)
        *kwds2 = 0;
    
    if (args2) {
        args1 = PyTuple_GetSlice(*args, 0, nargs);
        if (!args1)
            goto bad;
        *args2 = PyTuple_GetSlice(*args, nargs, PyTuple_GET_SIZE(*args));
        if (!*args2)
            goto bad;
    }
    else if (PyTuple_GET_SIZE(*args) > nargs) {
        int m = nargs;
        int n = PyTuple_GET_SIZE(*args);
        PyErr_Format(PyExc_TypeError,
            "function takes at most %d positional arguments (%d given)",
                m, n);
        goto bad;
    }
    else {
        args1 = *args;
        Py_INCREF(args1);
    }
    
    if (rqd_kwds && !*kwds)
            for (i = 0, p = kwd_list; *p; i++, p++)
                if (rqd_kwds[i])
                    goto missing_kwarg;
    
    if (kwds2) {
        if (*kwds) {
            kwds1 = PyDict_New();
            if (!kwds1)
                goto bad;
            *kwds2 = PyDict_Copy(*kwds);
            if (!*kwds2)
                goto bad;
            for (i = 0, p = kwd_list; *p; i++, p++) {
                x = PyDict_GetItemString(*kwds, *p);
                if (x) {
                    if (PyDict_SetItemString(kwds1, *p, x) < 0)
                        goto bad;
                    if (PyDict_DelItemString(*kwds2, *p) < 0)
                        goto bad;
                }
                else if (rqd_kwds && rqd_kwds[i])
                    goto missing_kwarg;
            }
        }
        else {
            *kwds2 = PyDict_New();
            if (!*kwds2)
                goto bad;
        }
    }
    else {
        kwds1 = *kwds;
        Py_XINCREF(kwds1);
        if (rqd_kwds && *kwds)
            for (i = 0, p = kwd_list; *p; i++, p++)
                if (rqd_kwds[i] && !PyDict_GetItemString(*kwds, *p))
                        goto missing_kwarg;
    }
    
    *args = args1;
    *kwds = kwds1;
    return 0;
missing_kwarg:
    PyErr_Format(PyExc_TypeError,
        "required keyword argument '%s' is missing", *p);
bad:
    Py_XDECREF(args1);
    Py_XDECREF(kwds1);
    if (args2) {
        Py_XDECREF(*args2);
    }
    if (kwds2) {
        Py_XDECREF(*kwds2);
    }
    return -1;
}

static PyObject *__Pyx_Import(PyObject *name, PyObject *from_list) {
    PyObject *__import__ = 0;
    PyObject *empty_list = 0;
    PyObject *module = 0;
    PyObject *global_dict = 0;
    PyObject *empty_dict = 0;
    PyObject *list;
    __import__ = PyObject_GetAttrString(__pyx_b, "__import__");
    if (!__import__)
        goto bad;
    if (from_list)
        list = from_list;
    else {
        empty_list = PyList_New(0);
        if (!empty_list)
            goto bad;
        list = empty_list;
    }
    global_dict = PyModule_GetDict(__pyx_m);
    if (!global_dict)
        goto bad;
    empty_dict = PyDict_New();
    if (!empty_dict)
        goto bad;
    module = PyObject_CallFunction(__import__, "OOOO",
        name, global_dict, empty_dict, list);
bad:
    Py_XDECREF(empty_list);
    Py_XDECREF(__import__);
    Py_XDECREF(empty_dict);
    return module;
}

static PyObject *__Pyx_GetName(PyObject *dict, PyObject *name) {
    PyObject *result;
    result = PyObject_GetAttr(dict, name);
    if (!result)
        PyErr_SetObject(PyExc_NameError, name);
    return result;
}

static int __Pyx_SetItemInt(PyObject *o, Py_ssize_t i, PyObject *v) {
    PyTypeObject *t = o->ob_type;
    int r;
    if (t->tp_as_sequence && t->tp_as_sequence->sq_item)
        r = PySequence_SetItem(o, i, v);
    else {
        PyObject *j = PyInt_FromLong(i);
        if (!j)
            return -1;
        r = PyObject_SetItem(o, j, v);
        Py_DECREF(j);
    }
    return r;
}

static PyObject *__Pyx_GetItemInt(PyObject *o, Py_ssize_t i) {
    PyTypeObject *t = o->ob_type;
    PyObject *r;
    if (t->tp_as_sequence && t->tp_as_sequence->sq_item)
        r = PySequence_GetItem(o, i);
    else {
        PyObject *j = PyInt_FromLong(i);
        if (!j)
            return 0;
        r = PyObject_GetItem(o, j);
        Py_DECREF(j);
    }
    return r;
}

static void __Pyx_Raise(PyObject *type, PyObject *value, PyObject *tb) {
    Py_XINCREF(type);
    Py_XINCREF(value);
    Py_XINCREF(tb);
    /* First, check the traceback argument, replacing None with NULL. */
    if (tb == Py_None) {
        Py_DECREF(tb);
        tb = 0;
    }
    else if (tb != NULL && !PyTraceBack_Check(tb)) {
        PyErr_SetString(PyExc_TypeError,
            "raise: arg 3 must be a traceback or None");
        goto raise_error;
    }
    /* Next, replace a missing value with None */
    if (value == NULL) {
        value = Py_None;
        Py_INCREF(value);
    }
    #if PY_VERSION_HEX < 0x02050000
    if (!PyClass_Check(type))
    #else
    if (!PyType_Check(type))
    #endif
    {
        /* Raising an instance.  The value should be a dummy. */
        if (value != Py_None) {
            PyErr_SetString(PyExc_TypeError,
                "instance exception may not have a separate value");
            goto raise_error;
        }
        /* Normalize to raise <class>, <instance> */
        Py_DECREF(value);
        value = type;
        #if PY_VERSION_HEX < 0x02050000
            if (PyInstance_Check(type)) {
                type = (PyObject*) ((PyInstanceObject*)type)->in_class;
                Py_INCREF(type);
            }
            else {
                PyErr_SetString(PyExc_TypeError,
                    "raise: exception must be an old-style class or instance");
                goto raise_error;
            }
        #else
            type = (PyObject*) type->ob_type;
            Py_INCREF(type);
            if (!PyType_IsSubtype((PyTypeObject *)type, (PyTypeObject *)PyExc_BaseException)) {
                PyErr_SetString(PyExc_TypeError,
                    "raise: exception class must be a subclass of BaseException");
                goto raise_error;
            }
        #endif
    }
    PyErr_Restore(type, value, tb);
    return;
raise_error:
    Py_XDECREF(value);
    Py_XDECREF(type);
    Py_XDECREF(tb);
    return;
}

static void __Pyx_UnpackError(void) {
    PyErr_SetString(PyExc_ValueError, "unpack sequence of wrong size");
}

static PyObject *__Pyx_UnpackItem(PyObject *iter) {
    PyObject *item;
    if (!(item = PyIter_Next(iter))) {
        if (!PyErr_Occurred())
            __Pyx_UnpackError();
    }
    return item;
}

static int __Pyx_EndUnpack(PyObject *iter) {
    PyObject *item;
    if ((item = PyIter_Next(iter))) {
        Py_DECREF(item);
        __Pyx_UnpackError();
        return -1;
    }
    else if (!PyErr_Occurred())
        return 0;
    else
        return -1;
}

static int __Pyx_GetException(PyObject **type, PyObject **value, PyObject **tb) {
    PyThreadState *tstate = PyThreadState_Get();
    PyErr_Fetch(type, value, tb);
    PyErr_NormalizeException(type, value, tb);
    if (PyErr_Occurred())
        goto bad;
    Py_INCREF(*type);
    Py_INCREF(*value);
    Py_INCREF(*tb);
    Py_XDECREF(tstate->exc_type);
    Py_XDECREF(tstate->exc_value);
    Py_XDECREF(tstate->exc_traceback);
    tstate->exc_type = *type;
    tstate->exc_value = *value;
    tstate->exc_traceback = *tb;
    return 0;
bad:
    Py_XDECREF(*type);
    Py_XDECREF(*value);
    Py_XDECREF(*tb);
    return -1;
}

static int __Pyx_TypeTest(PyObject *obj, PyTypeObject *type) {
    if (!type) {
        PyErr_Format(PyExc_SystemError, "Missing type object");
        return 0;
    }
    if (obj == Py_None || PyObject_TypeCheck(obj, type))
        return 1;
    PyErr_Format(PyExc_TypeError, "Cannot convert %s to %s",
        obj->ob_type->tp_name, type->tp_name);
    return 0;
}

static int __Pyx_InternStrings(__Pyx_InternTabEntry *t) {
    while (t->p) {
        *t->p = PyString_InternFromString(t->s);
        if (!*t->p)
            return -1;
        ++t;
    }
    return 0;
}

static int __Pyx_InitStrings(__Pyx_StringTabEntry *t) {
    while (t->p) {
        *t->p = PyString_FromStringAndSize(t->s, t->n - 1);
        if (!*t->p)
            return -1;
        ++t;
    }
    return 0;
}

#ifndef __PYX_HAVE_RT_ImportType
#define __PYX_HAVE_RT_ImportType
static PyTypeObject *__Pyx_ImportType(char *module_name, char *class_name, 
    long size) 
{
    PyObject *py_module = 0;
    PyObject *result = 0;
    
    py_module = __Pyx_ImportModule(module_name);
    if (!py_module)
        goto bad;
    result = PyObject_GetAttrString(py_module, class_name);
    if (!result)
        goto bad;
    if (!PyType_Check(result)) {
        PyErr_Format(PyExc_TypeError, 
            "%s.%s is not a type object",
            module_name, class_name);
        goto bad;
    }
    if (((PyTypeObject *)result)->tp_basicsize != size) {
        PyErr_Format(PyExc_ValueError, 
            "%s.%s does not appear to be the correct type object",
            module_name, class_name);
        goto bad;
    }
    return (PyTypeObject *)result;
bad:
    Py_XDECREF(result);
    return 0;
}
#endif

#ifndef __PYX_HAVE_RT_ImportModule
#define __PYX_HAVE_RT_ImportModule
static PyObject *__Pyx_ImportModule(char *name) {
    PyObject *py_name = 0;
    
    py_name = PyString_FromString(name);
    if (!py_name)
        goto bad;
    return PyImport_Import(py_name);
bad:
    Py_XDECREF(py_name);
    return 0;
}
#endif

#include "compile.h"
#include "frameobject.h"
#include "traceback.h"

static void __Pyx_AddTraceback(char *funcname) {
    PyObject *py_srcfile = 0;
    PyObject *py_funcname = 0;
    PyObject *py_globals = 0;
    PyObject *empty_tuple = 0;
    PyObject *empty_string = 0;
    PyCodeObject *py_code = 0;
    PyFrameObject *py_frame = 0;
    
    py_srcfile = PyString_FromString(__pyx_filename);
    if (!py_srcfile) goto bad;
    py_funcname = PyString_FromString(funcname);
    if (!py_funcname) goto bad;
    py_globals = PyModule_GetDict(__pyx_m);
    if (!py_globals) goto bad;
    empty_tuple = PyTuple_New(0);
    if (!empty_tuple) goto bad;
    empty_string = PyString_FromString("");
    if (!empty_string) goto bad;
    py_code = PyCode_New(
        0,            /*int argcount,*/
        0,            /*int nlocals,*/
        0,            /*int stacksize,*/
        0,            /*int flags,*/
        empty_string, /*PyObject *code,*/
        empty_tuple,  /*PyObject *consts,*/
        empty_tuple,  /*PyObject *names,*/
        empty_tuple,  /*PyObject *varnames,*/
        empty_tuple,  /*PyObject *freevars,*/
        empty_tuple,  /*PyObject *cellvars,*/
        py_srcfile,   /*PyObject *filename,*/
        py_funcname,  /*PyObject *name,*/
        __pyx_lineno,   /*int firstlineno,*/
        empty_string  /*PyObject *lnotab*/
    );
    if (!py_code) goto bad;
    py_frame = PyFrame_New(
        PyThreadState_Get(), /*PyThreadState *tstate,*/
        py_code,             /*PyCodeObject *code,*/
        py_globals,          /*PyObject *globals,*/
        0                    /*PyObject *locals*/
    );
    if (!py_frame) goto bad;
    py_frame->f_lineno = __pyx_lineno;
    PyTraceBack_Here(py_frame);
bad:
    Py_XDECREF(py_srcfile);
    Py_XDECREF(py_funcname);
    Py_XDECREF(empty_tuple);
    Py_XDECREF(empty_string);
    Py_XDECREF(py_code);
    Py_XDECREF(py_frame);
}
