#ifndef NUMPY_CORE_SRC_COMMON_BINOP_OVERRIDE_H_
#define NUMPY_CORE_SRC_COMMON_BINOP_OVERRIDE_H_

#include <string.h>
#include <Python.h>
#include "numpy/arrayobject.h"

#include "get_attr_string.h"

/*
 * Logic for deciding when binops should return NotImplemented versus when
 * they should go ahead and call a ufunc (or similar).
 *
 * The interaction between binop methods (ndarray.__add__ and friends) and
 * ufuncs (which dispatch to __array_ufunc__) is both complicated in its own
 * right, and also has complicated historical constraints.
 *
 * In the very old days, the rules were:
 * - If the other argument has a higher __array_priority__, then return
 *   NotImplemented
 * - Otherwise, call the corresponding ufunc.
 *   - And the ufunc might return NotImplemented based on some complex
 *     criteria that I won't reproduce here.
 *
 * Ufuncs no longer return NotImplemented (except in a few marginal situations
 * which are being phased out -- see https://github.com/numpy/numpy/pull/5864)
 *
 * So as of 1.9, the effective rules were:
 * - If the other argument has a higher __array_priority__, and is *not* a
 *   subclass of ndarray, then return NotImplemented. (If it is a subclass,
 *   the regular Python rules have already given it a chance to run; so if we
 *   are running, then it means the other argument has already returned
 *   NotImplemented and is basically asking us to take care of things.)
 * - Otherwise call the corresponding ufunc.
 *
 * We would like to get rid of __array_priority__, and __array_ufunc__
 * provides a large part of a replacement for it. Once __array_ufunc__ is
 * widely available, the simplest dispatch rules that might possibly work
 * would be:
 * - Always call the corresponding ufunc.
 *
 * But:
 * - Doing this immediately would break backwards compatibility -- there's a
 *   lot of code using __array_priority__ out there.
 * - It's not at all clear whether __array_ufunc__ actually is sufficient for
 *   all use cases. (See https://github.com/numpy/numpy/issues/5844 for lots
 *   of discussion of this, and in particular
 *     https://github.com/numpy/numpy/issues/5844#issuecomment-112014014
 *   for a summary of some conclusions.) Also, python 3.6 defines a standard
 *   where setting a special-method name to None is a signal that that method
 *   cannot be used.
 *
 * So for 1.13, we are going to try the following rules.
 *
 * For binops like a.__add__(b):
 * - If b does not define __array_ufunc__, apply the legacy rule:
 *   - If not isinstance(b, a.__class__), and b.__array_priority__ is higher
 *     than a.__array_priority__, return NotImplemented
 * - If b does define __array_ufunc__ but it is None, return NotImplemented
 * - Otherwise, call the corresponding ufunc.
 *
 * For in-place operations like a.__iadd__(b)
 * - If b does not define __array_ufunc__, apply the legacy rule:
 *   - If not isinstance(b, a.__class__), and b.__array_priority__ is higher
 *     than a.__array_priority__, return NotImplemented
 * - Otherwise, call the corresponding ufunc.
 *
 * For reversed operations like b.__radd__(a) we call the corresponding ufunc.
 *
 * Rationale for __radd__: This is because by the time the reversed operation
 * is called, there are only two possibilities: The first possibility is that
 * the current class is a strict subclass of the other class. In practice, the
 * only way this will happen is if b is a strict subclass of a, and a is
 * ndarray or a subclass of ndarray, and neither a nor b has actually
 * overridden this method. In this case, Python will never call a.__add__
 * (because it's identical to b.__radd__), so we have no-one to defer to;
 * there's no reason to return NotImplemented. The second possibility is that
 * b.__add__ has already been called and returned NotImplemented. Again, in
 * this case there is no point in returning NotImplemented.
 *
 * Rationale for __iadd__: In-place operations do not take all the trouble
 * above, because if __iadd__ returns NotImplemented then Python will silently
 * convert the operation into an out-of-place operation, i.e. 'a += b' will
 * silently become 'a = a + b'. We don't want to allow this for arrays,
 * because it will create unexpected memory allocations, break views, etc.
 * However, backwards compatibility requires that we follow the rules of
 * __array_priority__ for arrays that define it. For classes that use the new
 * __array_ufunc__ mechanism we simply defer to the ufunc. That has the effect
 * that when the other array has__array_ufunc = None a TypeError will be raised.
 *
 * In the future we might change these rules further. For example, we plan to
 * eventually deprecate __array_priority__ in cases where __array_ufunc__ is
 * not present.
 */

static int
binop_should_defer(PyObject *self, PyObject *other, int inplace)
{
    /*
     * This function assumes that self.__binop__(other) is underway and
     * implements the rules described above. Python's C API is funny, and
     * makes it tricky to tell whether a given slot is called for __binop__
     * ("forward") or __rbinop__ ("reversed"). You are responsible for
     * determining this before calling this function; it only provides the
     * logic for forward binop implementations.
     */

    /*
     * NB: there's another copy of this code in
     *    numpy.ma.core.MaskedArray._delegate_binop
     * which should possibly be updated when this is.
     */

    PyObject *attr;
    double self_prio, other_prio;
    int defer;
    /*
     * attribute check is expensive for scalar operations, avoid if possible
     */
    if (other == NULL ||
        self == NULL ||
        Py_TYPE(self) == Py_TYPE(other) ||
        PyArray_CheckExact(other) ||
        PyArray_CheckAnyScalarExact(other)) {
        return 0;
    }
    /*
     * Classes with __array_ufunc__ are living in the future, and only need to
     * check whether __array_ufunc__ equals None.
     */
    attr = PyArray_LookupSpecial(other, npy_um_str_array_ufunc);
    if (attr != NULL) {
        defer = !inplace && (attr == Py_None);
        Py_DECREF(attr);
        return defer;
    }
    else if (PyErr_Occurred()) {
        PyErr_Clear(); /* TODO[gh-14801]: propagate crashes during attribute access? */
    }
    /*
     * Otherwise, we need to check for the legacy __array_priority__. But if
     * other.__class__ is a subtype of self.__class__, then it's already had
     * a chance to run, so no need to defer to it.
     */
    if(PyType_IsSubtype(Py_TYPE(other), Py_TYPE(self))) {
        return 0;
    }
    self_prio = PyArray_GetPriority((PyObject *)self, NPY_SCALAR_PRIORITY);
    other_prio = PyArray_GetPriority((PyObject *)other, NPY_SCALAR_PRIORITY);
    return self_prio < other_prio;
}

/*
 * A CPython slot like ->tp_as_number->nb_add gets called for *both* forward
 * and reversed operations. E.g.
 *   a + b
 * may call
 *   a->tp_as_number->nb_add(a, b)
 * and
 *   b + a
 * may call
 *   a->tp_as_number->nb_add(b, a)
 * and the only way to tell which is which is for a slot implementation 'f' to
 * check
 *   arg1->tp_as_number->nb_add == f
 *   arg2->tp_as_number->nb_add == f
 * If both are true, then CPython will as a special case only call the
 * operation once (i.e., it performs both the forward and reversed binops
 * simultaneously). This function is mostly intended for figuring out
 * whether we are a forward binop that might want to return NotImplemented,
 * and in the both-at-once case we never want to return NotImplemented, so in
 * that case BINOP_IS_FORWARD returns false.
 *
 * This is modeled on the checks in CPython's typeobject.c SLOT1BINFULL
 * macro.
 */
#define BINOP_IS_FORWARD(m1, m2, SLOT_NAME, test_func)  \
    (Py_TYPE(m2)->tp_as_number != NULL &&                               \
     (void*)(Py_TYPE(m2)->tp_as_number->SLOT_NAME) != (void*)(test_func))

#define BINOP_GIVE_UP_IF_NEEDED(m1, m2, slot_expr, test_func)           \
    do {                                                                \
        if (BINOP_IS_FORWARD(m1, m2, slot_expr, test_func) &&           \
            binop_should_defer((PyObject*)m1, (PyObject*)m2, 0)) {      \
            Py_INCREF(Py_NotImplemented);                               \
            return Py_NotImplemented;                                   \
        }                                                               \
    } while (0)

#define INPLACE_GIVE_UP_IF_NEEDED(m1, m2, slot_expr, test_func)         \
    do {                                                                \
        if (BINOP_IS_FORWARD(m1, m2, slot_expr, test_func) &&           \
            binop_should_defer((PyObject*)m1, (PyObject*)m2, 1)) {      \
            Py_INCREF(Py_NotImplemented);                               \
            return Py_NotImplemented;                                   \
        }                                                               \
    } while (0)

/*
 * For rich comparison operations, it's impossible to distinguish
 * between a forward comparison and a reversed/reflected
 * comparison. So we assume they are all forward. This only works because the
 * logic in binop_override_forward_binop_should_defer is essentially
 * asymmetric -- you can never have two duck-array types that each decide to
 * defer to the other.
 */
#define RICHCMP_GIVE_UP_IF_NEEDED(m1, m2)                               \
    do {                                                                \
        if (binop_should_defer((PyObject*)m1, (PyObject*)m2, 0)) {      \
            Py_INCREF(Py_NotImplemented);                               \
            return Py_NotImplemented;                                   \
        }                                                               \
    } while (0)

#endif  /* NUMPY_CORE_SRC_COMMON_BINOP_OVERRIDE_H_ */
