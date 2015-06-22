#ifndef __BINOP_OVERRIDE_H
#define __BINOP_OVERRIDE_H

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
 *   for a summary of some conclusions.)
 *
 * So for 1.10, we are going to try the following rules. a.__add__(b) will
 * be implemented as follows:
 * - If b does not define __array_ufunc__, apply the legacy rule:
 *   - If not isinstance(b, a.__class__), and b.__array_priority__ is higher
 *     than a.__array_priority__, return NotImplemented
 *   - Otherwise, fall through.
 * - If b->ob_type["__module__"].startswith("scipy.sparse."), then return
 *   NotImplemented. (Rationale: scipy.sparse defines __mul__ and np.multiply
 *   to do two totally different things. We want to grandfather this behavior
 *   in, but we don't want to support it in the long run, as per PEP
 *   465. Additionally, several versions of scipy.sparse were released with
 *   __array_ufunc__ implementations that don't match the final interface, and
 *   we don't want dense + sparse to suddenly start erroring out because
 *   dense.__add__ dispatched to a broken sparse.__array_ufunc__.)
 * - Otherwise, call the corresponding ufunc.
 *
 * For reversed operations like b.__radd__(a), and for in-place operations
 * like a.__iadd__(b), we:
 * - Call the corresponding ufunc
 *
 * Rationale for __radd__: This is because by the time the reversed operation
 * is called, there are only two possibilities: The first possibility is that
 * the current class is a strict subclass of the other class. In practice, the
 * only way this will happen is if b is a strict subclass of a, and a is
 * ndarray or a subclass of ndarray, and neither a nor b has actually
 * overridden this method. In this case, Python will never call a.__add__
 * (because it's identical to b.__radd__), so we have no-one to defer to;
 * there's no reason to return NotImplemented. The second possibility is that
 * a.__add__ has already been called and returned NotImplemented. Again, in
 * this case there is no point in returning NotImplemented.
 *
 * Rationale for __iadd__: In-place operations do not take all the trouble
 * above, because if __iadd__ returns NotImplemented then Python will silently
 * convert the operation into an out-of-place operation, i.e. 'a += b' will
 * silently become 'a = a + b'. We don't want to allow this for arrays,
 * because it will create unexpected memory allocations, break views,
 * etc.
 *
 * In the future we might change these rules further. For example, we plan to
 * eventually deprecate __array_priority__ in cases where __array_ufunc__ is
 * not present, and we might decide that we need somewhat more flexible
 * dispatch rules where the ndarray binops sometimes return NotImplemented
 * rather than always dispatching to ufuncs.
 *
 * Note that these rules are also implemented by ABCArray, so any changes here
 * should also be reflected there.
 */

static int
binop_override_has_ufunc_attr(PyObject *obj) {
    PyObject *attr;
    int result;

    /* attribute check is expensive for scalar operations, avoid if possible */
    if (PyArray_CheckExact(obj) || PyArray_CheckAnyScalarExact(obj) ||
        _is_basic_python_type(obj)) {
        return 0;
    }

    attr = PyArray_GetAttrString_SuppressException(obj, "__array_ufunc__");
    if (attr == NULL) {
        return 0;
    }
    else {
        /*
         * Pretend that non-callable __array_ufunc__ isn't there. This is an
         * escape hatch in case we want to assign some special meaning to
         * something like __array_ufunc__ = None, later on. (And can be
         * deleted if we decide we don't want to do that.) See these two
         * comments:
         *   https://github.com/numpy/numpy/issues/5844#issuecomment-105081603
         *   https://github.com/numpy/numpy/issues/5844#issuecomment-105170926
         */
        result = PyCallable_Check(attr);
        Py_DECREF(attr);
        return result;
    }
}

static int
binop_override_is_scipy_sparse(PyObject *obj) {
    PyObject *module_name = NULL;
    PyObject *bytes = NULL;
    int result = 0;
    char *contents;

    module_name = PyArray_GetAttrString_SuppressException(
        (PyObject*) Py_TYPE(obj),
        "__module__");
    if (module_name == NULL) {
        goto done;
    }
    if (PyBytes_CheckExact(module_name)) {
        contents = PyBytes_AS_STRING(module_name);
    }
#if PY_VERSION_HEX >= 0x03020000
    else if (PyUnicode_CheckExact(module_name)) {
#if (PY_VERSION_HEX >= 0x03020000) && (PY_VERSION_HEX < 0x03030000)
        /* Python 3.2: unicode, but old API */
        bytes = PyUnicode_AsLatin1String(module_name);
        if (bytes == NULL) {
            PyErr_Clear();
            goto done;
        }
        contents = PyString_AS_STRING(bytes);
#endif /* cpython == 3.2.x */
#if PY_VERSION_HEX >= 0x03030000
        /* Python 3.3+: new unicode API */
        if (PyUnicode_READY(module_name) < 0) {
            PyErr_Clear();
            goto done;
        }
        /*
         * We assume that scipy.sparse modules will always have ascii names
         */
        if (PyUnicode_KIND(module_name) != PyUnicode_1BYTE_KIND) {
            goto done;
        }
        contents = (char*) PyUnicode_1BYTE_DATA(module_name);
#endif /* cpython >= 3.3 */
    }
#endif /* cpython >= 3.2 */
    else {
        goto done;
    }
    if (strncmp("scipy.sparse", contents, 12) == 0) {
        result = 1;
    }

  done:
    Py_XDECREF(module_name);
    Py_XDECREF(bytes);
    return result;
}

static int
binop_override_forward_binop_should_defer(PyObject *self, PyObject *other)
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

    if (other == NULL ||
        self == NULL ||
        Py_TYPE(self) == Py_TYPE(other) ||
        PyArray_CheckExact(other) ||
        PyArray_CheckAnyScalar(other)) {
        /*
         * Quick cases
         */
        return 0;
    }

    /*
     * Classes with __array_ufunc__ are living in the future, and don't need
     * a check for the legacy __array_priority__. And if other.__class__ is a
     * subtype of self.__class__, then it's already had a chance to run, so no
     * need to defer to it.
     */
    if (!binop_override_has_ufunc_attr(other) &&
        !PyType_IsSubtype(Py_TYPE(other), Py_TYPE(self))) {
        double self_prio = PyArray_GetPriority((PyObject *)self,
                                               NPY_SCALAR_PRIORITY);
        double other_prio = PyArray_GetPriority((PyObject *)other,
                                                NPY_SCALAR_PRIORITY);
        if (self_prio < other_prio) {
            return 1;
        }
    }
    if (binop_override_is_scipy_sparse(other)) {
        /* Special case grandfathering in scipy.sparse */
        return 1;
    }

    return 0;
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
            binop_override_forward_binop_should_defer((PyObject*)m1, (PyObject*)m2)) { \
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
        if (binop_override_forward_binop_should_defer((PyObject*)m1, (PyObject*)m2)) { \
            Py_INCREF(Py_NotImplemented);                               \
            return Py_NotImplemented;                                   \
        }                                                               \
    } while (0)

#endif
