#ifndef NUMPY_CORE_SRC_COMMON_RAII_UTILS_HPP_
#define NUMPY_CORE_SRC_COMMON_RAII_UTILS_HPP_

//
// Utilities for RAII management of resources.
//
// Another (and arguably clearer) name for this resource management pattern
// is "Scope-Bound Resource Management", but RAII is much more common, so we
// use the familiar acronym.
//

#include <Python.h>

// For npy_string_allocator, PyArray_StringDTypeObject, NPY_NO_EXPORT:
#include "numpy/ndarraytypes.h"

// Forward declarations not currently in a header.
// XXX Where should these be moved?
NPY_NO_EXPORT npy_string_allocator *
NpyString_acquire_allocator(const PyArray_StringDTypeObject *descr);
NPY_NO_EXPORT void
NpyString_release_allocator(npy_string_allocator *allocator);


namespace np { namespace raii {

//
// RAII for PyGILState_* API.
//
// In C++ code, use this at the beginning of a scope, e.g.
//
//     {
//         np::raii::EnsureGIL ensure_gil{};
//         [code that uses the Python C API here]
//     }
//
// instead of
//
//     PyGILState_STATE gil_state =  PyGILState_Ensure();
//     [code that uses the Python C API here]
//     PyGILState_Release(gil_state);
//
// or
//     NPY_ALLOW_C_API_DEF
//     NPY_ALLOW_C_API
//     [code that uses the Python C API here]
//     NPY_DISABLE_C_API
//
// This ensures that PyGILState_Release(gil_state) is called,  even if the
// wrapped code throws an exception or executes a return or a goto.
//
class EnsureGIL
{
    PyGILState_STATE gil_state;

public:

    EnsureGIL() {
        gil_state = PyGILState_Ensure();
    }

    ~EnsureGIL() {
        PyGILState_Release(gil_state);
    }

    EnsureGIL(const EnsureGIL&) = delete;
    EnsureGIL(EnsureGIL&& other) = delete;
    EnsureGIL& operator=(const EnsureGIL&) = delete;
    EnsureGIL& operator=(EnsureGIL&&) = delete;
};


//
// RAII for Python thread state.
//
// In C++ code, use this at the beginning of a scope, e.g.
//
//     {
//         np::raii::SaveThreadState save_thread_state{};
//         [code...]
//     }
//
// instead of
//
//     PyThreadState *thread_state = PyEval_SaveThread();
//     [code...]
//     PyEval_RestoreThread(thread_state);
//
// or
//     Py_BEGIN_ALLOW_THREADS
//     [code...]
//     Py_END_ALLOW_THREADS
//
// or
//     NPY_BEGIN_THREADS_DEF
//     NPY_BEGIN_THREADS
//     [code...]
//     NPY_END_THREADS
//
// This ensures that PyEval_RestoreThread(thread_state) is called, even
// if the wrapped code throws an exception or executes a return or a goto.
//
class SaveThreadState
{
    PyThreadState *thread_state;

public:

    SaveThreadState() {
        thread_state = PyEval_SaveThread();
    }

    ~SaveThreadState() {
        PyEval_RestoreThread(thread_state);
    }

    SaveThreadState(const SaveThreadState&) = delete;
    SaveThreadState(SaveThreadState&& other) = delete;
    SaveThreadState& operator=(const SaveThreadState&) = delete;
    SaveThreadState& operator=(SaveThreadState&&) = delete;
};


//
// RAII for npy_string_allocator.
//
// Instead of
//
//   Py_INCREF(descr);
//   npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
//   [code that uses allocator]
//   NpyString_release_allocator(allocator);
//   Py_DECREF(descr);
//
// use
//
//   {
//       np::raii::NpyStringAcquireAllocator alloc(descr);
//       [code that uses alloc.allocator()]
//   }
//
class NpyStringAcquireAllocator
{
    PyArray_StringDTypeObject *_descr;
    npy_string_allocator *_allocator;

public:

    NpyStringAcquireAllocator(PyArray_StringDTypeObject *descr) : _descr(descr) {
        Py_INCREF(_descr);
        _allocator = NpyString_acquire_allocator(_descr);
    }

    ~NpyStringAcquireAllocator() {
        NpyString_release_allocator(_allocator);
        Py_DECREF(_descr);
    }

    NpyStringAcquireAllocator(const NpyStringAcquireAllocator&) = delete;
    NpyStringAcquireAllocator(NpyStringAcquireAllocator&& other) = delete;
    NpyStringAcquireAllocator& operator=(const NpyStringAcquireAllocator&) = delete;
    NpyStringAcquireAllocator& operator=(NpyStringAcquireAllocator&&) = delete;

    npy_string_allocator *allocator() {
        return _allocator;
    }
};

}}  // namespace np { namespace raii {

#endif
