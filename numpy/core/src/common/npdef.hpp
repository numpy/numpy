#ifndef NUMPY_CORE_SRC_COMMON_NPDEF_HPP
#define NUMPY_CORE_SRC_COMMON_NPDEF_HPP

#if !defined(__cplusplus) || __cplusplus < 201703L
    #error "NumPy requires a compiler with at least C++17 enabled"
#endif

/// @addtogroup cpp_core_defs
/// @{

/// Whether compiler supports C++20
#if __cplusplus > 202002L
    #define NP_HAS_CPP20 1
#else
    #define NP_HAS_CPP20 0
#endif

/// Wraps `__has_builtin`
#if defined(__has_builtin)
    #define NP_HAS_BUILTIN(INTRIN) __has_builtin(INTRIN)
#else
    #define NP_HAS_BUILTIN(INTRIN) 0
#endif

/// @} cpp_core_defs

#endif // NUMPY_CORE_SRC_COMMON_NPDEF_HPP

