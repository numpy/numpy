/******************************************************************************
 * config.h           Compiler language support flags
 *
 * This file was generated automatically by the script bzconfig.
 * You should rerun bzconfig each time you switch compilers, install new
 * standard libraries, or change compiler versions.
 *
 */

 
#ifndef BZ_CONFIG_H
#define BZ_CONFIG_H
 
#define BZ_COMPILER_NAME   "g++"
#define BZ_COMPILER_OPTIONS "-ftemplate-depth-30"
#define BZ_OS_NAME         "Linux 2.2.14-5.0"
#define BZ_BZCONFIG_DATE   "Tue Apr 10 21:43:49 EDT 2001"
#define BZ_PLATFORM        "i686-pc-linux-gnu"
 
#define BZ_NAMESPACES
#define BZ_EXCEPTIONS
#define BZ_RTTI
#define BZ_MEMBER_CONSTANTS
#undef  BZ_OLD_FOR_SCOPING
#define BZ_EXPLICIT
#define BZ_MUTABLE
#define BZ_TYPENAME
#undef  BZ_NCEG_RESTRICT
#define BZ_NCEG_RESTRICT_EGCS
#define BZ_BOOL
#define BZ_CONST_CAST
#define BZ_STATIC_CAST
#define BZ_REINTERPRET_CAST
#define BZ_DYNAMIC_CAST
#define BZ_TEMPLATES
#define BZ_PARTIAL_SPECIALIZATION
#define BZ_PARTIAL_ORDERING
#define BZ_DEFAULT_TEMPLATE_PARAMETERS
#define BZ_MEMBER_TEMPLATES
#define BZ_MEMBER_TEMPLATES_OUTSIDE_CLASS
#define BZ_FULL_SPECIALIZATION_SYNTAX
#define BZ_FUNCTION_NONTYPE_PARAMETERS
#define BZ_TEMPLATE_QUALIFIED_BASE_CLASS
#define BZ_TEMPLATE_QUALIFIED_RETURN_TYPE
#define BZ_EXPLICIT_TEMPLATE_FUNCTION_QUALIFICATION
#define BZ_TEMPLATES_AS_TEMPLATE_ARGUMENTS
#define BZ_TEMPLATE_KEYWORD_QUALIFIER
#define BZ_TEMPLATE_SCOPED_ARGUMENT_MATCHING
#define BZ_TYPE_PROMOTION
#define BZ_USE_NUMTRAIT
#define BZ_ENUM_COMPUTATIONS
#define BZ_ENUM_COMPUTATIONS_WITH_CAST
#define BZ_HAVE_COMPLEX

#if (__GNUC__ && __GNUC__ == 3)
    #define  BZ_HAVE_NUMERIC_LIMITS
#else
    #undef  BZ_HAVE_NUMERIC_LIMITS
#endif

#define BZ_HAVE_CLIMITS
#define BZ_HAVE_VALARRAY
#undef  BZ_HAVE_COMPLEX_MATH
#define BZ_HAVE_IEEE_MATH
#undef  BZ_HAVE_SYSTEM_V_MATH
#define BZ_MATH_FN_IN_NAMESPACE_STD
#define BZ_COMPLEX_MATH_IN_NAMESPACE_STD
#define BZ_HAVE_STD
#define BZ_HAVE_STL
#define BZ_HAVE_RUSAGE
 
#endif // BZ_CONFIG_H
