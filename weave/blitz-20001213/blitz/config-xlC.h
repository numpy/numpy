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
 
#define BZ_COMPILER_NAME   "xlC"
#define BZ_COMPILER_OPTIONS "-qlongdouble"
#define BZ_OS_NAME         "AIX 1"
#define BZ_BZCONFIG_DATE   "Sun Nov 10 12:47:17 EST 1996"
 
#undef  BZ_NAMESPACES
#undef  BZ_EXCEPTIONS
#undef  BZ_RTTI
#undef  BZ_MEMBER_CONSTANTS
#define BZ_OLD_FOR_SCOPING
#undef  BZ_EXPLICIT
#undef  BZ_MUTABLE
#undef  BZ_TYPENAME
#undef  BZ_NCEG_RESTRICT
#undef  BZ_BOOL
#undef  BZ_CONST_CAST
#undef  BZ_STATIC_CAST
#undef  BZ_REINTERPRET_CAST
#undef  BZ_DYNAMIC_CAST
#define BZ_TEMPLATES
#undef  BZ_PARTIAL_SPECIALIZATION
#undef  BZ_DEFAULT_TEMPLATE_PARAMETERS
#undef  BZ_MEMBER_TEMPLATES
#undef  BZ_MEMBER_TEMPLATES_OUTSIDE_CLASS
#undef  BZ_FULL_SPECIALIZATION_SYNTAX
#define BZ_FUNCTION_NONTYPE_PARAMETERS
#define BZ_TEMPLATE_QUALIFIED_BASE_CLASS
#define BZ_TEMPLATE_QUALIFIED_RETURN_TYPE
#undef  BZ_EXPLICIT_TEMPLATE_FUNCTION_QUALIFICATION
#undef  BZ_TEMPLATES_AS_TEMPLATE_ARGUMENTS
#undef  BZ_TEMPLATE_KEYWORD_QUALIFIER
#define BZ_TEMPLATE_SCOPED_ARGUMENT_MATCHING
#define BZ_TYPE_PROMOTION
#undef  BZ_HAVE_COMPLEX
#undef  BZ_HAVE_NUMERIC_LIMITS
#undef  BZ_HAVE_VALARRAY
#define BZ_LONGDOUBLE128
 
#endif // BZ_CONFIG_H
