#ifndef _CP_COMMON_H
#define _CP_COMMON_H

/** @{ */
/**
 * @file
 * common symbols for cprops library -- mostly error codes
 */

#ifdef	__cplusplus
#ifndef __BEGIN_DECLS
#define __BEGIN_DECLS	extern "C" {
#endif
#ifndef __END_DECLS
#define __END_DECLS	}
#endif
#else
#ifndef __BEGIN_DECLS
#define __BEGIN_DECLS
#endif
#ifndef __END_DECLS
#define __END_DECLS
#endif
#endif

#if defined(linux) || defined(__linux__) || defined (__linux) || defined(__gnu_linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif /* _GNU_SOURCE */
#endif /* linux */

#ifdef __NetBSD__
#ifndef __unix__
#define __unix__ 1
#endif /* __unix__ */
#endif /* __NetBSD__ */

#ifdef _WINDOWS

/* compatibility definitions */
typedef int pid_t;

#define SHUT_RD 	SD_RECEIVE
#define SHUT_WR 	SD_SEND
#define SHUT_RDWR 	SD_BOTH

#define close closesocket

// eric: done so that we aren't a dll.
#define CPROPS_DLL
//#ifdef CPROPS_EXPORTS
//#define CPROPS_DLL __declspec(dllexport)
//#else
//#define CPROPS_DLL __declspec(dllimport)
//#endif
#else /* _WINDOWS */
#define CPROPS_DLL
#endif /* _WINDOWS */

#if (defined linux || defined __linux || defined __gnu_linux__)

#ifndef _REENTRANT
#define _REENTRANT
#endif

/* for pthread_rwlock_t et al. */
#ifndef _XOPEN_SOURCE 
#define _XOPEN_SOURCE 600
#endif
#ifndef __USE_UNIX98
#define __USE_UNIX98
#endif

#include <features.h>

#endif

#define DEFAULT_LOGFILE                                    "cp.log"
#if defined(unix) || defined(__unix__) || defined(__MACH__)
#define DEFAULT_TIME_FORMAT                                "%Y-%m-%d %T"
#else
#define DEFAULT_TIME_FORMAT                                "%Y-%m-%d %H:%M:%S"
#endif /* unix */
        /* error codes */

#define CP_MEMORY_ALLOCATION_FAILURE                       10000
#define CP_INVALID_FUNCTION_POINTER                        10010
#define CP_THREAD_CREATION_FAILURE                         10020

#define CP_LOADLIB_FAILED                                  11010
#define CP_LOADFN_FAILED                                   11020
#define CP_MODULE_NOT_LOADED                               11030

#define CP_IO_ERROR                                        12000
#define CP_OPEN_PORT_FAILED                                12010
#define CP_HTTP_FETCH_FAILED                               12020
#define CP_INVALID_RESPONSE                                12030
#define CP_HTTP_EMPTY_REQUEST                              12100
#define CP_HTTP_INVALID_REQUEST_LINE                       12110
#define CP_HTTP_INVALID_STATUS_LINE                        12111
#define CP_HTTP_UNKNOWN_REQUEST_TYPE                       12120
#define CP_HTTP_INVALID_URI                                12130
#define CP_HTTP_INVALID_URL                                12131
#define CP_HTTP_VERSION_NOT_SPECIFIED                      12140
#define CP_HTTP_1_1_HOST_NOT_SPECIFIED                     12150
#define CP_HTTP_INCORRECT_REQUEST_BODY_LENGTH              12160
#define CP_SSL_CTX_INITIALIZATION_ERROR                    12200
#define CP_SSL_HANDSHAKE_FAILED                            12210
#define CP_SSL_VERIFICATION_ERROR                          12220

#define CP_LOG_FILE_OPEN_FAILURE                           13000
#define CP_LOG_NOT_OPEN                                    13010

#define CP_INVALID_VALUE                                   14000
#define CP_MISSING_PARAMETER                               14010
#define CP_BAD_PARAMETER_SET                               14020
#define CP_ITEM_EXISTS                                     14030
#define CP_UNHANDLED_SIGNAL                                14040
#define CP_FILE_NOT_FOUND                                  14050
#define CP_METHOD_NOT_IMPLEMENTED                          14060

#define CP_REGEX_COMPILATION_FAILURE                       15000
#define CP_COMPILATION_FAILURE                             15010

#define CP_DBMS_NO_DRIVER                                  16000
#define CP_DBMS_CONNECTION_FAILURE                         16010
#define CP_DBMS_QUERY_FAILED                               16020
#define CP_DBMS_CLIENT_ERROR                               16030
#define CP_DBMS_STATEMENT_ERROR                            16040

/** @} */

#endif

