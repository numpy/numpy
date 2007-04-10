#ifndef _CP_LOG_H
#define _CP_LOG_H

/** @{ */
/**
 * @file
 * libcprops logging facilities<p>
 *
 * All of these simply print to stdout.  They should probably write to
 * an thread-specific error buffer that can be retreived on failures.
 * 
 * <ul>
 * <li> cp_info(,sg) for printouts on LOG_LEVEL_INFO
 * <li> cp_warn(msg) for warning printouts - LOG_LEVEL_WARNING or lower
 * <li> cp_error(code, msg) for error messages - LOG_LEVEL_ERROR or lower
 * <li> cp_fatal(code, msg) for fatal error messages (LOG_LEVEL_FATAL)
 * </ul>
 */


/** debug level */
#define LOG_LEVEL_DEBUG				0
/** normal log level */
#define LOG_LEVEL_INFO				1
/** relatively quiet - warnings only */
#define LOG_LEVEL_WARNING			2
/** quit - severe errors only */
#define LOG_LEVEL_ERROR				3
/** very quiet - report fatal errors only */
#define LOG_LEVEL_FATAL				4
/** no logging */
#define LOG_LEVEL_SILENT			5

void cp_info(char *msg);
void cp_warn(char *msg);
void cp_error(int code, char *msg);
void cp_fatal(int code, char *msg);

#endif

