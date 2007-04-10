/** @{ */
/**
 * @file
 * very simple logging facilities.
 */

#include <stdio.h>

#include "common.h"
#include "log.h"

static int loglevel = LOG_LEVEL_INFO;

typedef struct _error_code_legend
{
	int code;
	char *msg;
} error_code_legend;

error_code_legend error_messages[] = 
{	
	{CP_MEMORY_ALLOCATION_FAILURE, "MEMORY ALLOCATION FAILURE"}, 
	{CP_INVALID_FUNCTION_POINTER, "INVALID FUNCTION POINTER"}, 
	{CP_THREAD_CREATION_FAILURE, "THREAD CREATION FAILURE"}, 

	{CP_LOADLIB_FAILED, "LOADLIB FAILED"}, 
	{CP_LOADFN_FAILED, "LOADFN FAILED"}, 
	{CP_MODULE_NOT_LOADED, "MODULE NOT LOADED"}, 

	{CP_IO_ERROR, "IO ERROR"},
	{CP_OPEN_PORT_FAILED, "OPEN PORT FAILED"}, 
	{CP_HTTP_FETCH_FAILED, "HTTP FETCH FAILED"}, 
	{CP_INVALID_RESPONSE, "INVALID RESPONSE"}, 
    {CP_HTTP_EMPTY_REQUEST, "EMPTY HTTP REQUEST"},
    {CP_HTTP_INVALID_REQUEST_LINE, "INVALID HTTP REQUEST LINE"},
    {CP_HTTP_INVALID_STATUS_LINE, "INVALID HTTP STATUS LINE"},
    {CP_HTTP_UNKNOWN_REQUEST_TYPE, "UNKNOWN HTTP REQUEST TYPE"},
    {CP_HTTP_INVALID_URI, "INVALID URI"},
    {CP_HTTP_INVALID_URL, "INVALID URL"},
    {CP_HTTP_VERSION_NOT_SPECIFIED, "HTTP VERSION NOT SPECIFIED"},
    {CP_HTTP_1_1_HOST_NOT_SPECIFIED, "HTTP 1.1 HOST NOT SPECIFIED"},
    {CP_HTTP_INCORRECT_REQUEST_BODY_LENGTH, "INCORRECT HTTP REQUEST BODY LENGTH"},
    {CP_SSL_CTX_INITIALIZATION_ERROR, "SSL CONTEXT INITIALIZATION ERROR"},
    {CP_SSL_HANDSHAKE_FAILED, "SSL HANDSHAKE FAILED"},

	{CP_LOG_FILE_OPEN_FAILURE, "LOG FILE OPEN FAILURE"}, 
	{CP_LOG_NOT_OPEN, "LOG NOT OPEN"}, 

	{CP_INVALID_VALUE, "INVALID VALUE"},
	{CP_MISSING_PARAMETER, "MISSING PARAMETER"},
	{CP_BAD_PARAMETER_SET, "BAD PARAMETER SET"},
    {CP_ITEM_EXISTS, "ITEM EXISTS"},
    {CP_UNHANDLED_SIGNAL, "UNHANDLED SIGNAL"},
    {CP_FILE_NOT_FOUND, "FILE NOT FOUND"},
	{CP_METHOD_NOT_IMPLEMENTED, "METHOD NOT IMPLEMENTED"},

	{CP_REGEX_COMPILATION_FAILURE, "INVALID REGULAR EXPRESSION"},
	{CP_COMPILATION_FAILURE, "COMPILATION FAILED"},

	{CP_DBMS_NO_DRIVER, "NO DRIVER"},
	{CP_DBMS_CONNECTION_FAILURE, "DBMS CONNECTION FAILED"},
	{CP_DBMS_QUERY_FAILED, "DBMS QUERY FAILED"},
	{CP_DBMS_CLIENT_ERROR, "DBMS CLIENT ERROR"},
	{CP_DBMS_STATEMENT_ERROR, "DBMS STATEMENT ERROR"},
	{-1, NULL},
}; 

char* error_message_lookup(int code)
{
    error_code_legend* entry=error_messages;
    
    while (entry->code != -1)
    {
        entry++;
    }        
    return entry->msg;    
}

void cp_info(char *msg)
{
	if (loglevel > LOG_LEVEL_INFO) return;

	printf("%s\n", msg);
}

void cp_warn(char *msg)
{
	if (loglevel > LOG_LEVEL_WARNING) return;

	printf("%s\n", msg);
}

void cp_error(int code, char *msg)
{
	char *code_msg;

	if (loglevel > LOG_LEVEL_ERROR) return;
	code_msg = error_message_lookup(code);
	printf("%s: %s\n", code_msg, msg);
}

void cp_fatal(int code, char *msg)
{
	char *code_msg;

	if (loglevel > LOG_LEVEL_FATAL) return;
	code_msg = error_message_lookup(code);
	printf("%s: %s\n", code_msg, msg);
	
	/* cprops has this exit, but that is a bad idea from a library, 
	 * so we don't. 
	 * fixme: This probably deserves some attention in the library.
	 */
}

