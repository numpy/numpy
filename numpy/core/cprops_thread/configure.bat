@echo off
echo +--------------------------------------------------------------------------+
echo ^|                                                                          ^|
echo ^|                  cprops configuration script for windows                 ^|
echo ^|                                                                          ^|
echo +--------------------------------------------------------------------------+

rem *****************************************************************************
rem *                                                                           *
rem *                              set up utilities                             *
rem *                                                                           *
rem *****************************************************************************

echo Checking for C compiler CL
cl > nul: 2>&1
if errorlevel 1 (
	@echo can^'t find CL on the path
	goto :EOF
)
echo found.
if defined LDFLAGS set LINKFLAGS=/link %LDFLAGS%
set CFG_CFLAGS=%CFLAGS%
set CFG_LDFLAGS=%LDFLAGS%

rem *****************************************************************************
rem *                                                                           *
rem *                                  newline                                  *
rem *                                                                           *
rem *****************************************************************************

if exist newline.c del /q newline.c
echo #include ^<stdio.h^> > newline.c
echo int main() >> newline.c
echo { >> newline.c
echo     printf("\n"); >> newline.c
echo     return 0; >> newline.c
echo } >> newline.c

cl %CFLAGS% newline.c %LINKFLAGS% > nul: 2>&1
if errorlevel 1 (
	@echo can^'t compile newline utility
	goto :END
)

if exist newline.c del /q newline.c
if exist newline.obj del /q newline.obj

rem *****************************************************************************
rem *                                                                           *
rem *                                 readline                                  *
rem *                                                                           *
rem *****************************************************************************

if exist readline.c del /q readline.c
echo #include ^<stdio.h^> > readline.c
echo #include ^<stdlib.h^> >> readline.c
echo #include ^<string.h^> >> readline.c
newline >> readline.c
echo #define MAXLEN 0x400 >> readline.c
newline >> readline.c
echo int main(int argc, char *argv[]) >> readline.c
echo { >> readline.c
echo     char *question; >> readline.c
echo     char answer[MAXLEN]; >> readline.c
echo     FILE *out; >> readline.c
newline >> readline.c
echo     if (argc ^> 1) printf("%%s ", argv[1]); >> readline.c
echo     if (argc ^> 2) printf("[%%s] ", argv[2]); >> readline.c
echo     printf(": "); >> readline.c
echo     fgets(answer, MAXLEN - 1, stdin); >> readline.c
echo     if ((answer == NULL ^|^| *answer == '\0' ^|^| *answer == '\n') ^&^& argc ^> 2) >> readline.c
echo         strcpy(answer, argv[2]); >> readline.c
echo     else /* chop newline */ >> readline.c
echo         answer[strlen(answer) - 1] = '\0'; >> readline.c
echo     if ((out = fopen("__answer", "w")) == NULL) >> readline.c
echo         return -1; >> readline.c
echo     fputs(answer, out); >> readline.c
echo     fclose(out); >> readline.c
newline >> readline.c
echo     return 0; >> readline.c
echo } >> readline.c

cl %CFLAGS% readline.c %LINKFLAGS% > nul: 2>&1
if errorlevel 1 (
	@echo can^'t compile readline utility
	goto :END
)

if exist readline.c del /q readline.c
if exist readline.obj del /q readline.obj

rem *****************************************************************************
rem *                                                                           *
rem *                                   match                                   *
rem *                                                                           *
rem *****************************************************************************

if exist match.c del /q match.c
echo #include ^<stdio.h^> > match.c
newline >> match.c
echo #define UC(c) ((c) ^>= 'a' ^&^& (c) ^<= 'z' ? (c) - 'a' + 'A' : (c)) >> match.c
newline >> match.c
echo char *stristr(char *buffer, char *key) >> match.c
echo { >> match.c
echo     char *p, *q, *r; >> match.c
newline >> match.c
echo     p = buffer; >> match.c
echo     while (*p != '\0') >> match.c
echo     { >> match.c
echo         q = p; >> match.c
echo         r = key; >> match.c
echo         while (*q != '\0' ^&^& *r != '\0' ^&^& UC(*q) == UC(*r)) >> match.c
echo         { >> match.c
echo             q++; >> match.c
echo             r++; >> match.c
echo         } >> match.c
echo         if (*r == '\0') return p; >> match.c
echo         p++; >> match.c
echo      } >> match.c
newline >> match.c
echo      return NULL; >> match.c
echo } >> match.c
newline >> match.c
echo char *strirstr(char *buffer, char *key) >> match.c
echo { >> match.c
echo     char *p, *q, *r; >> match.c
newline >> match.c
echo     p = ^&buffer[strlen(buffer) - 1]; >> match.c
echo     while (p ^>= buffer) >> match.c
echo     { >> match.c
echo         q = p; >> match.c
echo         r = key; >> match.c
echo         while (*q != '\0' ^&^& *r != '\0' ^&^& UC(*q) == UC(*r)) >> match.c
echo         { >> match.c
echo             q++; >> match.c
echo             r++; >> match.c
echo         } >> match.c
echo         if (*r == '\0') return p; >> match.c
echo         p--; >> match.c
echo      } >> match.c
newline >> match.c
echo      return NULL; >> match.c
echo } >> match.c
newline >> match.c
echo int main(int argc, char *argv[]) >> match.c
echo { >> match.c
echo     char *src; >> match.c
echo     char *key; >> match.c
echo     int len; >> match.c
echo     int reverse; >> match.c
echo     char *p, ch; >> match.c
echo     FILE *out; >> match.c
newline >> match.c
echo     if (argc ^< 3) return 1; >> match.c
echo     src = argv[1]; >> match.c
echo     key = argv[2]; >> match.c
echo     reverse = argc ^> 3 ^&^& strcmp(argv[3], "-r") == 0; >> match.c
echo     p = reverse ? strirstr(src, key) : stristr(src, key); >> match.c
echo     if (p == NULL) return 2; >> match.c  
newline >> match.c
echo     ch = *p; >> match.c
echo     *p = '\0'; >> match.c
echo     out = fopen("__prefix", "w"); >> match.c
echo     fputs(src, out); >> match.c
echo     fclose(out); >> match.c
echo     *p = ch; >> match.c
newline >> match.c
echo     src = p; >> match.c
echo     p += strlen(key); >> match.c
echo     ch = *p; >> match.c
echo     *p = '\0'; >> match.c
echo     out = fopen("__match", "w"); >> match.c
echo     fputs(src, out); >> match.c
echo     fclose(out); >> match.c
echo     *p = ch; >> match.c
newline >> match.c
echo     out = fopen("__suffix", "w"); >> match.c
echo     fputs(p, out); >> match.c
echo     fclose(out); >> match.c
newline >> match.c
echo     return 0; >> match.c
echo } >> match.c

cl %CFLAGS% match.c %LINKFLAGS% > nul: 2>&1
if errorlevel 1 (
	@echo can^'t compile match utility
	goto :END
)
if exist match.c del /q match.c
if exist match.obj del /q match.obj


rem *****************************************************************************
rem *                                                                           *
rem *                                 strsubst                                  *
rem *                                                                           *
rem *****************************************************************************

if exist strsubst.c del /q strsubst.c
echo #include ^<stdio.h^> > strsubst.c
echo #include ^<stdlib.h^> >> strsubst.c
echo #include ^<string.h^> >> strsubst.c
newline >> strsubst.c
echo int main(int argc, char *argv[]) >> strsubst.c
echo { >> strsubst.c
echo     char buf[0x400]; >> strsubst.c
echo     char *p, *q; >> strsubst.c
echo     int rc; >> strsubst.c
echo     FILE *in, *out; >> strsubst.c
echo     char *fname_in, *fname_out, *token, *subst; >> strsubst.c
newline >> strsubst.c
echo     if (argc ^< 4) >> strsubst.c
echo     { >> strsubst.c
echo         fprintf(stderr, "usage: %%s <filename> <src string> <dest string>\n", >> strsubst.c
echo                 argv[0]); >> strsubst.c
echo         return 1; >> strsubst.c
echo     } >> strsubst.c
newline >> strsubst.c
echo     fname_in = argv[1]; >> strsubst.c
echo     token = argv[2]; >> strsubst.c
echo     subst = argv[3]; >> strsubst.c
newline >> strsubst.c
echo     if (strcmp(fname_in, "-") == 0) >> strsubst.c
echo     { >> strsubst.c
echo         in = stdin; >> strsubst.c
echo         out = stdout; >> strsubst.c
echo     } >> strsubst.c
echo     else >> strsubst.c
echo     { >> strsubst.c
echo         fname_out = malloc(strlen(fname_in) + strlen(".strsubst") + 1); >> strsubst.c
echo         sprintf(fname_out, "%%s.strsubst", fname_in); >> strsubst.c
echo         if ((in = fopen(fname_in, "r")) == NULL) >> strsubst.c
echo         { >> strsubst.c
echo             fprintf(stderr, "%%s: can\'t open %%s\n", argv[0], fname_in); >> strsubst.c
echo             return 2; >> strsubst.c
echo         } >> strsubst.c
newline >> strsubst.c
echo         if ((out = fopen(fname_out, "w")) == NULL) >> strsubst.c
echo         { >> strsubst.c
echo             fprintf(stderr, "%%s: can\'t open %s\n", argv[0], fname_out); >> strsubst.c
echo             return 2; >> strsubst.c
echo         } >> strsubst.c
echo         free(fname_out); >> strsubst.c
echo     } >> strsubst.c
newline >> strsubst.c
echo     while ((fgets(buf, 0x400, in))) >> strsubst.c
echo     { >> strsubst.c
echo         q = buf; >> strsubst.c
echo         while ((p = strstr(q, token)) != NULL) >> strsubst.c
echo         { >> strsubst.c
echo             *p = '\0'; >> strsubst.c
echo             fprintf(out, "%%s%%s", q, subst); >> strsubst.c
echo             q = p + strlen(token); >> strsubst.c
echo         } >> strsubst.c
echo         fprintf(out, q); >> strsubst.c
echo     } >> strsubst.c
newline >> strsubst.c
echo     fclose(in); >> strsubst.c
echo     fclose(out); >> strsubst.c
newline >> strsubst.c
echo     return 0; >> strsubst.c
echo } >> strsubst.c

cl %CFLAGS% strsubst.c %LINKFLAGS% > nul: 2>&1
if errorlevel 1 (
	@echo can^'t compile strsubst utility
	goto :END
)
if exist strsubst.c del /q strsubst.c
if exist strsubst.obj del /q strsubst.obj

rem *****************************************************************************
rem *                                                                           *
rem *                                    head                                   *
rem *                                                                           *
rem *****************************************************************************

if exist head.c del /q head.c
echo #include ^<stdio.h^> > head.c
echo #include ^<stdlib.h^> >> head.c
newline >> head.c
echo int main(int argc, char *argv[]) >> head.c
echo { >> head.c
echo     char buf[0x400]; >> head.c
echo     int line_count; >> head.c
newline >> head.c
echo     if (argc ^< 2) >> head.c
echo     { >> head.c
echo         fprintf(stderr, "usage: %%s <line count>\n", argv[0]); >> head.c
echo         return 1; >> head.c
echo     } >> head.c
echo     line_count = atoi(argv[1]); >> head.c
newline >> head.c
echo     while (line_count-- ^&^& (fgets(buf, 0x400, stdin)) != NULL) >> head.c
echo         printf(buf); >> head.c
newline >> head.c
echo     return 0; >> head.c
echo } >> head.c

cl %CFLAGS% head.c %LINKFLAGS% > nul: 2>&1
if errorlevel 1 (
	@echo can^'t compile head utility
	goto :END
)
if exist head.c del /q head.c
if exist head.obj del /q head.obj


rem *****************************************************************************
rem *                                                                           *
rem *                                   filter                                  *
rem *                                                                           *
rem *****************************************************************************

if exist filter.c del /q filter.c
echo #include ^<stdio.h^> > filter.c
newline >> filter.c
echo #define BUFSIZE 0x1000 >> filter.c
newline >> filter.c
echo int main(int argc, char *argv[]) >> filter.c
echo { >> filter.c
echo 	char buf[BUFSIZE]; >> filter.c
echo 	char *mark; >> filter.c
echo 	char *p; >> filter.c
newline >> filter.c
echo 	mark = argv[1]; >> filter.c
newline >> filter.c
echo 	while (!feof(stdin)) >> filter.c
echo 	{ >> filter.c
echo 		if ((p = fgets(buf, BUFSIZE - 1, stdin)) == NULL) break; >> filter.c
echo 		if (strstr(buf, mark)) continue; >> filter.c
echo 		fprintf(stdout, "%%s", buf); >> filter.c
echo 	} >> filter.c
newline >> filter.c
echo 	return 0; >> filter.c
echo } >> filter.c

cl %CFLAGS% filter.c %LINKFLAGS% > nul: 2>&1
if errorlevel 1 (
	@echo can^'t compile filter utility
	goto :END
)
if exist filter.c del /q filter.c
if exist filter.obj del /q filter.obj


rem *****************************************************************************
rem *                                                                           *
rem *                               configuration                               *
rem *                                                                           *
rem *****************************************************************************

newline

set PREFIX=
set MATCH=
set SUFFIX=
set NEEDLE=
set HAYSTACK=
set QUESTION=
set ANSWER=
set USE_SSL=
set SSL_DIR=
set PCRE_DIR=
set SUBDIRS=
set DEBUG=

echo please choose a few configuration options.
newline


rem *****************************************************************************
rem *                                                                           *
rem *                                   PCRE                                    *
rem *                                                                           *
rem *****************************************************************************

echo the cprops build on windows requires PCRE. 
:GET_PCRE
newline
set _PCRE=
if exist _pcre (
	for /f "tokens=*" %%i in (_pcre) do set _PCRE=%%i
)

set QUESTION="Please specify PCRE include and library path"
set DEFAULT=
if defined _PCRE set DEFAULT=%_PCRE%
call :readline

if x%ANSWER% == x (
	newline
	echo libcprops requires PCRE to build on windows. If you do not have PCRE installed,
	echo you could download it from http://gnuwin32.sourceforge.net/packages/pcre.htm
	goto :END
)

set PCRE_DIR=%ANSWER%
if not exist %PCRE_DIR%\include\pcreposix.h (
	echo can^'t find %PCRE_DIR%\include\pcreposix.h
	goto :GET_PCRE
)
newline

if exist _pcre del /q _pcre
echo %PCRE_DIR% > _pcre

rem *****************************************************************************
rem *                                                                           *
rem *                             Create DLL or lib                             *
rem *                                                                           *
rem *****************************************************************************

set QUESTION="link libcprops dynamically (DLL) or statically (LIB)"
set DEFAULT=dll
call :readline
newline
set TARGET=libcprops.lib

if /i not "d"=="%ANSWER:~0,1%" goto :ENDLINK

set TARGET=libcprops.dll
set MAIN_CFLAGS=/D "_USRDLL" /D "CPROPS_EXPORTS" /GD
:ENDLINK

newline

rem *****************************************************************************
rem *                                                                           *
rem *                                  DEBUG                                    *
rem *                                                                           *
rem *****************************************************************************

echo cprops may be built in DEBUG mode. The resulting DLL or library includes debug
echo information, and some routines make debug printouts. 
set QUESTION="Build in DEBUG mode?"
set DEFAULT=no
call :readline
newline
if /i "n"=="%ANSWER:~0,1%" goto :END_DEBUG
set DEBUG=yes

echo defining __TRACE__ gives a higher resolution of debug printouts.
set QUESTION="define __TRACE__?"
set DEFAULT=no
call :readline
newline
if /i "n"=="%ANSWER:~0,1%" goto :END_DEBUG
set __TRACE__=yes
:END_DEBUG

rem *****************************************************************************
rem *                                                                           *
rem *                       Multiple values in hash tables                      *
rem *                                                                           *
rem *****************************************************************************

set QUESTION="allow multiple values in hash tables"
set DEFAULT=no
call :readline
newline
set HASHTABLE_MULTIPLES=

if /i not "y"=="%ANSWER:~0,1%" goto :END_HASHTABLE_MULTIPLES

set HASHTABLE_MULTIPLES=yes

:END_HASHTABLE_MULTIPLES


rem *****************************************************************************
rem *                                                                           *
rem *                       Multiple values in hash lists                       *
rem *                                                                           *
rem *****************************************************************************

set QUESTION="allow multiple values in hash lists"
set DEFAULT=no
call :readline
newline
set HASHLIST_MULTIPLES=

if /i not "y"=="%ANSWER:~0,1%" goto :END_HASHLIST_MULTIPLES

set HASHLIST_MULTIPLES=yes

:END_HASHLIST_MULTIPLES


rem *****************************************************************************
rem *                                                                           *
rem *                            HTTP cookie support                            *
rem *                                                                           *
rem *****************************************************************************

set QUESTION="include support for HTTP cookies"
set DEFAULT="yes"
call :readline
newline
set USE_COOKIES=yes

if /i "y"=="%ANSWER:~0,1%" goto :END_COOKIES

set USE_COOKIES=

:END_COOKIES


rem *****************************************************************************
rem *                                                                           *
rem *                            HTTP session support                           *
rem *                                                                           *
rem *****************************************************************************

set QUESTION="include support for HTTP sessions"
set DEFAULT="yes"
call :readline
newline
set USE_HTTP_SESSIONS=yes

if /i "y"=="%ANSWER:~0,1%" goto :END_HTTP_SESSIONS

set USE_HTTP_SESSIONS=

:END_HTTP_SESSIONS


rem *****************************************************************************
rem *                                                                           *
rem *                                 Open SSL                                  *
rem *                                                                           *
rem *****************************************************************************

set QUESTION="include ssl support (this requires Open SSL)" 
set DEFAULT="yes"
call :readline
newline

if /i not "y"=="%ANSWER:~0,1%" goto :ENDSSL

set USE_SSL=1

if not defined OPENSSL_CONF goto :NO_OPENSSL_CONF

echo found OPENSSL_CONF at %OPENSSL_CONF%
set NEEDLE=OpenSSL
set HAYSTACK=%OPENSSL_CONF%
call :match
set SSL_DIR=%PREFIX%%MATCH%
if defined MATCH goto ENDSSL

:NO_OPENSSL_CONF

set QUESTION="please specify the location of your Open SSL installation"
set DEFAULT=
call :readline

set SSL_DIR=%ANSWER%
@echo using openssl installation at %SSL_DIR%

:ENDSSL

newline
newline

rem *****************************************************************************
rem *                                                                           *
rem *                                  cpsvc                                    *
rem *                                                                           *
rem *****************************************************************************

echo cpsvc is a simple web server included as sample code with the cprops 
echo distribution. cpsvc is based on the cp_httpsocket API and supports CGI, 
echo HTTP sessions, request piplining, and SSL if libcprops is configured 
echo accordingly.
set QUESTION="build cpsvc?"
set DEFAULT=yes
call :readline
newline

if /i not "y"=="%ANSWER:~0,1%" goto :END_CPSVC

set BUILD_CPSVC=yes

echo cpsp is an html page scripting framework allowing embedding C code in web 
echo pages. Requires lex or an equivalent and yacc or an equivalent. 
set QUESTION="build cpsp?"
set DEFAULT=yes
call :readline
newline

if /i not "y"=="%ANSWER:~0,1%" goto :END_CPSVC

set BUILD_CPSP=yes
set SUBDIRS=%SUBDIRS% svc\cpsp

lex --version > nul: 2>&1
if errorlevel 1 goto :FLEX 
set LEX=lex
goto :FIND_YACC

:FLEX
flex --version > nul: 2>&1
if errorlevel 1 goto :ASK_LEX

set LEX=flex
goto :FIND_YACC

:ASK_LEX
echo can't find lex or flex on the path. If you don't have lex installed, a flex 
echo version for windows is vailable at 
echo http://www.monmouth.com/~wstreett/lex-yacc/lex-yacc.html
newline

set _LEX=
if exist _lex (
	for /f "tokens=*" %%i in (_lex) do set _LEX=%%i
)

:LEX
set QUESTION="please specify path to lex executable"
set DEFAULT=
if defined _LEX set DEFAULT=%_LEX%

call :readline
newline
if x%ANSWER%==x (
	echo no lex available, stopping
	goto :END
)
set LEX=%ANSWER%
%LEX% --version > nul: 2>&1
if errorlevel 1 (
	echo can't execute lex at [%LEX%]
	goto :LEX
)
if exist _lex del /q _lex
echo %LEX% > _lex

:FIND_YACC
yacc --version > nul: 2>&1
if errorlevel 1 goto :BISON

set YACC=yacc
goto :END_CPSVC


:BISON
bison --version > nul: 2>&1 
if errorlevel 1 goto :ASK_YACC

set YACC=bison
goto :END_CPSVC

:ASK_YACC
echo can't find yacc or bison on the path. If you don't have yacc installed, a 
echo bison version for windows is available at 
echo http://www.monmouth.com/~wstreett/lex-yacc/lex-yacc.html
newline

set _YACC=
if exist _yacc (
	for /f "tokens=*" %%i in (_yacc) do set _YACC=%%i
)

:YACC
set QUESTION="please specify path to yacc executable"
set DEFAULT=
if defined _YACC set DEFAULT=%_YACC%

call :readline
if x%ANSWER%==x (
	echo no yacc available, stopping
	goto :END
)
set YACC=%ANSWER%
%YACC% --version > nul: 2>&1
if errorlevel 1 (
	echo can't execute yacc at [%YACC%]
	goto :YACC
)

if exist _yacc del /q _yacc
echo %YACC% > _yacc

if exist %YACC%.simple copy %YACC%.simple svc\cpsp > nul: 2>&1 

:END_CPSVC


rem *****************************************************************************
rem *                                                                           *
rem *                            DBMS abstraction layer                         *
rem *                                                                           *
rem *****************************************************************************

set QUESTION="Build cp_dbms - DBMS abstraction layer"
set DEFAULT="yes"
call :readline
newline
set BUILD_DBMS=dynamic

if /i "y"=="%ANSWER:~0,1%" goto :DBMS_LINKAGE

set BUILD_CP_DBMS=

goto :END_DBMS

:DBMS_LINKAGE

set QUESTION="link DBMS driver/s statically or dynamically"
set DEFAULT="dynamic"
call :readline
newline

if /i not "s"=="%ANSWER:~0,1%" goto :DBMS_DRIVERS

set BUILD_DBMS=static

:DBMS_DRIVERS
set CP_DBMS_DRIVERS=
set QUESTION="install PostgresQL driver (requires libpq)"
set DEFAULT=no
call :readline
newline

if /i not "y"=="%ANSWER:~0,1%" goto :DBMS_MYSQL

:GET_PGSQL_PATH
set _PGSQL=
if exist _pgsql (
	for /f "tokens=*" %%i in (_pgsql) do set _PGSQL=%%i
)

set QUESTION="please specify path to postgres headers and include files"
set DEFAULT=
if defined _PGSQL set DEFAULT=%_PGSQL%
call :readline
set _PGSQL=
newline
set POSTGRES_DIR=%ANSWER%

if "x%ANSWER%"=="x" (
	echo "postgres path not specified, stopping"
	goto :END
)

if not exist %POSTGRES_DIR%\include\libpq-fe.h (
	echo can't find libpq-fe.h under %POSTGRES_DIR%\include
	goto :GET_PGSQL_PATH
)
if exist %POSTGRES_DIR%\lib\libpq.lib goto :GOT_LIBPQ
if exist %POSTGRES_DIR%\lib\ms\libpq.lib goto :GOT_LIBPQ
echo can't find libpq.lib under %POSTGRES_DIR%\lib or %POSTGRES_DIR%\lib\ms
echo did you install postgres from the no-installer zip? You'll need the 
echo installer. Make sure to select developer files and MSVC libraries. 
goto :GET_PGSQL_PATH
:GOT_LIBPQ

echo %POSTGRES_DIR%>_pgsql

:DBMS_MYSQL
set QUESTION="install MySQL driver (requires mysqlclient.lib)"
set DEFAULT=no
call :readline
newline

if /i not "y"=="%ANSWER:~0,1%" goto :END_DBMS

:GET_MYSQL_PATH
set _MYSQL=
if exist _mysql (
	for /f "tokens=*" %%i in (_mysql) do set _MYSQL=%%i
)
set QUESTION="please specify path to MySQL headers and include files"
set DEFAULT=
if defined _MYSQL set DEFAULT=%_MYSQL%
call :readline
set _MYSQL=
newline
set MYSQL_DIR=%ANSWER%

if "x%ANSWER%"=="x" (
	echo "MySQL path not specified, stopping"
	goto :END
)

if not exist "%MYSQL_DIR%\lib\opt\mysqlclient.lib" (
	echo can't find mysqlclient.lib on mysql path [%MYSQL_DIR%]
	goto :GET_MYSQL_PATH
)
if not exist "%MYSQL_DIR%\include\mysql.h" (
	echo can't find mysql.h header in mysql installation - possibly MySQL was 
	echo installed without developer files
	goto :GET_MYSQL_PATH
)

echo "%MYSQL_DIR%">_mysql
:END_DBMS



rem *****************************************************************************
rem *                                                                           *
rem *                                write output                               *
rem *                                                                           *
rem *****************************************************************************

newline
echo +--------------------------------------------------------------------------+
echo ^|                                                                          ^|
echo ^|              generating configuration headers and make files             ^|
echo ^|                                                                          ^|
echo +--------------------------------------------------------------------------+
newline

if exist config-cpwin.h del /q config-cpwin.h

set CFG_CFLAGS=%CFG_CFLAGS% /I%PCRE_DIR%\include
set CFG_LDFLAGS=%CFG_LDFLAGS%
set CFG_LIBS=%PCRE_DIR%\lib\pcre.lib

newline
copy config.h.vc config.h > nul: 2>&1

echo writing config-cpwin.h

if defined HASHTABLE_MULTIPLES (
	echo #define CP_HASHTABLE_MULTIPLE_VALUES 1 >> config-cpwin.h
)

if defined HASHLIST_MULTIPLES (
	echo #define CP_HASHLIST_MULTIPLE_VALUES 1 >> config-cpwin.h
)

if defined USE_COOKIES (
	echo #define CP_USE_COOKIES 1 >> config-cpwin.h
)

if defined USE_HTTP_SESSIONS (
	echo #define CP_USE_HTTP_SESSIONS 1 >> config-cpwin.h
)

if defined BUILD_CPSVC (
	set SUBDIRS=svc %SUBDIRS%
)

if defined BUILD_DBMS (
	set OPT_OBJS=%OPT_OBJS% db.obj
	if /i "d"=="%BUILD_DBMS:~0,1%" call :set_db_directories
	if /i "s"=="%BUILD_DBMS:~0,1%" call :set_db_objects
)

if defined USE_SSL (
	echo #define CP_USE_SSL 1 >> config-cpwin.h
	set CFG_CFLAGS=%CFG_CFLAGS% /I%SSL_DIR%\include
	set CFG_LDFLAGS=%CFG_LDFLAGS% %SSL_DIR%\lib\VC\libeay32MT.lib %SSL_DIR%\lib\VC\ssleay32MT.lib
)

if defined DEBUG (
	set CFG_CFLAGS=%CFG_CFLAGS% /DDEBUG /D_DEBUG /Zi /MTd
	set CFG_LDFLAGS=%CFG_LDFLAGS% /debug /pdb:libcprops.pdb /pdbtype:sept 
)
if not defined DEBUG set CFG_CFLAGS=%CFG_CFLAGS% /MT

if defined __TRACE__ set CFG_CFLAGS=%CFG_CFLAGS% /D__TRACE__

if exist Makefile del /q Makefile
echo writing Makefile

echo ############################################################################ > Makefile
echo # >> Makefile
echo # This makefile was generated by the configure.bat script. run nmake in this >> Makefile
echo # directory to build libcprops. >> Makefile
echo # >> Makefile
echo # Copyright Ilan Aelion 2005, 2006 >> Makefile
echo # >> Makefile
echo # Please send bug reports, comments, suggestions, patches etc. to iaelion at  >> Makefile 
echo # users dot sourceforge dot net. >> Makefile
echo # >> Makefile
echo ############################################################################ >> Makefile
newline >> Makefile
echo CC=CL >> Makefile
echo LD=LINK >> Makefile
echo CFLAGS=%MAIN_CFLAGS% %CFG_CFLAGS% >> Makefile
echo LDFLAGS=%CFG_LDFLAGS% >> Makefile
echo LIBS=%CFG_LIBS% >> Makefile
newline >> Makefile
echo TARGET=%TARGET% >> Makefile
echo OPT_OBJS=%OPT_OBJS% >> Makefile
echo OPT_TARGETS=%OPT_TARGETS% >> Makefile
if defined POSTGRES_DIR (
	newline >> Makefile
	echo PGSQL_CFLAGS=/I"%POSTGRES_DIR%"\include >> Makefile
	echo PGSQL_LDFLAGS=/libpath:"%POSTGRES_DIR%"\lib /libpath:"%POSTGRES_DIR%"\lib\ms>> Makefile
	echo PGSQL_LIBS=libpq.lib >> Makefile
)
if defined MYSQL_DIR (
	newline >> Makefile
	echo MYSQL_CFLAGS=/I"%MYSQL_DIR%"\include >> Makefile
	echo MYSQL_LDFLAGS=/libpath:"%MYSQL_DIR%"\lib\opt >> Makefile
	echo MYSQL_LIBS=mysqlclient.lib >> Makefile
)

CD > CWD
for /f "tokens=*" %%i in (CWD) do set CWD=%%i
echo top_builddir=%CWD%>> Makefile
del /q CWD
newline >> Makefile
echo subdirs=%SUBDIRS% >> Makefile
newline >> Makefile

type Makefile.vc >> Makefile
if errorlevel 1 (
	echo can't find Makefile.vc
	goto :EOF
)

if defined BUILD_CPSVC (
	echo writing svc\win.mak
	if exist svc\win.mak del /q svc\win.mak > nul: 2>&1

	copy svc\Makefile.vc svc\Makefile > nul: 2>&1
	if errorlevel 1 (
		echo can't find svc\Makefile.vc
		goto :EOF
	)
	call :write_svc_mak

	if exist svc\runcpsvc.bat del /q svc\runcpsvc.bat
	CD > CWD
	for /f "tokens=*" %%i in (CWD) do set CWD=%%i
	echo set PATH=%%^PATH%%;%CWD%;%PCRE_DIR%\lib;%SSL_DIR%\lib\vc > svc\runcpsvc.bat
	echo cpsvc %%^1 %%^2 %%^3 %%^4 %%^5 %%^6 %%^7 %%^8 %%^9 >> svc\runcpsvc.bat
)

if defined BUILD_CPSP (
	echo #define CP_USE_CPSP 1 >> config-cpwin.h

	if exist svc\cpsp\win.cpsp.mak del svc\cpsp\win.cpsp.mak
	call :write_svc_cpsp_mak
	CD > CWD
	for /f "tokens=*" %%i in (CWD) do set CWD=%%i
	echo top_builddir=%CWD%>> svc\cpsp\win.cpsp.mak
	del /q CWD
	echo prefix= >> svc\cpsp\win.cpsp.mak
	echo exec_prefix= >> svc\cpsp\win.cpsp.mak
	newline >> svc\cpsp\win.cpsp.mak
	echo libdir=%CWD% >> svc\cpsp\win.cpsp.mak
	echo incdir=%CWD%\.. >> svc\cpsp\win.cpsp.mak
	echo bindir=%CWD%\svc >> svc\cpsp\win.cpsp.mak
	newline >> svc\cpsp\win.cpsp.mak
	echo LEX=%LEX% >> svc\cpsp\win.cpsp.mak
	echo YACC=%YACC% >> svc\cpsp\win.cpsp.mak
	
	echo CPSP_SOURCES=cpsp.c cpsp_invoker.c >> svc\win.mak

	copy svc\cpsp\Makefile.vc svc\cpsp\Makefile > nul: 2>&1
	copy svc\cpsp\Makefile.cpsp.vc svc\cpsp\Makefile.cpsp > nul: 2>&1
	
	set SETPATH=svc\cpsp\setpath.bat
	call :write_setpath
	set SETPATH=svc\setpath.bat
	call :write_setpath
	set SETPATH=
	copy newline.exe svc > nul: 2>&1
	copy newline.exe svc\cpsp > nul: 2>&1
	copy match.exe svc > nul: 2>&1
	copy match.exe svc\cpsp > nul: 2>&1
	copy strsubst.exe svc > nul: 2>&1
	copy strsubst.exe svc\cpsp > nul: 2>&1
	copy filter.exe svc > nul: 2>&1
	copy filter.exe svc\cpsp > nul: 2>&1
)

echo writing example\win.mak
copy example\Makefile.vc example\Makefile > nul: 2>&1
if errorlevel 1 (
	echo can't find example\Makefile.vc
	goto :EOF
)
if exist example\win.mak del /q example\win.mak > nul: 2>&1
echo CFLAGS=$(CFLAGS) %CFG_CFLAGS% > example\win.mak
echo LDFLAGS=$(LDFLAGS) %CFG_LDFLAGS% >> example\win.mak
echo LIBS=$(LIBS) %CFG_LIBS% >> example\win.mak
if defined POSTGRES_DIR echo OPT_SRC=test_pq.c>> example\win.mak
if defined MYSQL_DIR echo OPT_SRC=$(OPT_SRC) test_mysql.c>> example\win.mak

echo writing http.h
head 1 < VERSION > LIBVERSION
for /f "tokens=*" %%i in (LIBVERSION) do set VERSION=%%i
if exist http.h.in del http.h.in
ren http.h http.h.in
strsubst http.h.in __CPROPSVERSION %VERSION%
copy http.h.in.strsubst http.h > nul: 2>&1
set VERSION=

echo done.
newline 
echo run nmake to build libcprops.

goto :END


rem *****************************************************************************
rem *                                                                           *
rem *                              utility invocations                          *
rem *                                                                           *
rem *****************************************************************************

:readline
set ANSWER=
readline %QUESTION% %DEFAULT%
if errorlevel 1 goto :EOF
for /f "tokens=*" %%i in (__answer) do set ANSWER=%%i
if exist __answer del __answer
goto :EOF

:match
set MATCH=
set PREFIX=
set SUFFIX=
match %HAYSTACK% %NEEDLE%
if errorlevel 1 goto :EOF

for /f "tokens=*" %%i in (__prefix) do set PREFIX=%%i
for /f "tokens=*" %%i in (__match) do set MATCH=%%i
for /f "tokens=*" %%i in (__suffix) do set SUFFIX=%%i
if exist __prefix del __prefix
if exist __match del __match
if exist __suffix del __suffix
goto :EOF

:write_svc_mak
echo CFLAGS=$(CFLAGS) %CFG_CFLAGS% > svc\win.mak
echo LDFLAGS=$(LDFLAGS) %CFG_LDFLAGS% >> svc\win.mak
echo LIBS=$(LIBS) %CFG_LIBS% >> svc\win.mak
goto :EOF

:write_svc_cpsp_mak
echo CFLAGS=$(CFLAGS) %CFG_CFLAGS% > svc\cpsp\win.cpsp.mak
echo LDFLAGS=$(LDFLAGS) %CFG_LDFLAGS% >> svc\cpsp\win.cpsp.mak
echo LIBS=$(LIBS) %CFG_LIBS% >> svc\cpsp\win.cpsp.mak
goto :EOF

:set_db_directories
if defined POSTGRES_DIR set OPT_TARGETS=libcp_dbms_postgres.dll
if defined MYSQL_DIR set OPT_TARGETS=%OPT_TARGETS% libcp_dbms_mysql.dll
goto :EOF

:set_db_objects
if defined POSTGRES_DIR (
	set OPT_OBJS=%OPT_OBJS% db_postgres.obj
	set CFG_CFLAGS=%CFG_CFLAGS% /I%POSTGRES_DIR%\include
	set CFG_LDFLAGS=%CFG_LDFLAGS% /libpath:%POSTGRES_DIR%\lib /libpath:%POSTGRES_DIR%\lib\ms libpq.lib
)
if defined MYSQL_DIR (
	set OPT_OBJS=%OPT_OBJS% db_mysql.obj
	set CFG_CFLAGS=%CFG_CFLAGS% /I"%MYSQL_DIR%"\include
	set CFG_LDFLAGS=%CFG_LDFLAGS% /libpath:"%MYSQL_DIR%"\lib\opt mysqlclient.lib
	set CFG_LIBS=%CFG_LIBS% crypt32.lib advapi32.lib
)
goto :EOF

:write_setpath
if exist %SETPATH% del /q %SETPATH%
set PATHSTR=%%^PATH%%
echo set PCRE=> %SETPATH%
echo match "%PATHSTR%" "%PCRE_DIR%">> %SETPATH%
echo for /f "tokens=*" %%^%%^i in (__match) do set PCRE=%%^%%^i>> %SETPATH%
echo if defined PCRE goto :CPROPS>> %SETPATH%
echo @set PATH=%%^PATH%%;%PCRE_DIR%\lib>> %SETPATH%
newline>> %SETPATH%
echo :CPROPS>> %SETPATH%
echo set CPDLL=>> %SETPATH%
echo match "%PATHSTR%" "%CWD%">> %SETPATH%
echo for /f "tokens=*" %%^%%^i in (__match) do set CPDLL=%%^%%^i>> %SETPATH%
echo if defined CPDLL goto :CPSPEXE>> %SETPATH%
echo @set PATH=%%^PATH%%;%CWD%>> %SETPATH%
newline>> %SETPATH%
echo :CPSPEXE>> %SETPATH%
echo set CPSPPATH=>> %SETPATH%
echo match "%PATHSTR%" "%CWD%\svc\cpsp">> %SETPATH%
echo for /f "tokens=*" %%^%%^i in (__match) do set CPSPPATH=%%^%%^i>> %SETPATH%
echo if defined CPSPPATH goto :SSL>> %SETPATH%
echo @set PATH=%%^PATH%%;%CWD%\svc\cpsp>> %SETPATH%
newline>> %SETPATH%
echo :SSL>> %SETPATH%
echo set OPENSSL=>> %SETPATH%
echo match "%PATHSTR%" "%SSL_DIR%\lib\vc">> %SETPATH%
echo for /f "tokens=*" %%^%%^i in (__match) do set OPENSSL=%%^%%^i>> %SETPATH%
echo if defined OPENSSL goto :DONE>> %SETPATH%
echo @set PATH=%%^PATH%%;%SSL_DIR%\lib\vc>> %SETPATH%
echo :DONE>> %SETPATH%
newline>> %SETPATH%
echo set PCRE=>> %SETPATH%
echo set CPDLL=>> %SETPATH%
echo set CPSPPATH=>> %SETPATH%
echo set OPENSSL=>> %SETPATH%
set PATHSTR=
goto :EOF

:END
rem clear variables and temporary files
set QUESTION=
set ANSWER=
set NEEDLE=
set HAYSTACK=
set MATCH=
set PREFIX=
set SUFFIX=

set CPROPS_LINK=
set HASHTABLE_MULTIPLES=
set HASHLIST_MULTIPLES=
set USE_COOKIES=
set USE_HTTP_SESSIONS=
set BUILD_CPSVC=
set BUILD_CPSP=
set LEX=
set YACC=
set BUILD_DBMS=
set TARGET=
set OPT_OBJS=
set OPT_TARGETS=
set SUBDIRS=
set PCRE_DIR=
set SSL_DIR=
set POSTGRES_DIR=
set MYSQL_DIR=
set DEBUG=
set __TRACE__=
set CFG_CFLAGS=
set CFG_LDFLAGS=
set LINKFLAGS=
set CFG_LIBS=
set MAIN_CFLAGS=

if exist __suffix del __suffix
if exist __answer del __answer
if exist __prefix del __prefix 
set USE_SSL=

