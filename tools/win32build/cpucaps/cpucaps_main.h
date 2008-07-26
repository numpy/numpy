#ifndef _EXDLL_H_
#define _EXDLL_H_

#include <windows.h>

#if defined(__GNUC__)
#define UNUSED __attribute__((unused))
#else
#define UNUSED
#endif

// only include this file from one place in your DLL.
// (it is all static, if you use it in two places it will fail)

#define EXDLL_INIT()           {  \
        g_stringsize=string_size; \
        g_stacktop=stacktop;      \
        g_variables=variables; }

// For page showing plug-ins
#define WM_NOTIFY_OUTER_NEXT (WM_USER+0x8)
#define WM_NOTIFY_CUSTOM_READY (WM_USER+0xd)
#define NOTIFY_BYE_BYE 'x'

typedef struct _stack_t {
  struct _stack_t *next;
  char text[1]; // this should be the length of string_size
} stack_t;


static unsigned int g_stringsize;
static stack_t **g_stacktop;
static char *g_variables;

static int __stdcall popstring(char *str) UNUSED; // 0 on success, 1 on empty stack
static void __stdcall pushstring(const char *str) UNUSED;
static char * __stdcall getuservariable(const int varnum) UNUSED;
static void __stdcall setuservariable(const int varnum, const char *var) UNUSED;

enum
{
INST_0,         // $0
INST_1,         // $1
INST_2,         // $2
INST_3,         // $3
INST_4,         // $4
INST_5,         // $5
INST_6,         // $6
INST_7,         // $7
INST_8,         // $8
INST_9,         // $9
INST_R0,        // $R0
INST_R1,        // $R1
INST_R2,        // $R2
INST_R3,        // $R3
INST_R4,        // $R4
INST_R5,        // $R5
INST_R6,        // $R6
INST_R7,        // $R7
INST_R8,        // $R8
INST_R9,        // $R9
INST_CMDLINE,   // $CMDLINE
INST_INSTDIR,   // $INSTDIR
INST_OUTDIR,    // $OUTDIR
INST_EXEDIR,    // $EXEDIR
INST_LANG,      // $LANGUAGE
__INST_LAST
};

typedef struct {
  int autoclose;
  int all_user_var;
  int exec_error;
  int abort;
  int exec_reboot;
  int reboot_called;
  int XXX_cur_insttype; // deprecated
  int XXX_insttype_changed; // deprecated
  int silent;
  int instdir_error;
  int rtl;
  int errlvl;
  int alter_reg_view;
} exec_flags_type;

typedef struct {
  exec_flags_type *exec_flags;
  int (__stdcall *ExecuteCodeSegment)(int, HWND);
  void (__stdcall *validate_filename)(char *);
} extra_parameters;

// utility functions (not required but often useful)
static int __stdcall popstring(char *str)
{
  stack_t *th;
  if (!g_stacktop || !*g_stacktop) return 1;
  th=(*g_stacktop);
  lstrcpyA(str,th->text);
  *g_stacktop = th->next;
  GlobalFree((HGLOBAL)th);
  return 0;
}

static void __stdcall pushstring(const char *str)
{
  stack_t *th;
  if (!g_stacktop) return;
  th=(stack_t*)GlobalAlloc(GPTR,sizeof(stack_t)+g_stringsize);
  lstrcpynA(th->text,str,g_stringsize);
  th->next=*g_stacktop;
  *g_stacktop=th;
}

static char * __stdcall getuservariable(const int varnum)
{
  if (varnum < 0 || varnum >= __INST_LAST) return NULL;
  return g_variables+varnum*g_stringsize;
}

static void __stdcall setuservariable(const int varnum, const char *var)
{
	if (var != NULL && varnum >= 0 && varnum < __INST_LAST) 
		lstrcpyA(g_variables + varnum*g_stringsize, var);
}



#endif//_EXDLL_H_
