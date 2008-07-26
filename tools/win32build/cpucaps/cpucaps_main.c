#include <stdio.h>

#include <windows.h>
#include "cpucaps_main.h"

#include "cpuid.h"

HINSTANCE g_hInstance;

HWND g_hwndParent;

#define CPUID_FAILED "Unknown"

/*
 * if val is true, str is the "Y" string, otherwise the "N" string
 */
static int _set_bool_str(int val, char* str)
{
	if (val) {
		str[0] = 'Y';
  	} else {
		str[0] = 'N';
  	}
  	str[1] = '\0';

	return 0;
}

void __declspec(dllexport) hasSSE3(HWND hwndParent, int string_size, 
                                   char *variables, stack_t **stacktop,
                                   extra_parameters *extra)
{
  cpu_caps_t *cpu;
  char has_sse3[2];

  //g_hwndParent=hwndParent;

  EXDLL_INIT();


  // note if you want parameters from the stack, pop them off in order.
  // i.e. if you are called via exdll::myFunction file.dat poop.dat
  // calling popstring() the first time would give you file.dat,
  // and the second time would give you poop.dat. 
  // you should empty the stack of your parameters, and ONLY your
  // parameters.

  // do your stuff here
  cpu = malloc(sizeof(*cpu));
  if (cpu == NULL) {
	  fprintf(stderr, "malloc call failed\n");
  	  _set_bool_str(0, has_sse3);
	  goto push_vars;
  }
  cpuid_get_caps(cpu);
  _set_bool_str(cpu->has_sse3, has_sse3);


push_vars:
  pushstring(has_sse3);
  
  return ;
}


void __declspec(dllexport) hasSSE2(HWND hwndParent, int string_size, 
                                   char *variables, stack_t **stacktop,
                                   extra_parameters *extra)
{
  cpu_caps_t *cpu;
  char has_sse2[2];

  //g_hwndParent=hwndParent;

  EXDLL_INIT();


  // note if you want parameters from the stack, pop them off in order.
  // i.e. if you are called via exdll::myFunction file.dat poop.dat
  // calling popstring() the first time would give you file.dat,
  // and the second time would give you poop.dat. 
  // you should empty the stack of your parameters, and ONLY your
  // parameters.

  // do your stuff here
  cpu = malloc(sizeof(*cpu));
  if (cpu == NULL) {
	  fprintf(stderr, "malloc call failed\n");
  	  _set_bool_str(0, has_sse2);
	  goto push_vars;
  }
  cpuid_get_caps(cpu);
  _set_bool_str(cpu->has_sse2, has_sse2);


push_vars:
  pushstring(has_sse2);
  
  return ;
}



BOOL WINAPI DllMain(HANDLE hInst, ULONG ul_reason_for_call, LPVOID lpReserved)
{
  g_hInstance=hInst;
	return TRUE;
}
