#ifndef _GABOU_CPUID_H 
#define _GABOU_CPUID_H 

#include <stdlib.h>

#define CPUID_VENDOR_STRING_LEN  12

struct _cpu_caps {
	int has_cpuid;
	int has_mmx;
	int has_sse;
	int has_sse2;
	int has_sse3;
	char vendor[CPUID_VENDOR_STRING_LEN+1];
};
typedef struct _cpu_caps cpu_caps_t;

int cpuid_get_caps(cpu_caps_t *cpuinfo);

#endif
