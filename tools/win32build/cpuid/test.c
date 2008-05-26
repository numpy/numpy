#include <stdio.h>

#include "cpuid.h"

int main()
{
	cpu_caps_t *cpuinfo;

	cpuinfo = malloc(sizeof(*cpuinfo));

	if (cpuinfo == NULL) {
		fprintf(stderr, "Error allocating\n");
	}

	cpuid_get_caps(cpuinfo);
	printf("This cpu string is %s\n", cpuinfo->vendor);

	if (cpuinfo->has_mmx) {
		printf("This cpu has mmx instruction set\n");
	} else {
		printf("This cpu does NOT have mmx instruction set\n");
	}

	if (cpuinfo->has_sse) {
		printf("This cpu has sse instruction set\n");
	} else {
		printf("This cpu does NOT have sse instruction set\n");
	}

	if (cpuinfo->has_sse2) {
		printf("This cpu has sse2 instruction set\n");
	} else {
		printf("This cpu does NOT have sse2 instruction set\n");
	}

	if (cpuinfo->has_sse3) {
		printf("This cpu has sse3 instruction set\n");
	} else {
		printf("This cpu does NOT have sse3 instruction set\n");
	}

	free(cpuinfo);
	return 0;
}
