/*
 * TODO:
 *  - test for cpuid availability
 *  - test for OS support (tricky)
 */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "cpuid.h"

#ifndef __GNUC__
#error "Sorry, this code can only be compiled with gcc for now"
#endif

/*
 * SIMD: SSE 1, 2 and 3, MMX
 */
#define CPUID_FLAG_MMX  1 << 23 /* in edx */
#define CPUID_FLAG_SSE  1 << 25 /* in edx */
#define CPUID_FLAG_SSE2 1 << 26 /* in edx */
#define CPUID_FLAG_SSE3 1 << 0  /* in ecx */

/*
 * long mode (AMD64 instruction set)
 */
#define CPUID_FLAGS_LONG_MODE   1 << 29 /* in edx */

/*
 * struct reprensenting the cpuid flags as put in the register
 */
typedef struct {
        uint32_t eax;
        uint32_t ebx;
        uint32_t ecx;
        uint32_t edx;
} cpuid_t;

/*
 * Union to read bytes in 32 (intel) bits registers
 */
union _le_reg {
        uint8_t ccnt[4];
        uint32_t reg;
} __attribute__ ((packed)); 
typedef union _le_reg le_reg_t ;

/*
 * can_cpuid and read_cpuid are the two only functions using asm
 */
static int can_cpuid(void)
{
    	int has_cpuid = 0 ;

	/*
 	 * See intel doc on cpuid (pdf)
 	 */
    	asm volatile (
      		"pushfl			\n\t"
      		"popl %%eax		\n\t"
      		"movl %%eax, %%ecx	\n\t"
      		"xorl $0x200000, %%eax	\n\t"
      		"pushl %%eax		\n\t"
      		"popfl			\n\t"
      		"pushfl			\n\t"
      		"popl %%eax		\n\t"
      		"xorl %%ecx, %%eax	\n\t"
      		"andl $0x200000, %%eax	\n\t"
      		"movl %%eax,%0		\n\t"
    		:"=m" (has_cpuid)
    		: /*no input*/
    		: "eax","ecx","cc");

    	return (has_cpuid != 0) ;
}

/*
 * func is the "level" of cpuid. See for cpuid.txt
 */
static cpuid_t read_cpuid(unsigned int func)
{
        cpuid_t res; 

	/* we save ebx because it is used when compiled by -fPIC */
        asm volatile(
                "pushl %%ebx      \n\t" /* save %ebx */
                "cpuid            \n\t"
                "movl %%ebx, %1   \n\t" /* save what cpuid just put in %ebx */
                "popl %%ebx       \n\t" /* restore the old %ebx */
                : "=a"(res.eax), "=r"(res.ebx), 
                  "=c"(res.ecx), "=d"(res.edx)
                : "a"(func)
                : "cc"); 

        return res;
}

static uint32_t get_max_func()
{
        cpuid_t cpuid;

        cpuid = read_cpuid(0);
        return cpuid.eax;
}

/*
 * vendor should have at least CPUID_VENDOR_STRING_LEN characters
 */
static int get_vendor_string(cpuid_t cpuid, char vendor[])
{
        int i;
        le_reg_t treg;

        treg.reg = cpuid.ebx;
        for (i = 0; i < 4; ++i) {
                vendor[i] = treg.ccnt[i];
        }

        treg.reg = cpuid.edx;
        for (i = 0; i < 4; ++i) {
                vendor[i+4] = treg.ccnt[i];
        }

        treg.reg = cpuid.ecx;
        for (i = 0; i < 4; ++i) {
                vendor[i+8] = treg.ccnt[i];
        }
        vendor[12] = '\0';
        return 0;
}

int cpuid_get_caps(cpu_caps_t *cpu)
{
	cpuid_t cpuid;
	int max;

	memset(cpu, 0, sizeof(*cpu));

	if (!can_cpuid()) {
		return 0;
	}

	max = get_max_func();

	/* Read vendor string */
	cpuid = read_cpuid(0);
	get_vendor_string(cpuid, cpu->vendor);
	
	if (max < 0x00000001) {
		return 0;
	}
	cpuid = read_cpuid(0x00000001);

	/* We can read mmx, sse 1 2 and 3 when cpuid level >= 0x00000001 */
        if (cpuid.edx & CPUID_FLAG_MMX) {
		cpu->has_mmx = 1;
	}
        if (cpuid.edx & CPUID_FLAG_SSE) {
		cpu->has_sse = 1;
	}
        if (cpuid.edx & CPUID_FLAG_SSE2) {
		cpu->has_sse2 = 1;
	}
        if (cpuid.ecx & CPUID_FLAG_SSE3) {
		cpu->has_sse3 = 1;
	}
	return 0;
}
