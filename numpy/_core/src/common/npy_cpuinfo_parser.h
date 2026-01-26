/*
 * Copyright (C) 2010 The Android Open Source Project
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
#ifndef NUMPY_CORE_SRC_COMMON_NPY_CPUINFO_PARSER_H_
#define NUMPY_CORE_SRC_COMMON_NPY_CPUINFO_PARSER_H_
#include <errno.h>
#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stddef.h>

#define NPY__HWCAP  16
#define NPY__HWCAP2 26

#ifdef __arm__
    // arch/arm/include/uapi/asm/hwcap.h
    #define NPY__HWCAP_HALF	    (1 << 1)
    #define NPY__HWCAP_NEON	    (1 << 12)
    #define NPY__HWCAP_VFPv3	(1 << 13)
    #define NPY__HWCAP_VFPv4	(1 << 16)

    #define NPY__HWCAP_FPHP	    (1 << 22)
    #define NPY__HWCAP_ASIMDHP	(1 << 23)
    #define NPY__HWCAP_ASIMDDP	(1 << 24)
    #define NPY__HWCAP_ASIMDFHM	(1 << 25)

    #define NPY__HWCAP2_AES	    (1 << 0)
    #define NPY__HWCAP2_PMULL	(1 << 1)
    #define NPY__HWCAP2_SHA1	(1 << 2)
    #define NPY__HWCAP2_SHA2	(1 << 3)
    #define NPY__HWCAP2_CRC32	(1 << 4)
#else
    // arch/arm64/include/uapi/asm/hwcap.h
    #define NPY__HWCAP_FP		(1 << 0)
    #define NPY__HWCAP_ASIMD	(1 << 1)

    #define NPY__HWCAP_FPHP		(1 << 9)
    #define NPY__HWCAP_ASIMDHP	(1 << 10)
    #define NPY__HWCAP_ASIMDDP	(1 << 20)
    #define NPY__HWCAP_ASIMDFHM	(1 << 23)

    #define NPY__HWCAP_AES		(1 << 3)
    #define NPY__HWCAP_PMULL	(1 << 4)
    #define NPY__HWCAP_SHA1		(1 << 5)
    #define NPY__HWCAP_SHA2		(1 << 6)
    #define NPY__HWCAP_CRC32	(1 << 7)
    #define NPY__HWCAP_SVE		(1 << 22)
#endif


/*
 * Get the size of a file by reading it until the end. This is needed
 * because files under /proc do not always return a valid size when
 * using fseek(0, SEEK_END) + ftell(). Nor can they be mmap()-ed.
 */
static int
get_file_size(const char* pathname)
{
    int fd, result = 0;
    char buffer[256];

    fd = open(pathname, O_RDONLY);
    if (fd < 0) {
        return -1;
    }

    for (;;) {
        int ret = read(fd, buffer, sizeof buffer);
        if (ret < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        if (ret == 0) {
            break;
        }
        result += ret;
    }
    close(fd);
    return result;
}

/*
 * Read the content of /proc/cpuinfo into a user-provided buffer.
 * Return the length of the data, or -1 on error. Does *not*
 * zero-terminate the content. Will not read more
 * than 'buffsize' bytes.
 */
static int
read_file(const char*  pathname, char*  buffer, size_t  buffsize)
{
    int  fd, count;

    fd = open(pathname, O_RDONLY);
    if (fd < 0) {
        return -1;
    }
    count = 0;
    while (count < (int)buffsize) {
        int ret = read(fd, buffer + count, buffsize - count);
        if (ret < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (count == 0) {
                count = -1;
            }
            break;
        }
        if (ret == 0) {
            break;
        }
        count += ret;
    }
    close(fd);
    return count;
}

/*
 * Extract the content of a the first occurrence of a given field in
 * the content of /proc/cpuinfo and return it as a heap-allocated
 * string that must be freed by the caller.
 *
 * Return NULL if not found
 */
static char*
extract_cpuinfo_field(const char* buffer, int buflen, const char* field)
{
    int fieldlen = strlen(field);
    const char* bufend = buffer + buflen;
    char* result = NULL;
    int len;
    const char *p, *q;

    /* Look for first field occurrence, and ensures it starts the line. */
    p = buffer;
    for (;;) {
        p = memmem(p, bufend-p, field, fieldlen);
        if (p == NULL) {
            goto EXIT;
        }

        if (p == buffer || p[-1] == '\n') {
            break;
        }

        p += fieldlen;
    }

    /* Skip to the first column followed by a space */
    p += fieldlen;
    p = memchr(p, ':', bufend-p);
    if (p == NULL || p[1] != ' ') {
        goto EXIT;
    }

    /* Find the end of the line */
    p += 2;
    q = memchr(p, '\n', bufend-p);
    if (q == NULL) {
        q = bufend;
    }

    /* Copy the line into a heap-allocated buffer */
    len = q - p;
    result = malloc(len + 1);
    if (result == NULL) {
        goto EXIT;
    }

    memcpy(result, p, len);
    result[len] = '\0';

EXIT:
    return result;
}

/*
 * Checks that a space-separated list of items contains one given 'item'.
 * Returns 1 if found, 0 otherwise.
 */
static int
has_list_item(const char* list, const char* item)
{
    const char* p = list;
    int itemlen = strlen(item);

    if (list == NULL) {
        return 0;
    }

    while (*p) {
        const char*  q;

        /* skip spaces */
        while (*p == ' ' || *p == '\t') {
            p++;
        }

        /* find end of current list item */
        q = p;
        while (*q && *q != ' ' && *q != '\t') {
            q++;
        }

        if (itemlen == q-p && !memcmp(p, item, itemlen)) {
            return 1;
        }

        /* skip to next item */
        p = q;
    }
    return 0;
}

static int
get_feature_from_proc_cpuinfo(unsigned long *hwcap, unsigned long *hwcap2) {
    *hwcap = 0;
    *hwcap2 = 0;

    int cpuinfo_len = get_file_size("/proc/cpuinfo");
    if (cpuinfo_len < 0) {
        return 0;
    }
    char *cpuinfo = malloc(cpuinfo_len);
    if (cpuinfo == NULL) {
        return 0;
    }

    cpuinfo_len = read_file("/proc/cpuinfo", cpuinfo, cpuinfo_len);
    char *cpuFeatures = extract_cpuinfo_field(cpuinfo, cpuinfo_len, "Features");
    if (cpuFeatures == NULL) {
        free(cpuinfo);
        return 0;
    }
    *hwcap |= has_list_item(cpuFeatures, "fphp") ? NPY__HWCAP_FPHP : 0;
    *hwcap |= has_list_item(cpuFeatures, "asimdhp") ? NPY__HWCAP_ASIMDHP : 0;
    *hwcap |= has_list_item(cpuFeatures, "asimddp") ? NPY__HWCAP_ASIMDDP : 0;
    *hwcap |= has_list_item(cpuFeatures, "asimdfhm") ? NPY__HWCAP_ASIMDFHM : 0;
#ifdef __arm__
    *hwcap |= has_list_item(cpuFeatures, "neon") ? NPY__HWCAP_NEON : 0;
    *hwcap |= has_list_item(cpuFeatures, "half") ? NPY__HWCAP_HALF : 0;
    *hwcap |= has_list_item(cpuFeatures, "vfpv3") ? NPY__HWCAP_VFPv3 : 0;
    *hwcap |= has_list_item(cpuFeatures, "vfpv4") ? NPY__HWCAP_VFPv4 : 0;
    *hwcap2 |= has_list_item(cpuFeatures, "aes") ? NPY__HWCAP2_AES : 0;
    *hwcap2 |= has_list_item(cpuFeatures, "pmull") ? NPY__HWCAP2_PMULL : 0;
    *hwcap2 |= has_list_item(cpuFeatures, "sha1") ? NPY__HWCAP2_SHA1 : 0;
    *hwcap2 |= has_list_item(cpuFeatures, "sha2") ? NPY__HWCAP2_SHA2 : 0;
    *hwcap2 |= has_list_item(cpuFeatures, "crc32") ? NPY__HWCAP2_CRC32 : 0;
#else
    *hwcap |= has_list_item(cpuFeatures, "asimd") ? NPY__HWCAP_ASIMD : 0;
    *hwcap |= has_list_item(cpuFeatures, "fp") ? NPY__HWCAP_FP : 0;
    *hwcap |= has_list_item(cpuFeatures, "aes") ? NPY__HWCAP_AES : 0;
    *hwcap |= has_list_item(cpuFeatures, "pmull") ? NPY__HWCAP_PMULL : 0;
    *hwcap |= has_list_item(cpuFeatures, "sha1") ? NPY__HWCAP_SHA1 : 0;
    *hwcap |= has_list_item(cpuFeatures, "sha2") ? NPY__HWCAP_SHA2 : 0;
    *hwcap |= has_list_item(cpuFeatures, "crc32") ? NPY__HWCAP_CRC32 : 0;
#endif
    free(cpuinfo);
    free(cpuFeatures);
    return 1;
}
#endif  /* NUMPY_CORE_SRC_COMMON_NPY_CPUINFO_PARSER_H_ */
