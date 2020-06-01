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
#ifndef __NPY_CPUINFO_PARSER_H__
#define __NPY_CPUINFO_PARSER_H__
#include <errno.h>
#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stddef.h>

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
 * Extract the content of a the first occurence of a given field in
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

    /* Look for first field occurence, and ensures it starts the line. */
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
get_feature_from_proc_cpuinfo(unsigned char npy__cpu_have[NPY_CPU_FEATURE_MAX]) {
    char* cpuinfo = NULL;
    int cpuinfo_len;
    long architecture = 0;
    cpuinfo_len = get_file_size("/proc/cpuinfo");
    if (cpuinfo_len < 0) {
        return 0;
    }
    cpuinfo = malloc(cpuinfo_len);
    if (cpuinfo == NULL) {
        return 0;
    }
    cpuinfo_len = read_file("/proc/cpuinfo", cpuinfo, cpuinfo_len);
    char* cpuArch = extract_cpuinfo_field(cpuinfo, cpuinfo_len, "CPU architecture");
    int isV8 = 0;
    if (cpuArch) {
        architecture = strtol(cpuArch, NULL, 10);
        free(cpuArch);
        if (architecture >= 8L) {
            isV8 = 1;
            npy__cpu_init_features_arm8();
        }
    }
    char* cpuFeatures = extract_cpuinfo_field(cpuinfo, cpuinfo_len, "Features");
    if (cpuFeatures != NULL) {
        npy__cpu_have[NPY_CPU_FEATURE_FPHP]       = has_list_item(cpuFeatures, "fphp");
        npy__cpu_have[NPY_CPU_FEATURE_ASIMDHP]    = has_list_item(cpuFeatures, "asimdhp");
        npy__cpu_have[NPY_CPU_FEATURE_ASIMDDP]    = has_list_item(cpuFeatures, "asimddp");
        npy__cpu_have[NPY_CPU_FEATURE_ASIMDFHM]   = has_list_item(cpuFeatures, "asimdfhm");
        npy__cpu_have[NPY_CPU_FEATURE_NEON]       = has_list_item(cpuFeatures, "neon") ||
                                                    has_list_item(cpuFeatures, "asimd");
        npy__cpu_have[NPY_CPU_FEATURE_NEON_FP16]  = isV8 || has_list_item(cpuFeatures, "neon") ||
                                                     has_list_item(cpuFeatures, "vfpv3") || has_list_item(cpuFeatures, "half");
        npy__cpu_have[NPY_CPU_FEATURE_NEON_VFPV4] = isV8 || has_list_item(cpuFeatures, "vfpv4");
    }
    return 1;
}
#endif