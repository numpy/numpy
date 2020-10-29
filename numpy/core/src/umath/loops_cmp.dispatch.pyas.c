/*@targets
 ** $werror $maxopt baseline
 ** sse2 xop avx2
 ** vsx vsx2 vsx3
 ** neon
 **/
#define NPY_NO_EXPORT NPY_VISIBILITY_HIDDEN

#include "numpy/npy_common.h"
#include "numpy/npy_math.h" // npy_clear_floatstatus_barrier
#include "numpy/halffloat.h"
#include "simd/simd.h"
#include "loops.h"
#include "fast_loop_macros.h" // BINARY_LOOP

NPY_FINLINE int
is_overlap(char *src, npy_intp ssrc, char *dst, npy_intp sdst, int wstep)
{
    if (ssrc && sdst) {
        char *src_step1 = src + ssrc*wstep;
        char *dst_step1 = dst + sdst*wstep;
        if (dst_step1 > src_step1) {
            return (dst_step1 - src_step1) < (ssrc*wstep);
        }
        if (dst_step1 < src_step1) {
            return (src_step1 - dst_step1) < (ssrc*wstep);
        }
    }
    return 0;
}
<?python
#//###############################################################################
#//## Defining the SIMD kernels
#//###############################################################################
from itertools import product
def sel(stype, scalar='', contig='', ncontig=''):
    if stype == 'scalar':
        return scalar
    if stype == 'contig':
        return contig
    return ncontig

types_suffixes = (
    'u8', 's8', 'u16', 's16', 'u32', 's32', 'u64', 's64', 'f32', 'f64'
)
operations = (
    # ufunc name      npyv intrin    op
    ('equal',         'cmpeq',       '=='), #// for unsigned only, signed mapped
    ('not_equal',     'cmpneq',      '!='), #// for unsigned only, signed mapped
    ('greater',       'cmpgt',       '>'),  #// for both
    ('greater_equal', 'cmpge',       '>='), #// for both
    ('less',          'cmplt',       '<'),  #// swap and map to greater
    ('less_equal',    'cmple',       '<='), #// swap and map to greater_equal
    ('logical_and',   'and',         '&&'), #// for unsigned only, signed mapped
    ('logical_or',    'or',          '||'), #// for unsigned only, signed mapped
)
stride_types = (
    # steps[0]  steps[1]   steps[2]
    ('contig',  'contig',  'contig'),  #// for both sources
    ('contig',  'contig',  'ncontig'), #// for both sources
    ('ncontig', 'contig',  'contig'),  #// swap and map the second
    ('contig',  'ncontig', 'contig'),  #// only for gt and ge
    ('scalar',  'contig',  'contig'),  #// swap and map the second
    ('contig',  'scalar',  'contig'),  #// only for gt and ge
)
for sfx, (ufunc_name, intrin, op), (s0type, s1type, d0type) in product(
    types_suffixes, operations, stride_types
):
    #//###########################################################################
    #//## Skip mappable types and operations
    #//###########################################################################
    if ufunc_name in ('less', 'less_equal'):
        continue
    if intrin not in ('cmpgt', 'cmpge'):
        if sfx[0] == 's':
            continue
        if s0type != s1type and s1type in ('scalar', 'ncontig'):
            continue
    #//###########################################################################
    #//## Constants
    #//###########################################################################
    simd_func_name = f"simd_{ufunc_name}_{sfx}_{s0type}_{s1type}_{d0type}"
    num_of_loads = int(int(sfx[1:]) / 8)
    #//###########################################################################
    #//## Load template translations
    #//###########################################################################
    tr('_sfx',    f'_{sfx}')
    tr('_bsfx',   f'_b{sfx[1:]}')
    tr('_intrin', f'_{intrin}')
    tr('_s0type', f'_{s0type}')
    tr('_s1type', f'_{s1type}')
    tr('_d0type', f'_{d0type}')

    tr('IS_LOGICAL', int(ufunc_name.startswith('logical')), '/*IS_LOGICAL*/')
    tr('NUM_OF_LOADS', num_of_loads, '/*NUM_OF_LOADS*/')
    #// we use C directives instead of Python to increase the readability
    for stride_type in ('contig', 'ncontig', 'scalar'):
        cdef_name0 = 'IS_S0TYPE_' + stride_type.upper()
        cdef_name1 = 'IS_S1TYPE_' + stride_type.upper()
        tr(cdef_name0, int(s0type == stride_type), f'/*{cdef_name0}*/')
        tr(cdef_name1, int(s1type == stride_type), f'/*{cdef_name1}*/')

    if sfx.startswith('f64'):
        tr('NPY_SIMD', 'NPY_SIMD_F64')

    #//###########################################################################
    #//## Template begin
    #//###########################################################################
    <0%
    #line @LINENO@
    static void {{simd_func_name}}(char **args, npy_intp const *dimensions, npy_intp const *steps)
    {
        // types
        const npyv_lanetype_sfx *src0 = (npyv_lanetype_sfx*)args[0];
        const npyv_lanetype_sfx *src1 = (npyv_lanetype_sfx*)args[1];
              npyv_lanetype_u8  *dst  = (npyv_lanetype_u8*)args[2];
                         npy_intp len = dimensions[0];
        // strides
        const npy_intp ssrc0 = {{sel(s0type, 0, 1, 'steps[0] / sizeof(src0[0])')}};
        const npy_intp ssrc1 = {{sel(s1type, 0, 1, 'steps[1] / sizeof(src1[0])')}};
        const npy_intp sdst  = {{sel(d0type, 0, 1, 'steps[2]')}};
    //  0. SIMD Check
    #if NPY_SIMD
        #if IS_LOGICAL
            const npyv_sfx v_zero = npyv_zero_sfx();
            #if IS_S0TYPE_SCALAR
                const npyv_bsfx v_scalar0 = npyv_cmpneq_sfx(npyv_setall_sfx(*src0), v_zero);
            #endif
            #if IS_S1TYPE_SCALAR
                const npyv_bsfx v_scalar1 = npyv_cmpneq_sfx(npyv_setall_sfx(*src1), v_zero);
            #endif
        #else
            #if IS_S0TYPE_SCALAR
                const npyv_sfx v_scalar0 = npyv_setall_sfx(*src0);
            #endif
            #if IS_S1TYPE_SCALAR
                const npyv_sfx v_scalar1 = npyv_setall_sfx(*src1);
            #endif
        #endif
    //  2. loop begain
        const npyv_u8 v_one = npyv_setall_u8(1);
        const int wstep = npyv_nlanes_u8;
        for (; len >= wstep; len -= wstep, src0 += wstep*ssrc0, src1 += wstep*ssrc1, dst += wstep*sdst)
        {
    %0>
    #// 1. load
    for n in range(0, num_of_loads):
        <2%
            #line @LINENO@
            #if IS_S0TYPE_CONTIG
                npyv_sfx v_src0{{n}} = npyv_load_sfx(
                    src0 + npyv_nlanes_sfx*{{n}}
                );
            #elif IS_S0TYPE_NCONTIG
                npyv_sfx v_src0{{n}} = npyv_loadn_sfx(
                    src0 + ssrc0 * npyv_nlanes_sfx * {{n}}, (int)ssrc0
                );
            #endif
            #if IS_S1TYPE_CONTIG
                npyv_sfx v_src1{{n}} = npyv_load_sfx(
                    src1 + npyv_nlanes_sfx*{{n}}
                );
            #elif IS_S1TYPE_NCONTIG
                npyv_sfx v_src1{{n}} = npyv_loadn_sfx(
                    src1 + ssrc1 * npyv_nlanes_sfx * {{n}}, (int)ssrc1
                );
            #endif
        %2>
    #// 2. Test
    for n in range(0, num_of_loads):
        <2%
            #line @LINENO@
            #if IS_LOGICAL
                #if IS_S0TYPE_SCALAR
                    npyv_bsfx v_mask{{n}} = npyv_intrin_bsfx(
                        v_scalar0, npyv_cmpneq_sfx(v_src1{{n}}, v_zero)
                    );
                #elif IS_S1TYPE_SCALAR
                    npyv_bsfx v_mask{{n}} = npyv_intrin_bsfx(
                        npyv_cmpneq_sfx(v_src0{{n}}, v_zero), v_scalar1
                    );
                #else
                    npyv_bsfx v_mask{{n}} = npyv_intrin_bsfx(
                        npyv_cmpneq_sfx(v_src0{{n}}, v_zero),
                        npyv_cmpneq_sfx(v_src1{{n}}, v_zero)
                    );
                #endif
            #else
                #if IS_S0TYPE_SCALAR
                    npyv_bsfx v_mask{{n}} = npyv_intrin_sfx(v_scalar0, v_src1{{n}});
                #elif IS_S1TYPE_SCALAR
                    npyv_bsfx v_mask{{n}} = npyv_intrin_sfx(v_src0{{n}}, v_scalar1);
                #else
                    npyv_bsfx v_mask{{n}} = npyv_intrin_sfx(v_src0{{n}}, v_src1{{n}});
                #endif
            #endif
        %2>
    #// 3. Pack & Store
    <0%
            #line @LINENO@
            #if NUM_OF_LOADS == 2
                npyv_b8 v_mask = npyv_pack_b16(v_mask0, v_mask1);
            #elif NUM_OF_LOADS == 4
                npyv_b8 v_mask = npyv_pack_b8_b32(v_mask0, v_mask1, v_mask2, v_mask3);
            #elif NUM_OF_LOADS == 8
                npyv_b8 v_mask = npyv_pack_b8_b64(
                    v_mask0, v_mask1, v_mask2, v_mask3,
                    v_mask4, v_mask5, v_mask6, v_mask7
                );
            #else
                npyv_b8 v_mask = v_mask0;
            #endif
            npyv_u8 vu_mask = npyv_and_u8(npyv_cvt_u8_b8(v_mask), v_one);
            #if D0TYPE_IS_CONTIG
                npyv_store_u8(dst, vu_mask);
            #else
                npyv_storen_u8(dst, (int)sdst, vu_mask);
            #endif
        }
        npyv_cleanup();
    // 4 - unroll for non-supported architectures
    #elif !defined(NPY_DISABLE_OPTIMIZATION)
        for (; len >= 4; len -= 4, src0 += ssrc0*4, src1 += ssrc1*4, dst += sdst*4)
        {
    %0>
    for n in range(0, 4):<2%
            const npyv_lanetype_sfx src0{{n}} = src0[ssrc0*{{n}}];
            const npyv_lanetype_sfx src1{{n}} = src1[ssrc1*{{n}}];
    %2>
    for n in range(0, 4):<2%
            dst[sdst*{{n}}] = src0{{n}} {{op}} src1{{n}};
    %2>
    <0%
        }
    #endif // NPY_SIMD
    // 5- The rest of scalars
        for (; len > 0; --len, src0 += ssrc0, src1 += ssrc1, dst += sdst)
        {
            const npyv_lanetype_sfx src00 = *src0;
            const npyv_lanetype_sfx src10 = *src1;
            *dst = src00 {{op}} src10;
        }
    } // end of {simd_func_name}
    %0>
    #//###########################################################################
    #//## Template End
    #//###########################################################################
    clear_tr()

#//###############################################################################
#//## Defining the ufunc inner functions
#//###############################################################################
long_types_suffixes = (
    'u8', 's8', 'u16', 's16', 'u32', 's32', 'u64', 's64', 'f32', 'f64', 'ULONG', 'LONG'
)
types_names = (
    'UBYTE', 'BYTE', 'USHORT', 'SHORT', 'UINT', 'INT',
    'ULONGLONG', 'LONGLONG', 'FLOAT', 'DOUBLE', 'ULONG', 'LONG'
)
for (sfx, tp_name), (ufunc_name, intrin, op) in product(
    zip(long_types_suffixes, types_names), operations
):
    #//###########################################################################
    #//## Constants
    #//###########################################################################
    full_ufunc_name  = f"{tp_name}_{ufunc_name}"
    simd_func_prefix = f"simd_{ufunc_name}_{sfx}"
    #//###########################################################################
    #//## Mapping LONG/ULONG
    #//###########################################################################
    if tp_name in ('LONG', 'ULONG'):
        U = 'U' if tp_name[0] == 'U' else ''
        <0%
        #line @LINENO@
        NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX({{full_ufunc_name}})
        (char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
        {
            if (sizeof(npy_long) == sizeof(npy_int)) {
                NPY_CPU_DISPATCH_CURFX({{U}}INT_{{ufunc_name}})(args, dimensions, steps, func);
            } else {
                NPY_CPU_DISPATCH_CURFX({{U}}LONGLONG_{{ufunc_name}})(args, dimensions, steps, func);
            }
        }
        %0>
        continue
    #//###########################################################################
    #//## Mapping less to greater
    #//###########################################################################
    if ufunc_name in ('less', 'less_equal'):
        map_to = f"{tp_name}_greater{ufunc_name[4:]}"
        <0%
        #line @LINENO@
        NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX({{full_ufunc_name}})
        (char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
        {
            char *nargs[3] = {args[1], args[0], args[2]};
            npy_intp nsteps[3] = {steps[1], steps[0], steps[2]};
            NPY_CPU_DISPATCH_CURFX({{map_to}})(nargs, dimensions, nsteps, func);
        }
        %0>
        continue
    #//###########################################################################
    #//## Mapping signed to unsigned
    #//###########################################################################
    if ufunc_name not in ('greater', 'greater_equal') and sfx[0] == 's':
        map_to = f"U{tp_name}_{ufunc_name}"
        <0%
        #line @LINENO@
        NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX({{full_ufunc_name}})
        (char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
        {
            NPY_CPU_DISPATCH_CURFX({{map_to}})(args, dimensions, steps, func);
        }
        %0>
        continue
    #//###########################################################################
    #//## Defining the ufunc inner functions
    #//###########################################################################
    tr('IS_GT', int(ufunc_name in ('greater', 'greater_equal')), '/*IS_GT*/')
    tr('funcp', simd_func_prefix)
    <0%
    #line @LINENO@
    NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX({{full_ufunc_name}})
    (char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
    {
        const npy_intp s0 = steps[0], s1 = steps[1], d0 = steps[2];
        const int sz = sizeof(npyv_lanetype_{{sfx}});
        const int dz = sizeof(npyv_lanetype_u8);
        npy_intp len = dimensions[0];
    #if NPY_SIMD
        const int unroll_size = npyv_nlanes_u8;
    #else
        // we unroll by 4 but 16 is a good hint
        const int unroll_size = 16;
    #endif
        // TODO: check alignment size for arm7
        // TODO: check max stride size for non-contig
        // TODO: specialize rgba channels and reverse
        // TODO: specialize broadcast
        if (sizeof(npy_bool) != 1 || len < unroll_size || (s0 == 0 && s1 == 0)) {
            goto NO_SIMD;
        }
        if (is_overlap(args[0], s0, args[2], d0, unroll_size) ||
            is_overlap(args[1], s1, args[2], d0, unroll_size)
        ) {
            goto NO_SIMD;
        }
        if (s0==sz && s1==sz && d0==dz) {
            funcp_contig_contig_contig(args, dimensions, steps);
        }
        else if (s0==sz && s1==sz && d0!=0) {
            funcp_contig_contig_ncontig(args, dimensions, steps);
        }
        else if (s0==0 && s1==sz && d0==dz) {
            funcp_scalar_contig_contig(args, dimensions, steps);
        }
        else if (/*s0!=0*/s1==sz && d0==dz) {
            funcp_ncontig_contig_contig(args, dimensions, steps);
        }
    #if IS_GT
        else if (s0==sz && s1==0 && d0==dz) {
            funcp_contig_scalar_contig(args, dimensions, steps);
        }
        else if (s0==sz && /*s!=0*/ d0==dz) {
            funcp_contig_ncontig_contig(args, dimensions, steps);
        }
    #else
        else if (s0==sz && s1==0 && d0==dz)
        {
            char *nargs[3] = {args[1], args[0], args[2]};
            npy_intp nsteps[3] = {steps[1], steps[0], steps[2]};
            funcp_scalar_contig_contig(nargs, dimensions, nsteps);
        }
        else if (s0==sz && /*s!=0*/ d0==dz)
        {
            char *nargs[3] = {args[1], args[0], args[2]};
            npy_intp nsteps[3] = {steps[1], steps[0], steps[2]};
            funcp_ncontig_contig_contig(nargs, dimensions, nsteps);
        }
    #endif
        else {
        NO_SIMD:;
            const char *src0 = args[0], *src1 = args[1];
                  char *dst  = args[2];
            for (; len > 0; --len, src0 += s0, src1 += s1, dst += d0)
            {
                const npyv_lanetype_{{sfx}} s0 = *((npyv_lanetype_{{sfx}}*)src0);
                const npyv_lanetype_{{sfx}} s1 = *((npyv_lanetype_{{sfx}}*)src1);
                *((npy_bool*)dst)= s0 {{op}} s1;
            }
        }
        {{"npy_clear_floatstatus_barrier((char*)dimensions);" if sfx[0] == 'f' else ''}}
    } // end of {{full_ufunc_name}}
    %0>
    clear_tr()
?>
/**************************************************************************
 ** TODO: Optimize the following
 **************************************************************************/
<?python
for ufunc_name, op in (
    ('equal', '=='), ('not_equal', '!='), ('greater', '>'),
    ('greater_equal', '>='), ('less', '<'), ('less_equal', '<='),
):
    <0%
    #line @LINENO@
    NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(BOOL_{{ufunc_name}})
    (char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
    {
        BINARY_LOOP {
            npy_bool in1 = *((npy_bool *)ip1) != 0;
            npy_bool in2 = *((npy_bool *)ip2) != 0;
            *((npy_bool *)op1)= in1 {{op}} in2;
        }
    }
    %0>

for ufunc_name, op, sc in (
    ('logical_and', '&&', '=='), ('logical_or', '||', '!=')
):
    tr('IS_AND', int(op == '&&'), '/*IS_AND*/')
    <0%
    #line @LINENO@
    NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(BOOL_{{ufunc_name}})
    (char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
    {
        if(IS_BINARY_REDUCE) {
            /* for now only use libc on 32-bit/non-x86 */
            if (steps[1] == 1) {
                npy_bool * op = (npy_bool *)args[0];
        #if IS_AND
                /* np.all(), search for a zero (false) */
                if (*op) {
                    *op = memchr(args[1], 0, dimensions[0]) == NULL;
                }
        #else
                /*
                 * np.any(), search for a non-zero (true) via comparing against
                 * zero blocks, memcmp is faster than memchr on SSE4 machines
                 * with glibc >= 2.12 and memchr can only check for equal 1
                 */
                static const npy_bool zero[4096]; /* zero by C standard */
                npy_uintp i, n = dimensions[0];

                for (i = 0; !*op && i < n - (n % sizeof(zero)); i += sizeof(zero)) {
                    *op = memcmp(&args[1][i], zero, sizeof(zero)) != 0;
                }
                if (!*op && n - i > 0) {
                    *op = memcmp(&args[1][i], zero, n - i) != 0;
                }
        #endif // IS_AND
                return;
            }
            else {
                BINARY_REDUCE_LOOP(npy_bool) {
                    const npy_bool in2 = *(npy_bool *)ip2;
                    io1 = io1 {{op}} in2;
                    if (io1 {{sc}} 0) {
                        break;
                    }
                }
                *((npy_bool *)iop1) = io1;
            }
        }
        else {
            BINARY_LOOP {
                const npy_bool in1 = *(npy_bool *)ip1;
                const npy_bool in2 = *(npy_bool *)ip2;
                *((npy_bool *)op1) = in1 {{op}} in2;
            }
        }
    }
    %0>

for ufunc_name, op in (
    ('equal', '=='), ('not_equal', '!='), ('greater', '>'),
    ('greater_equal', '>='), ('less', '<'), ('less_equal', '<='),
    ('logical_and', '&&'), ('logical_or', '||')
):
    <0%
    #line @LINENO@
    NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(LONGDOUBLE_{{ufunc_name}})
    (char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
    {
        BINARY_LOOP {
            const npy_longdouble in1 = *(npy_longdouble *)ip1;
            const npy_longdouble in2 = *(npy_longdouble *)ip2;
            *((npy_bool *)op1) = in1 {{op}} in2;
        }
        npy_clear_floatstatus_barrier((char*)dimensions);
    }
    %0>
?>
#define _HALF_LOGICAL_AND(a,b) (!npy_half_iszero(a) && !npy_half_iszero(b))
#define _HALF_LOGICAL_OR(a,b) (!npy_half_iszero(a) || !npy_half_iszero(b))
<?python
for ufunc_name, op in (
    ('equal', 'npy_half_eq'), ('not_equal', 'npy_half_ne'), ('greater', 'npy_half_gt'),
    ('greater_equal', 'npy_half_ge'), ('less', 'npy_half_lt'), ('less_equal', 'npy_half_le'),
    ('logical_and', '_HALF_LOGICAL_AND'), ('logical_or', '_HALF_LOGICAL_OR')
):
    <0%
    #line @LINENO@
    NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(HALF_{{ufunc_name}})
    (char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
    {
        BINARY_LOOP {
            const npy_half in1 = *(npy_half *)ip1;
            const npy_half in2 = *(npy_half *)ip2;
            *((npy_bool *)op1) = {{op}}(in1, in2);
        }
    }
    %0>
?>
#undef _HALF_LOGICAL_AND
#undef _HALF_LOGICAL_OR

#define CGE(xr,xi,yr,yi)  ((xr > yr && !npy_isnan(xi) && !npy_isnan(yi)) \
                          || (xr == yr && xi >= yi))
#define CLE(xr,xi,yr,yi)  ((xr < yr && !npy_isnan(xi) && !npy_isnan(yi)) \
                          || (xr == yr && xi <= yi))
#define CGT(xr,xi,yr,yi)  ((xr > yr && !npy_isnan(xi) && !npy_isnan(yi)) \
                          || (xr == yr && xi > yi))
#define CLT(xr,xi,yr,yi)  ((xr < yr && !npy_isnan(xi) && !npy_isnan(yi)) \
                          || (xr == yr && xi < yi))
#define CEQ(xr,xi,yr,yi)  (xr == yr && xi == yi)
#define CNE(xr,xi,yr,yi)  (xr != yr || xi != yi)
#define COR(xr,xi,yr,yi)  (xr || xi) || (yr || yi)
#define CAND(xr,xi,yr,yi) (xr || xi) && (yr || yi)

<?python
types = (
    'npy_float', 'npy_double', 'npy_longdouble'
)
operations = (
    # ufunc name      cdef
    ('equal',         'CEQ'),
    ('not_equal',     'CNE'),
    ('greater',       'CGT'),
    ('greater_equal', 'CGE'),
    ('less',          'CLT'),
    ('less_equal',    'CLE'),
    ('logical_and',   'CAND'),
    ('logical_or',    'COR'),
)
for type, (ufunc_name, cdef) in product(types, operations):
    TYPE = f'C{type[4:].upper()}'
    tr('auto', type) #// :)
    <0%
    #line @LINENO@
    NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX({{TYPE}}_{{ufunc_name}})
    (char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
    {
        BINARY_LOOP {
            const auto in1r = ((auto *)ip1)[0];
            const auto in1i = ((auto *)ip1)[1];
            const auto in2r = ((auto *)ip2)[0];
            const auto in2i = ((auto *)ip2)[1];
            *((npy_bool *)op1) = {{cdef}}(in1r,in1i,in2r,in2i);
        }
    }
    %0>
?>
