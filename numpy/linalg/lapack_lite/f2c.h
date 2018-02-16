/* f2c.h  --  Standard Fortran to C header file */

/**  barf  [ba:rf]  2.  "He suggested using FORTRAN, and everybody barfed."

	- From The Shogakukan DICTIONARY OF NEW ENGLISH (Second edition) */

#ifndef F2C_INCLUDE
#define F2C_INCLUDE

typedef int integer;
typedef char *address;
typedef short int shortint;
typedef float real;
typedef double doublereal;
typedef struct { real r, i; } complex;
typedef struct { doublereal r, i; } doublecomplex;
typedef int logical;
typedef short int shortlogical;
typedef char logical1;
typedef char integer1;

#define TRUE_ (1)
#define FALSE_ (0)

/* Extern is for use with -E */
#ifndef Extern
#define Extern extern
#endif

/* I/O stuff */

#ifdef f2c_i2
/* for -i2 */
typedef short flag;
typedef short ftnlen;
typedef short ftnint;
#else
typedef int flag;
typedef int ftnlen;
typedef int ftnint;
#endif

/*external read, write*/
typedef struct
{	flag cierr;
	ftnint ciunit;
	flag ciend;
	char *cifmt;
	ftnint cirec;
} cilist;

/*internal read, write*/
typedef struct
{	flag icierr;
	char *iciunit;
	flag iciend;
	char *icifmt;
	ftnint icirlen;
	ftnint icirnum;
} icilist;

/*open*/
typedef struct
{	flag oerr;
	ftnint ounit;
	char *ofnm;
	ftnlen ofnmlen;
	char *osta;
	char *oacc;
	char *ofm;
	ftnint orl;
	char *oblnk;
} olist;

/*close*/
typedef struct
{	flag cerr;
	ftnint cunit;
	char *csta;
} cllist;

/*rewind, backspace, endfile*/
typedef struct
{	flag aerr;
	ftnint aunit;
} alist;

/* inquire */
typedef struct
{	flag inerr;
	ftnint inunit;
	char *infile;
	ftnlen infilen;
	ftnint	*inex;	/*parameters in standard's order*/
	ftnint	*inopen;
	ftnint	*innum;
	ftnint	*innamed;
	char	*inname;
	ftnlen	innamlen;
	char	*inacc;
	ftnlen	inacclen;
	char	*inseq;
	ftnlen	inseqlen;
	char 	*indir;
	ftnlen	indirlen;
	char	*infmt;
	ftnlen	infmtlen;
	char	*inform;
	ftnint	informlen;
	char	*inunf;
	ftnlen	inunflen;
	ftnint	*inrecl;
	ftnint	*innrec;
	char	*inblank;
	ftnlen	inblanklen;
} inlist;

#define VOID void

union Multitype {	/* for multiple entry points */
	shortint h;
	integer i;
	real r;
	doublereal d;
	complex c;
	doublecomplex z;
	};

typedef union Multitype Multitype;

typedef long Long;	/* No longer used; formerly in Namelist */

struct Vardesc {	/* for Namelist */
	char *name;
	char *addr;
	ftnlen *dims;
	int  type;
	};
typedef struct Vardesc Vardesc;

struct Namelist {
	char *name;
	Vardesc **vars;
	int nvars;
	};
typedef struct Namelist Namelist;

#ifndef abs
#define abs(x) ((x) >= 0 ? (x) : -(x))
#endif
#define dabs(x) (doublereal)abs(x)
#ifndef min
#define min(a,b) ((a) <= (b) ? (a) : (b))
#endif
#ifndef max
#define max(a,b) ((a) >= (b) ? (a) : (b))
#endif
#define dmin(a,b) (doublereal)min(a,b)
#define dmax(a,b) (doublereal)max(a,b)

/* procedure parameter types for -A and -C++ */

#define F2C_proc_par_types 1
#ifdef __cplusplus
typedef int /* Unknown procedure type */ (*U_fp)(...);
typedef shortint (*J_fp)(...);
typedef integer (*I_fp)(...);
typedef real (*R_fp)(...);
typedef doublereal (*D_fp)(...), (*E_fp)(...);
typedef /* Complex */ VOID (*C_fp)(...);
typedef /* Double Complex */ VOID (*Z_fp)(...);
typedef logical (*L_fp)(...);
typedef shortlogical (*K_fp)(...);
typedef /* Character */ VOID (*H_fp)(...);
typedef /* Subroutine */ int (*S_fp)(...);
#else
typedef int /* Unknown procedure type */ (*U_fp)(void);
typedef shortint (*J_fp)(void);
typedef integer (*I_fp)(void);
typedef real (*R_fp)(void);
typedef doublereal (*D_fp)(void), (*E_fp)(void);
typedef /* Complex */ VOID (*C_fp)(void);
typedef /* Double Complex */ VOID (*Z_fp)(void);
typedef logical (*L_fp)(void);
typedef shortlogical (*K_fp)(void);
typedef /* Character */ VOID (*H_fp)(void);
typedef /* Subroutine */ int (*S_fp)(void);
#endif
/* E_fp is for real functions when -R is not specified */
typedef VOID C_f;	/* complex function */
typedef VOID H_f;	/* character function */
typedef VOID Z_f;	/* double complex function */
typedef doublereal E_f;	/* real function with -R not specified */

/* undef any lower-case symbols that your C compiler predefines, e.g.: */

#ifndef Skip_f2c_Undefs
#undef cray
#undef gcos
#undef mc68010
#undef mc68020
#undef mips
#undef pdp11
#undef sgi
#undef sparc
#undef sun
#undef sun2
#undef sun3
#undef sun4
#undef u370
#undef u3b
#undef u3b2
#undef u3b5
#undef unix
#undef vax
#endif

/*  https://anonscm.debian.org/cgit/collab-maint/libf2c2.git/tree/f2ch.add  */

/* If you are using a C++ compiler, append the following to f2c.h
   for compiling libF77 and libI77. */

#ifdef __cplusplus
extern "C" {
#endif

extern int abort_(void);
extern double c_abs(complex *);
extern void c_cos(complex *, complex *);
extern void c_div(complex *, complex *, complex *);
extern void c_exp(complex *, complex *);
extern void c_log(complex *, complex *);
extern void c_sin(complex *, complex *);
extern void c_sqrt(complex *, complex *);
extern double d_abs(double *);
extern double d_acos(double *);
extern double d_asin(double *);
extern double d_atan(double *);
extern double d_atn2(double *, double *);
extern void d_cnjg(doublecomplex *, doublecomplex *);
extern double d_cos(double *);
extern double d_cosh(double *);
extern double d_dim(double *, double *);
extern double d_exp(double *);
extern double d_imag(doublecomplex *);
extern double d_int(double *);
extern double d_lg10(double *);
extern double d_log(double *);
extern double d_mod(double *, double *);
extern double d_nint(double *);
extern double d_prod(float *, float *);
extern double d_sign(double *, double *);
extern double d_sin(double *);
extern double d_sinh(double *);
extern double d_sqrt(double *);
extern double d_tan(double *);
extern double d_tanh(double *);
extern double derf_(double *);
extern double derfc_(double *);
extern void do_fio(ftnint *, char *, ftnlen);
extern integer do_lio(ftnint *, ftnint *, char *, ftnlen);
extern integer do_uio(ftnint *, char *, ftnlen);
extern integer e_rdfe(void);
extern integer e_rdue(void);
extern integer e_rsfe(void);
extern integer e_rsfi(void);
extern integer e_rsle(void);
extern integer e_rsli(void);
extern integer e_rsue(void);
extern integer e_wdfe(void);
extern integer e_wdue(void);
extern void e_wsfe(void);
extern integer e_wsfi(void);
extern integer e_wsle(void);
extern integer e_wsli(void);
extern integer e_wsue(void);
extern int ef1asc_(ftnint *, ftnlen *, ftnint *, ftnlen *);
extern integer ef1cmc_(ftnint *, ftnlen *, ftnint *, ftnlen *);

extern double erf_(float *);
extern double erfc_(float *);
extern integer f_back(alist *);
extern integer f_clos(cllist *);
extern integer f_end(alist *);
extern void f_exit(void);
extern integer f_inqu(inlist *);
extern integer f_open(olist *);
extern integer f_rew(alist *);
extern int flush_(void);
extern void getarg_(integer *, char *, ftnlen);
extern void getenv_(char *, char *, ftnlen, ftnlen);
extern short h_abs(short *);
extern short h_dim(short *, short *);
extern short h_dnnt(double *);
extern short h_indx(char *, char *, ftnlen, ftnlen);
extern short h_len(char *, ftnlen);
extern short h_mod(short *, short *);
extern short h_nint(float *);
extern short h_sign(short *, short *);
extern short hl_ge(char *, char *, ftnlen, ftnlen);
extern short hl_gt(char *, char *, ftnlen, ftnlen);
extern short hl_le(char *, char *, ftnlen, ftnlen);
extern short hl_lt(char *, char *, ftnlen, ftnlen);
extern integer i_abs(integer *);
extern integer i_dim(integer *, integer *);
extern integer i_dnnt(double *);
extern integer i_indx(char *, char *, ftnlen, ftnlen);
extern integer i_len(char *, ftnlen);
extern integer i_mod(integer *, integer *);
extern integer i_nint(float *);
extern integer i_sign(integer *, integer *);
extern integer iargc_(void);
extern ftnlen l_ge(char *, char *, ftnlen, ftnlen);
extern ftnlen l_gt(char *, char *, ftnlen, ftnlen);
extern ftnlen l_le(char *, char *, ftnlen, ftnlen);
extern ftnlen l_lt(char *, char *, ftnlen, ftnlen);
extern void pow_ci(complex *, complex *, integer *);
extern double pow_dd(double *, double *);
extern double pow_di(double *, integer *);
extern short pow_hh(short *, shortint *);
extern integer pow_ii(integer *, integer *);
extern double pow_ri(float *, integer *);
extern void pow_zi(doublecomplex *, doublecomplex *, integer *);
extern void pow_zz(doublecomplex *, doublecomplex *, doublecomplex *);
extern double r_abs(float *);
extern double r_acos(float *);
extern double r_asin(float *);
extern double r_atan(float *);
extern double r_atn2(float *, float *);
extern void r_cnjg(complex *, complex *);
extern double r_cos(float *);
extern double r_cosh(float *);
extern double r_dim(float *, float *);
extern double r_exp(float *);
extern float r_imag(complex *);
extern double r_int(float *);
extern float r_lg10(real *);
extern double r_log(float *);
extern double r_mod(float *, float *);
extern double r_nint(float *);
extern double r_sign(float *, float *);
extern double r_sin(float *);
extern double r_sinh(float *);
extern double r_sqrt(float *);
extern double r_tan(float *);
extern double r_tanh(float *);
extern void s_cat(char *, char **, integer *, integer *, ftnlen);
extern integer s_cmp(char *, char *, ftnlen, ftnlen);
extern void s_copy(char *, char *, ftnlen, ftnlen);
extern int s_paus(char *, ftnlen);
extern integer s_rdfe(cilist *);
extern integer s_rdue(cilist *);
extern integer s_rnge(char *, integer, char *, integer);
extern integer s_rsfe(cilist *);
extern integer s_rsfi(icilist *);
extern integer s_rsle(cilist *);
extern integer s_rsli(icilist *);
extern integer s_rsne(cilist *);
extern integer s_rsni(icilist *);
extern integer s_rsue(cilist *);
extern int s_stop(char *, ftnlen);
extern integer s_wdfe(cilist *);
extern integer s_wdue(cilist *);
extern void s_wsfe(	cilist *);
extern integer s_wsfi(icilist *);
extern integer s_wsle(cilist *);
extern integer s_wsli(icilist *);
extern integer s_wsne(cilist *);
extern integer s_wsni(icilist *);
extern integer s_wsue(cilist *);
extern void sig_die(char *, int);
extern integer signal_(integer *, void (*)(int));
extern integer system_(char *, ftnlen);
extern double z_abs(doublecomplex *);
extern void z_cos(doublecomplex *, doublecomplex *);
extern void z_div(doublecomplex *, doublecomplex *, doublecomplex *);
extern void z_exp(doublecomplex *, doublecomplex *);
extern void z_log(doublecomplex *, doublecomplex *);
extern void z_sin(doublecomplex *, doublecomplex *);
extern void z_sqrt(doublecomplex *, doublecomplex *);

#ifdef __cplusplus
	}
#endif

#endif
