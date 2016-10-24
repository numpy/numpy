#include <stddef.h>       /* for NULL */
#include <limits.h>       /* for ULONG_MAX */
#include <assert.h>
#include <math.h>         /* fmod, fabs */
#include <cmath>          /* expm1 */
#include <algorithm>      /* std::sort */

#include "mkl.h"
#include "mkl_vml.h"
#include "mkl_distributions.h"

void
vrk_double_vec(vrk_state *state, const int len, double *res)
{
    int err;
    const double d_zero = 0.0, d_one = 1.0;

    if(len < 1)
        return;

    err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

}


void
vrk_uniform_vec(vrk_state *state, const int len, double *res, const double low, const double high)
{
    int err;

    if(len < 1)
        return;

    err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, len, res, low, high);
    assert(err == VSL_STATUS_OK);

}


void
vrk_standard_normal_vec_ICDF(vrk_state *state, const int len, double *res)
{
    int err;
    const double d_zero = 0.0, d_one = 1.0;

    if(len < 1)
        return;

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

}

void
vrk_normal_vec_ICDF(vrk_state *state, const int len, double *res, const double loc, const double scale)
{
    int err;

    if(len < 1)
        return;

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->stream, len, res, loc, scale);
    assert(err == VSL_STATUS_OK);

}


void
vrk_standard_normal_vec_BM1(vrk_state *state, const int len, double *res)
{
    int err;
    const double d_zero = 0.0, d_one = 1.0;

    if(len < 1)
        return;

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

}

void
vrk_normal_vec_BM1(vrk_state *state, const int len, double *res, const double loc, const double scale)
{
    int err;

    if(len < 1)
        return;

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, state->stream, len, res, loc, scale);
    assert(err == VSL_STATUS_OK);

}


void
vrk_standard_normal_vec_BM2(vrk_state *state, const int len, double *res)
{
    int err;
    const double d_zero = 0.0, d_one = 1.0;

    if(len < 1)
        return;

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

}

void
vrk_normal_vec_BM2(vrk_state *state, const int len, double *res, const double loc, const double scale)
{
    int err;

    if(len < 1)
        return;

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, state->stream, len, res, loc, scale);
    assert(err == VSL_STATUS_OK);

}


void
vrk_standard_exponential_vec(vrk_state *state, const int len, double *res)
{
    int err;
    const double d_zero = 0.0, d_one = 1.0;

    if(len < 1)
        return;

    err = vdRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

}

void
vrk_exponential_vec(vrk_state *state, const int len, double *res, const double scale)
{
    int err;
    const double d_zero = 0.0;

    if(len < 1)
        return;

    err = vdRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE, state->stream, len, res, d_zero, scale);
    assert(err == VSL_STATUS_OK);

}


void
vrk_standard_cauchy_vec(vrk_state *state, const int len, double *res)
{
    int err;
    const double d_zero = 0.0, d_one = 1.0;

    if(len < 1)
        return;

    err = vdRngCauchy(VSL_RNG_METHOD_CAUCHY_ICDF, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

}

void
vrk_standard_gamma_vec(vrk_state *state, const int len, double *res, const double shape)
{
    int err;
    const double d_zero = 0.0, d_one = 1.0;

    if(len < 1)
        return;

    err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, len, res, shape, d_zero, d_one);

    assert(err == VSL_STATUS_OK);
}

void
vrk_gamma_vec(vrk_state *state, const int len, double *res, const double shape, const double scale)
{
    int err;
    const double d_zero = 0.0;

    if(len < 1)
        return;

    err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, len, res, shape, d_zero, scale);

    assert(err == VSL_STATUS_OK);

}


/*  X ~ Z * (G*(2/df))**-0.5 */
void
vrk_standard_t_vec(vrk_state *state, const int len, double *res, const double df)
{
    int err;
    const double d_zero = 0.0, d_one = 1.0;
    double shape = df/2;
    double *sn = NULL;

    if(len < 1)
        return;

    err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, len, res, shape, d_zero, 1.0/shape);
    assert(err == VSL_STATUS_OK);

    vmdInvSqrt(len, res, res, VML_HA);

    sn = (double *) mkl_malloc(len*sizeof(double), 64);
    assert(sn != NULL);

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->stream, len, sn, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

    vmdMul(len, res, sn, res, VML_HA);
    mkl_free(sn);

}

/* chisquare(df) ~ G(df/2, 2) */
void
vrk_chisquare_vec(vrk_state *state, const int len, double *res, const double df)
{
    int err;
    const double d_zero = 0.0, d_two = 2.0;
    double shape = 0.5*df;

    if(len < 1)
        return;

    err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, len, res, shape, d_zero, d_two);
    assert(err == VSL_STATUS_OK);

}

/*    P ~ U^(-1/a) - 1 =  */
void
vrk_pareto_vec(vrk_state *state, const int len, double *res, const double alp)
{
    int i, err;
    const double d_zero = 0.0, d_one = 1.0;
    double neg_rec_alp = -1.0/alp;

    if (len<1)
        return;

    err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

    /* res[i] = pow(res[i], neg_rec_alp) */
    vmdPowx(len, res, neg_rec_alp, res, VML_HA);

    #pragma ivdep
    for(i=0; i < len; i++) res[i] -= 1.0;

}

/*  W ~ E^(1/alp) */
void
vrk_weibull_vec(vrk_state *state, const int len, double *res, const double alp)
{
    int i, err;
    const double d_zero = 0.0, d_one = 1.0;
    double rec_alp = 1.0/alp;

    if (len<1)
        return;

    err = vdRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

    vmdPowx(len, res, rec_alp, res, VML_HA);

}

/*  pow(1 - exp(-E(1))), 1./a) == pow(U, 1./a) */
void
vrk_power_vec(vrk_state *state, const int len, double *res, const double alp)
{
    int i, err;
    const double d_zero = 0.0, d_one = 1.0;
    double rec_alp = 1.0/alp;

    if (len<1)
        return;

    err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

    /* res[i] = pow(res[i], rec_alp) */
    vmdPowx(len, res, rec_alp, res, VML_HA);

}

/*  scale * sqrt(2.0 * E(1))  */
void
vrk_rayleigh_vec(vrk_state *state, const int len, double *res, const double scale)
{
    int i, err;
    const double d_zero = 0.0, d_two = 2.0;

    if (len<1)
        return;

    err = vdRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE, state->stream, len, res, d_zero, d_two);
    assert(err == VSL_STATUS_OK);

    vmdSqrt(len, res, res, VML_HA);

    #pragma ivdep
    for(i=0; i < len; i++) res[i] *= scale;

}

void
vrk_beta_vec(vrk_state *state, const int len, double *res, const double a, const double b)
{
    int err;
    const double d_zero = 0.0, d_one = 1.0;

    if(len < 1)
        return;

    err = vdRngBeta(VSL_RNG_METHOD_BETA_CJA_ACCURATE, state->stream, len, res, a, b, d_zero, d_one);

    assert(err == VSL_STATUS_OK);
}

/*  F(df_num, df_den) ~ G( df_num/2, 2/df_num) / G(df_den/2, 2/df_den))  */
void
vrk_f_vec(vrk_state *state, const int len, double *res, const double df_num, const double df_den)
{
    int err;
    const double d_zero = 0.0, d_one = 1.0;
    double shape = 0.5*df_num, scale = 2.0/df_num;
    double *den = NULL;

    if(len < 1)
        return;

    err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, len, res, shape, d_zero, scale);
    assert(err == VSL_STATUS_OK);

    den = (double *) mkl_malloc(len*sizeof(double), 64);
    assert(den != NULL);

    shape = 0.5*df_den;
    scale = 2.0/df_den;
    err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, len, den, shape, d_zero, scale);
    assert(err == VSL_STATUS_OK);

    vmdDiv(len, res, den, res, VML_HA);
    mkl_free(den);

}

/*
   for df > 1, X ~ Chi2(df - 1) + ( sqrt(nonc) + Z)^2
   for df <=1, X ~ Chi2( df + 2*I), where I ~ Poisson( nonc/2.0)
*/
void
vrk_noncentral_chisquare_vec(vrk_state *state, const int len, double *res, const double df, const double nonc)
{
    int i, err;
    const double d_zero = 0.0, d_one = 1.0, d_two = 2.0;
    double shape, loc;

    if (len < 1)
        return;

    if (df > 1) {
        double *nvec;

        shape = 0.5*(df - 1.0);
        /* res has chi^2 with (df - 1) */
        err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, len, res, shape, d_zero, d_two);

        nvec = (double *) mkl_malloc(len*sizeof(double), 64);
        assert(nvec != NULL);

        loc = sqrt(nonc);
        err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->stream, len, nvec, loc, d_one);
        assert(err == VSL_STATUS_OK);

        /* squaring could result in an overflow */
        vmdSqr(len, nvec, nvec, VML_HA);
        vmdAdd(len, res, nvec, res, VML_HA);

        mkl_free(nvec);

    } else {
        if (df == 0.) {
            return vrk_chisquare_vec(state, len, res, df);
        }
        if(df < 1) {
            /* noncentral_chisquare(df, nonc) ~ G( df/2 + Poisson(nonc/2), 2) */
            double lambda;
            int *pvec = (int *) mkl_malloc(len*sizeof(int), 64);

            assert(pvec != NULL);

            lambda = 0.5*nonc;
            err = viRngPoisson(VSL_RNG_METHOD_POISSON_PTPE, state->stream, len, pvec, lambda);
            assert(err == VSL_STATUS_OK);

            shape = 0.5*df;

            if(0.125 * len > sqrt(lambda))  {
                int *idx = NULL;
                double *tmp = NULL;

                idx = (int *) mkl_malloc(len * sizeof(int), 64);
                assert( idx != NULL );

                #pragma ivdep
                for(i=0; i <len; i++) idx[i] = i;

                std::sort(idx, idx + len, [pvec](int i1, int i2){ return pvec[i1] < pvec[i2]; } );
                /* idx now contains original indexes of ordered Poisson outputs */

                /* allocate workspace to store samples of gamma, enough to hold entire output */
                tmp = (double *) mkl_malloc( len * sizeof(double), 64);
                assert( tmp != NULL );

                for(i = 0; i < len; ) {
                    int k, j, cv = pvec[idx[i]];

                    #pragma ivdep
                    for(j=i+1; (j < len) && (pvec[idx[j]] == cv); j++) {}

                    assert(j > i);
                    err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, j - i, tmp,
                                shape + cv, d_zero, d_two);
                    assert(err == VSL_STATUS_OK);

                    #pragma ivdep
                    for(k = i; k < j; k++) res[idx[k]] = tmp[k - i];

                    i = j;
                }

                mkl_free(tmp);
                mkl_free(idx);

            } else {

                for(i=0; i<len; i++) {
                    err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, 1,
                        res + i, shape + pvec[i], d_zero, d_two);
                    assert(err == VSL_STATUS_OK);
                }
            }

            mkl_free(pvec);
        } else {
            float *fuvec = NULL;

            /* noncentral_chisquare(1, nonc) ~ sqrt(nonc)*(-1)^[U<0.5] + Z */
            err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->stream, len, res, d_zero, d_one);
            loc = sqrt(nonc);

            fuvec = (float *) mkl_malloc(len*sizeof(float), 64);
            assert(fuvec != NULL);

            err = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, fuvec, (const float) d_zero, (const float) d_one);
            assert(err == VSL_STATUS_OK);

            #pragma ivdep
            for(i=0; i<len; i++) res[i] += (fuvec[i] < 0.5) ? -loc : loc;

            mkl_free(fuvec);
        }
    }

}

void
vrk_laplace_vec(vrk_state *state, const int len, double *res, const double loc, const double scale)
{
    int err;

    if(len < 1)
        return;

    err = vdRngLaplace(VSL_RNG_METHOD_LAPLACE_ICDF, state->stream, len, res, loc, scale);
    assert(err == VSL_STATUS_OK);

}


void
vrk_gumbel_vec(vrk_state *state, const int len, double *res, const double loc, const double scale)
{
    int err;

    if(len < 1)
        return;

    err = vdRngGumbel(VSL_RNG_METHOD_GUMBEL_ICDF, state->stream, len, res, loc, scale);
    assert(err == VSL_STATUS_OK);

}

/*   Logistic(loc, scale) ~ loc + scale * log(u/(1.0 - u)) */
void
vrk_logistic_vec(vrk_state *state, const int len, double *res, const double loc, const double scale)
{
    int i, err;
    const double d_one = 1.0, d_mone = -1.0, d_zero = 0.0;

    if(len < 1)
        return;

    err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

    /* can MKL optimize computation of the logit function  p \mapsto \ln(p/(1-p)) */
    #pragma ivdep
    for(i=0; i<len; i++) res[i] = log(res[i]/(1.0 - res[i]));

    #pragma ivdep
    for(i=0; i<len; i++) res[i] = loc + scale*res[i];
}

void
vrk_lognormal_vec_ICDF(vrk_state *state, const int len, double *res, const double mean, const double sigma)
{
    int err;
    const double d_zero = 0.0, d_one = 1.0;

    if(len < 1)
        return;

    err = vdRngLognormal(VSL_RNG_METHOD_LOGNORMAL_ICDF_ACCURATE, state->stream, len, res, mean, sigma, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

}

void
vrk_lognormal_vec_BM(vrk_state *state, const int len, double *res, const double mean, const double sigma)
{
    int err;
    const double d_zero = 0.0, d_one = 1.0;

    if(len < 1)
        return;

    err = vdRngLognormal(VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2_ACCURATE, state->stream, len, res, mean, sigma, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

}

/* direct transformation method */
void
vrk_wald_vec(vrk_state *state, const int len, double *res, const double mean, const double scale)
{
    int i, err;
    const double d_zero = 0., d_one = 1.0;
    double *uvec = NULL;
    double gsc = 0.5*sqrt(mean / scale);

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->stream, len, res, d_zero, gsc);
    assert(err == VSL_STATUS_OK);

    /* Y = mean/(4 scale) * Z^2 */
    vmdSqr(len, res, res, VML_HA);

    #pragma ivdep
    for(i = 0; i < len; i++) {
        if(res[i] <= 1.0) {
            res[i] = 1.0 + res[i] - sqrt( res[i] * (res[i] + 2.0));
        } else {
            res[i] = 1.0 - 2.0/(1.0 + sqrt( 1 + 2.0/res[i]));
        }
    }

    uvec = (double *) mkl_malloc(len*sizeof(double), 64);
    assert(uvec != NULL);

    err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, len, uvec, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

    #pragma ivdep
    for(i=0; i<len; i++) {
        if (uvec[i]*(1.0 + res[i]) <= 1.0)
            res[i] = mean*res[i];
        else
            res[i] = mean/res[i];
    }

    mkl_free(uvec);

}

#ifndef M_PI
/*  128-bits worth of pi */
#define M_PI 3.141592653589793238462643383279502884197
#endif


/* Uses the rejection algorithm compared against the wrapped Cauchy
   distribution suggested by Best and Fisher and documented in
   Chapter 9 of Luc's Non-Uniform Random Variate Generation.
   http://cg.scs.carleton.ca/~luc/rnbookindex.html
   (but corrected to match the algorithm in R and Python)
*/
static void
vrk_vonmises_vec_small_kappa(vrk_state *state, const int len, double *res, const double mu, const double kappa)
{
    int i, err, n, size;
    double rho_over_kappa, rho, r, s_kappa, Z, W, Y, V;
    double *Uvec = NULL, *Vvec = NULL;
    float *VFvec = NULL;
    const double d_zero = 0.0, d_one = 1.0;

    assert(0. < kappa <= 1.0);

    r = 1 + sqrt(1 + 4*kappa*kappa);
    rho_over_kappa = (2) / (r + sqrt(2*r));
    rho = rho_over_kappa * kappa;

    /* s times kappa */
    s_kappa = (1 + rho*rho)/(2*rho_over_kappa);

    Uvec = (double *) mkl_malloc(len*sizeof(double), 64);
    assert(Uvec != NULL);
    Vvec = (double *) mkl_malloc(len*sizeof(double), 64);
    assert(Vvec != NULL);

    for(n = 0; n < len; )
    {
        size = len - n;
        err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, size, Uvec, d_zero, M_PI);
        assert(err == VSL_STATUS_OK);
        err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, size, Vvec, d_zero, d_one);
        assert(err == VSL_STATUS_OK);

        for(i = 0; i < size; i++ ) {
            Z = cos(Uvec[i]);  V = Vvec[i];
            W = (kappa + s_kappa * Z) / (s_kappa + kappa * Z);
            Y = s_kappa - kappa * W;
            if ((Y*(2 - Y) >= V) || (log(Y/V) + 1 >= Y)) {
                res[n++] = acos(W);
            }
        }
    }

    mkl_free(Uvec);

    VFvec = (float *) Vvec;
    err = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, VFvec, (float) d_zero, (float) d_one);
    assert(err == VSL_STATUS_OK);

    #pragma ivdep
    for(i = 0; i < len; i++) {
        double mod, resi;

        resi = (VFvec[i] < 0.5) ? mu - res[i] : mu + res[i];
        mod = fabs(resi);
        mod = (fmod(mod + M_PI, 2*M_PI) - M_PI);
        res[i] = (resi < 0) ? -mod : mod;
    }

    mkl_free(Vvec);

}

static void
vrk_vonmises_vec_large_kappa(vrk_state *state, const int len, double *res, const double mu, const double kappa)
{
    int i, err, n, size;
    double r_over_two_kappa, recip_two_kappa;
    double s_minus_one, hpt, r_over_two_kappa_minus_one, rho_minus_one, neg_W_minus_one;
    double *Uvec = NULL, *Vvec = NULL;
    float *VFvec = NULL;
    const double d_zero = 0.0, d_one = 1.0;

    assert(kappa > 1.0);

    recip_two_kappa = 1 / (2 * kappa);

    /* variables here are dwindling to zero as kappa grows */
    hpt = sqrt(1 + recip_two_kappa * recip_two_kappa);
    r_over_two_kappa_minus_one = recip_two_kappa * (1 + recip_two_kappa / (1 + hpt));
    r_over_two_kappa = 1 + r_over_two_kappa_minus_one;
    rho_minus_one = r_over_two_kappa_minus_one - sqrt(2 * r_over_two_kappa * recip_two_kappa);
    s_minus_one = rho_minus_one*(0.5 * rho_minus_one/(1 + rho_minus_one));

    Uvec = (double *) mkl_malloc(len * sizeof(double), 64);
    assert(Uvec != NULL);
    Vvec = (double *) mkl_malloc(len * sizeof(double), 64);
    assert(Vvec != NULL);

    for(n = 0; n < len; )
    {
        size = len - n;
        err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, size, Uvec, d_zero, 0.5*M_PI);
        assert(err == VSL_STATUS_OK);
        err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, size, Vvec, d_zero, d_one);
        assert(err == VSL_STATUS_OK);

        #pragma ivdep
        for(i = 0; i < size; i++ ) {
	    double sn, cn, sn2, cn2;
	    double neg_W_minus_one, V, Y;

	    sn = sin(Uvec[i]);  cn = cos(Uvec[i]); V = Vvec[i];
	    sn2 = sn*sn;  cn2 = cn*cn;

            neg_W_minus_one = s_minus_one * sn2 / (0.5*s_minus_one + cn2);
            Y = kappa * (s_minus_one + neg_W_minus_one);

            if ((Y*(2 - Y) >= V) || (log(Y/V) + 1 >= Y)) {
	        Y = neg_W_minus_one * (2 - neg_W_minus_one);
		if (Y < 0)
		    Y = 0.;
		else
		    if (Y > 1.0)
		        Y = 1.0;

	        res[n++] = asin(sqrt(Y));
            }
        }
    }

    mkl_free(Uvec);

    VFvec = (float *) Vvec;
    err = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, VFvec, (float) d_zero, (float) d_one);
    assert(err == VSL_STATUS_OK);

    #pragma ivdep
    for(i = 0; i < len; i++) {
        double mod, resi;

        resi = (VFvec[i] < 0.5) ? mu - res[i] : mu + res[i];
        mod = fabs(resi);
        mod = (fmod(mod + M_PI, 2*M_PI) - M_PI);
        res[i] = (resi < 0) ? -mod : mod;
    }

    mkl_free(Vvec);
}

void
vrk_vonmises_vec(vrk_state *state, const int len, double *res, const double mu, const double kappa)
{
    if(len < 1)
        return;

    if(kappa > 1.0)
        vrk_vonmises_vec_large_kappa(state, len, res, mu, kappa);
    else
        vrk_vonmises_vec_small_kappa(state, len, res, mu, kappa);

}

void
vrk_noncentral_f_vec(vrk_state *state, const int len, double *res, const double df_num, const double df_den, const double nonc)
{
    int i;
    double *den = NULL, fctr;

    if(len < 1)
        return;

    if(nonc == 0.)
        return vrk_f_vec(state, len, res, df_num, df_den);

    vrk_noncentral_chisquare_vec(state, len, res, df_num, nonc);

    den = (double *) mkl_malloc(len*sizeof(double), 64);

    if(den == NULL)
        return;

    vrk_noncentral_chisquare_vec(state, len, den, df_den, nonc);

    vmdDiv(len, res, den, res, VML_HA);

    mkl_free(den);
    fctr = df_den/df_num;

    #pragma ivdep
    for(i = 0; i < len; i++) res[i] *= fctr;

}


void
vrk_triangular_vec(vrk_state *state, const int len, double *res, const double x_min, const double x_mode, const double x_max)
{
    int i, err;
    const double d_zero = 0.0, d_one = 1.0;
    double ratio, lpr, rpr;

    if (len < 1)
        return;

    err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

    {
        double wtot, wl, wr;

        wtot = x_max - x_min;
        wl = x_mode - x_min;
        wr = x_max - x_mode;

        ratio = wl / wtot;
        lpr = wl * wtot;
        rpr = wr * wtot;
    }

    assert( 0 <= ratio && ratio <= 1);

    if (ratio <= 0) {
        #pragma ivdep
        for(i = 0; i < len; i++) {
            /* U and 1 - U are equal in distribution */
            res[i] = x_max - sqrt(res[i] * rpr);
        }
    } else if (ratio >= 1) {
        #pragma ivdep
        for(i = 0; i < len; i++) {
            res[i] = x_min + sqrt(res[i]*lpr);
        }
    } else {
        #pragma ivdep
        for(i = 0; i < len; i++) {
            double ui = res[i];
            res[i] = (ui > ratio) ? x_max - sqrt((1.0 - ui) * rpr) : x_min + sqrt(ui*lpr);
        }
    }
}

void
vrk_binomial_vec(vrk_state *state, const int len, int *res, const int n, const double p)
{
    int err;

    if(len < 1)
        return;

    if (n==0) {
        int i;

        #pragma ivdep
        for(i=0; i < len; i++) res[i] = 0;
    }
    else {
        err = viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, state->stream, len, res, n, p);
        assert(err == VSL_STATUS_OK);
    }
}

void
vrk_geometric_vec(vrk_state *state, const int len, int *res, const double p)
{
    int err;

    if(len < 1)
        return;

    err = viRngGeometric(VSL_RNG_METHOD_GEOMETRIC_ICDF, state->stream, len, res, p);
    assert(err == VSL_STATUS_OK);

}

void
vrk_negbinomial_vec(vrk_state *state, const int len, int *res, const double a, const double p)
{
    int err;

    if(len < 1)
        return;

    err = viRngNegbinomial(VSL_RNG_METHOD_NEGBINOMIAL_NBAR, state->stream, len, res, a, p);
    assert(err == VSL_STATUS_OK);

}

void
vrk_hypergeometric_vec(vrk_state *state, const int len, int *res, const int lot_s,
        const int sampling_s, const int marked_s)
{
    int err;

    if(len < 1)
        return;

    err = viRngHypergeometric(VSL_RNG_METHOD_HYPERGEOMETRIC_H2PE, state->stream, len, res,
                lot_s, sampling_s, marked_s);
    assert(err == VSL_STATUS_OK);

}

void
vrk_poisson_vec_PTPE(vrk_state *state, const int len, int *res, const double lambda)
{
    int err;

    if(len < 1)
        return;

    err = viRngPoisson(VSL_RNG_METHOD_POISSON_PTPE, state->stream, len, res, lambda);
    assert(err == VSL_STATUS_OK);

}

void
vrk_poisson_vec_POISNORM(vrk_state *state, const int len, int *res, const double lambda)
{
    int err;

    if(len < 1)
        return;

    err = viRngPoisson(VSL_RNG_METHOD_POISSON_POISNORM, state->stream, len, res, lambda);
    assert(err == VSL_STATUS_OK);

}

void
vrk_poisson_vec_V(vrk_state *state, const int len, int *res, double *lambdas)
{
    int err;

    if(len < 1)
        return;

    err = viRngPoissonV(VSL_RNG_METHOD_POISSONV_POISNORM, state->stream, len, res, lambdas);
    assert(err == VSL_STATUS_OK);

}


void
vrk_zipf_long_vec(vrk_state *state, const int len, long *res, const double a)
{
    int i, err, n_accepted, batch_size;
    double T, U, V, am1, b;
    double *Uvec = NULL, *Vvec = NULL;
    long X;
    const double d_zero = 0.0, d_one = 1.0;

    if(len < 1)
        return;

    am1 = a - d_one;
    b = pow(2.0, am1);

    Uvec = (double *) mkl_malloc(len * sizeof(double), 64);
    assert(Uvec != NULL);
    Vvec = (double *) mkl_malloc(len * sizeof(double), 64);
    assert(Vvec != NULL);

    for(n_accepted=0; n_accepted < len; ) {
        batch_size = len - n_accepted;
        err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, batch_size, Uvec, d_zero, d_one);
        assert(err == VSL_STATUS_OK);
        err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, batch_size, Vvec, d_zero, d_one);
        assert(err == VSL_STATUS_OK);

        #pragma ivdep
        for(i = 0; i < batch_size; i++) {
            U = d_one - Uvec[i]; V = Vvec[i];
            X = (long)floor(pow(U, (-1.0)/am1));
            /* The real result may be above what can be represented in a signed
             * long. It will get casted to -sys.maxint-1. Since this is
             * a straightforward rejection algorithm, we can just reject this value
             * in the rejection condition below. This function then models a Zipf
             * distribution truncated to sys.maxint.
             */
            T = pow(d_one + d_one/X, am1);
            if ( (X > 0) && ( (V * X) * (T - d_one)/(b - d_one) <= T/b) ) {
                res[n_accepted++] = X;
            }
        }
    }

    mkl_free(Vvec);
    mkl_free(Uvec);

}

void
vrk_logseries_vec(vrk_state *state, const int len, int *res, const double theta)
{
    int i, err, n_accepted, batch_size;
    double q, r, V;
    double *Uvec = NULL, *Vvec = NULL;
    int result;
    const double d_zero = 0.0, d_one = 1.0;

    if(len < 1)
        return;

    r = log(d_one - theta);

    Uvec = (double *) mkl_malloc(len * sizeof(double), 64);
    assert(Uvec != NULL);
    Vvec = (double *) mkl_malloc(len * sizeof(double), 64);
    assert(Vvec != NULL);

    for(n_accepted=0; n_accepted < len; ) {
        batch_size = len - n_accepted;
        err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, batch_size, Uvec, d_zero, d_one);
        assert(err == VSL_STATUS_OK);
        err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, batch_size, Vvec, d_zero, d_one);
        assert(err == VSL_STATUS_OK);

        #pragma ivdep
        for(i = 0; i < batch_size; i++) {
            V = Vvec[i];
            if (V >= theta) {
                res[n_accepted++] = 1;
            } else {
#if __cplusplus > 199711L
                q = -expm1(r * Uvec[i]);
#else
                /*  exp(x) - 1 == 2 * exp(x/2) * sinh(x/2)  */
                q = r * Uvec[i];
                if (q > 1.) {
                    q = 1.0 - exp(q);
                } else {
                    q = 0.5 * q;
                    q = -2.0 * exp(q) * sinh(q);
                }
#endif
                if (V <= q*q) {
                    result = (int) floor(1 + log(V)/log(q));
                    if(result > 0) {
                        res[n_accepted++] = result;
                    }
                } else {
                    res[n_accepted++] = (V < q) ? 2 : 1;
                }
            }
        }
    }

    mkl_free(Vvec);

}

/* samples discrete uniforms from [low, high) */
void
vrk_discrete_uniform_vec(vrk_state *state, const int len, int *res, const int low, const int high)
{
    int err;

    if (len  < 1)
        return;

    err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, res, low, high);
    assert(err == VSL_STATUS_OK);

}

void
vrk_discrete_uniform_long_vec(vrk_state *state, const int len, long *res, const long low, const long high)
{
    int err;
    unsigned long max;
    int i;

    if (len  < 1)
        return;


    max = ((unsigned long) high) - ((unsigned long) low) - 1UL;
    if(max == 0) {
        #pragma ivdep
        for(i=0; i < len; i++) res[i] = low;

        return;
    }

    if (max <= (unsigned long) INT_MAX) {
        int *buf = (int*) mkl_malloc( len*sizeof(int), 64);
        assert(buf != NULL);

        err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, buf, -1, (const int) max);
        assert(err == VSL_STATUS_OK);

        #pragma ivdep
        for(i=0; i < len; i++) res[i] = low + ((long) buf[i]) + 1L;

        mkl_free(buf);

    } else {
        unsigned long mask = max;
        unsigned long *buf = NULL;
        int n_accepted;

        /* Smallest bit mask >= max */
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
#if ULONG_MAX > 0xffffffffUL
        mask |= mask >> 32;
#endif

        buf = (unsigned long *) mkl_malloc( len*sizeof(long), 64);
        assert(buf != NULL);
        n_accepted = 0;

        while(n_accepted < len) {
            int k, batchSize = len - n_accepted;

            err = viRngUniformBits64(VSL_RNG_METHOD_UNIFORM_STD, state->stream, batchSize, (unsigned MKL_INT64 *) buf);
            assert(err == VSL_STATUS_OK);

            for(k=0; k < batchSize; k++) {
                unsigned long value = buf[k] & mask;
                if ( value <= max) {
                    res[n_accepted++] = low + value;
                }
            }
        }

        mkl_free(buf);
    }
}


void
vrk_ulong_vec(vrk_state *state, const int len, unsigned long *res)
{
    int err;

#if ULONG_MAX <= 0xffffffffUL
    err = viRngUniformBits32(VSL_RNG_METHOD_UNIFORMBITS32_STD, state->stream, len, (unsigned MKL_INT*) res);
#else
    err = viRngUniformBits64(VSL_RNG_METHOD_UNIFORMBITS64_STD, state->stream, len, (unsigned MKL_INT64*) res);
#endif

    assert(err == VSL_STATUS_OK);
}

void
vrk_long_vec(vrk_state *state, const int len, long *res)
{
    int i;
    unsigned long *ulptr = (unsigned long*) res;

    vrk_ulong_vec(state, len, ulptr);

    #pragma ivdep
    for(i=0; i<len; i++)
        res[i] = (long) (ulptr[i] >> 1);

}

void
vrk_rand_bool_vec(vrk_state *state, const int len, npy_bool *res, const npy_bool lo, const npy_bool hi)
{
    int err, i;
    int *buf = NULL;

    if(len < 1)
        return;

    if (lo == hi) {
        #pragma ivdep
        for(i = 0; i < len; i++) res[i] = lo;

        return;
    }

    assert( (lo == 0) && (hi == 1) );
    buf = (int *) mkl_malloc(len * sizeof(int), 64);
    assert( buf != NULL);

    err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, buf, (const int) lo, (const int) hi + 1);
    assert(err == VSL_STATUS_OK);

    #pragma ivdep
    for(i = 0; i < len; i++) res[i] = (npy_bool) buf[i];

    mkl_free(buf);
}

void
vrk_rand_uint8_vec(vrk_state *state, const int len, npy_uint8 *res, const npy_uint8 lo, const npy_uint8 hi)
{
    int err, i;
    int *buf = NULL;

    if(len < 1)
        return;

    if (lo == hi) {
        #pragma ivdep
        for(i = 0; i < len; i++) res[i] = lo;

        return;
    }

    assert( lo < hi );
    buf = (int *) mkl_malloc(len * sizeof(int), 64);
    assert( buf != NULL);

    err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, buf, (const int) lo, (const int) hi + 1);
    assert(err == VSL_STATUS_OK);

    #pragma ivdep
    for(i = 0; i < len; i++) res[i] = (npy_uint8) buf[i];

    mkl_free(buf);

}

void
vrk_rand_int8_vec(vrk_state *state, const int len, npy_int8 *res, const npy_int8 lo, const npy_int8 hi)
{
    int err, i;
    int *buf = NULL;

    if(len < 1)
        return;

    if (lo == hi) {
        #pragma ivdep
        for(i = 0; i < len; i++) res[i] = lo;

        return;
    }

    assert( lo < hi );
    buf = (int *) mkl_malloc(len * sizeof(int), 64);
    assert( buf != NULL);

    err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, buf, (const int) lo, (const int) hi + 1);
    assert(err == VSL_STATUS_OK);

    #pragma ivdep
    for(i = 0; i < len; i++) res[i] = (npy_int8) buf[i];

    mkl_free(buf);
}

void
vrk_rand_uint16_vec(vrk_state *state, const int len, npy_uint16 *res, const npy_uint16 lo, const npy_uint16 hi)
{
    int err, i;
    int *buf = NULL;

    if(len < 1)
        return;

    if (lo == hi) {
        #pragma ivdep
        for(i = 0; i < len; i++) res[i] = lo;

        return;
    }

    assert( lo < hi );
    buf = (int *) mkl_malloc(len * sizeof(int), 64);
    assert( buf != NULL);

    err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, buf, (const int) lo, (const int) hi + 1);
    assert(err == VSL_STATUS_OK);

    #pragma ivdep
    for(i = 0; i < len; i++) res[i] = (npy_uint16) buf[i];

    mkl_free(buf);
}

void
vrk_rand_int16_vec(vrk_state *state, const int len, npy_int16 *res, const npy_int16 lo, const npy_int16 hi)
{
    int err, i;
    int *buf = NULL;

    if(len < 1)
        return;

    if (lo == hi) {
        #pragma ivdep
        for(i = 0; i < len; i++) res[i] = lo;

        return;
    }

    assert( lo < hi );
    buf = (int *) mkl_malloc(len * sizeof(int), 64);
    assert( buf != NULL);

    err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, buf, (const int) lo, (const int) hi + 1);
    assert(err == VSL_STATUS_OK);

    #pragma ivdep
    for(i = 0; i < len; i++) res[i] = (npy_int16) buf[i];

    mkl_free(buf);
}

void
vrk_rand_uint32_vec(vrk_state *state, const int len, npy_uint32 *res, const npy_uint32 lo, const npy_uint32 hi)
{
    int err;
    unsigned int intm = INT_MAX;

    if (len < 1)
        return;

    /* optimization for lo = 0 and hi = 2**32-1 */
    if (!(lo || ~hi)) {
        err = viRngUniformBits32(VSL_RNG_METHOD_UNIFORMBITS32_STD, state->stream, len, (unsigned MKL_INT *) res);
        assert(err == VSL_STATUS_OK);

        return;
    }

    if (hi >= intm) {

        npy_int32 shft = ((npy_uint32) intm) + ((npy_uint32) 1);
        int i;

        /* if lo is non-zero, shift one more to accommodate possibility of hi being ULONG_MAX */
        if (lo) shft++;

        err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, (int *) res, (const int) (lo - shft), (const int) (hi - shft + 1U));
        assert(err == VSL_STATUS_OK);

        #pragma ivdep
        for(i=0; i < len; i++) res[i] += shft;

    } else {
        err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, (int *) res, (const int) lo, (const int) hi + 1);
        assert(err == VSL_STATUS_OK);
    }
}

void
vrk_rand_int32_vec(vrk_state *state, const int len, npy_int32 *res, const npy_int32 lo, const npy_int32 hi)
{
    int err;
    int intm = INT_MAX;

    if (len < 1)
        return;

    if(hi >= intm) {
        int i;

        vrk_rand_uint32_vec(state, len, (npy_uint32 *) res, 0U, (npy_uint32) (hi - lo));

        #pragma ivdep
        for(i=0; i < len; i++) res[i] += lo;

    } else {
        err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, (int *) res, (const int) lo, (const int) hi + 1);
        assert(err == VSL_STATUS_OK);
    }
}

void
vrk_rand_uint64_vec(vrk_state *state, const int len, npy_uint64 *res, const npy_uint64 lo, const npy_uint64 hi)
{
    npy_uint64 rng;
    int i, err;

    if(len < 1)
        return;

    /* optimization for lo = 0 and hi = 2**64-1 */
    if (!(lo || ~hi)) {
        err = viRngUniformBits64(VSL_RNG_METHOD_UNIFORMBITS64_STD, state->stream, len, (unsigned MKL_INT64 *) res);
        assert(err == VSL_STATUS_OK);

        return;
    }

    rng = hi - lo;
    if(!rng) {
        #pragma ivdep
        for(i = 0; i < len; i++) res[i] = lo;

        return;
    }

    rng++;

    if(rng <= (npy_uint64) INT_MAX) {
        int *buf = (int*) mkl_malloc(len * sizeof(int), 64);
        assert(buf != NULL);

        err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, buf, 0, (const int) rng);
        assert(err == VSL_STATUS_OK);

        #pragma ivdep
        for(i=0; i < len; i++) res[i] = lo + ((npy_uint64) buf[i]);

        mkl_free(buf);

    } else {
        npy_uint64 mask = rng;
        npy_uint64 *buf = NULL;
        int n_accepted = 0;

        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;

        buf = (npy_uint64 *) mkl_malloc(len * sizeof(npy_uint64), 64);
        assert(buf != NULL);

        while(n_accepted < len) {
            int k, batchSize = len - n_accepted;

            err = viRngUniformBits64(VSL_RNG_METHOD_UNIFORM_STD, state->stream, batchSize, (unsigned MKL_INT64 *) buf);
            assert(err == VSL_STATUS_OK);

            for(k=0; k < batchSize; k++) {
                npy_uint64 value = buf[k] & mask;
                if ( value <= rng) {
                    res[n_accepted++] = lo + value;
                }
            }
        }

        mkl_free(buf);
    }
}

void
vrk_rand_int64_vec(vrk_state *state, const int len, npy_int64 *res, const npy_int64 lo, const npy_int64 hi)
{
    npy_uint64 rng;
    int i, err;

    if (len < 1)
        return;

    rng = ((npy_uint64) hi) - ((npy_uint64) lo);

    vrk_rand_uint64_vec(state, len, (npy_uint64 *) res, 0, rng);

    for(i = 0; i < len; i++)
        res[i] = res[i] + lo;

}

const MKL_INT cholesky_storage_flags[3] = {
    VSL_MATRIX_STORAGE_FULL,
    VSL_MATRIX_STORAGE_PACKED,
    VSL_MATRIX_STORAGE_DIAGONAL
};

void
vrk_multinormal_vec_ICDF(vrk_state *state, const int len, double *res, const int dim, double * mean_vec, double *ch,
    const ch_st_enum storage_flag)
{
    int err;
    const MKL_INT storage_mode = cholesky_storage_flags[storage_flag];

    err = vdRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_ICDF, state->stream, len, res, dim, storage_mode, mean_vec, ch);
    assert(err == VSL_STATUS_OK);

}

void
vrk_multinormal_vec_BM1(vrk_state *state, const int len, double *res, const int dim, double * mean_vec, double *ch,
    const ch_st_enum storage_flag)
{
    int err;
    const MKL_INT storage_mode = cholesky_storage_flags[storage_flag];

    err = vdRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER, state->stream, len, res, dim, storage_mode, mean_vec, ch);
    assert(err == VSL_STATUS_OK);

}

void
vrk_multinormal_vec_BM2(vrk_state *state, const int len, double *res, const int dim, double * mean_vec, double *ch,
    const ch_st_enum storage_flag)
{
    int err;
    const MKL_INT storage_mode = cholesky_storage_flags[storage_flag];

    err = vdRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER2, state->stream, len, res, dim, storage_mode, mean_vec, ch);
    assert(err == VSL_STATUS_OK);

}

/* This code is taken from distribution.c, and is currently unused. It is retained here for
   possible future optimization of sampling from multinomial */

static double vrk_double(vrk_state *state) {
    double res;

    vrk_double_vec(state, 1, &res);

    return res;
}

#ifndef min
#define min(x,y) (((x)<(y))?(x):(y))
#endif

static long vrk_binomial_btpe(vrk_state *state, long n, double p)
{
    double r,q,fm,p1,xm,xl,xr,c,laml,lamr,p2,p3,p4;
    double a,u,v,s,F,rho,t,A,nrq,x1,x2,f1,f2,z,z2,w,w2,x;
    long m,y,k,i;

    r = min(p, 1.0-p);
    q = 1.0 - r;
    fm = n*r+r;
    m = (long)floor(fm);
    p1 = floor(2.195*sqrt(n*r*q)-4.6*q) + 0.5;
    xm = m + 0.5;
    xl = xm - p1;
    xr = xm + p1;
    c = 0.134 + 20.5/(15.3 + m);
    a = (fm - xl)/(fm-xl*r);
    laml = a*(1.0 + a/2.0);
    a = (xr - fm)/(xr*q);
    lamr = a*(1.0 + a/2.0);
    p2 = p1*(1.0 + 2.0*c);
    p3 = p2 + c/laml;
    p4 = p3 + c/lamr;

  /* sigh ... */
  Step10:
    nrq = n*r*q;
    u = vrk_double(state)*p4;
    v = vrk_double(state);
    if (u > p1) goto Step20;
    y = (long)floor(xm - p1*v + u);
    goto Step60;

  Step20:
    if (u > p2) goto Step30;
    x = xl + (u - p1)/c;
    v = v*c + 1.0 - fabs(m - x + 0.5)/p1;
    if (v > 1.0) goto Step10;
    y = (long)floor(x);
    goto Step50;

  Step30:
    if (u > p3) goto Step40;
    y = (long)floor(xl + log(v)/laml);
    if (y < 0) goto Step10;
    v = v*(u-p2)*laml;
    goto Step50;

  Step40:
    y = (long)floor(xr - log(v)/lamr);
    if (y > n) goto Step10;
    v = v*(u-p3)*lamr;

  Step50:
    k = labs(y - m);
    if ((k > 20) && (k < ((nrq)/2.0 - 1))) goto Step52;

    s = r/q;
    a = s*(n+1);
    F = 1.0;
    if (m < y)
    {
        for (i=m+1; i<=y; i++)
        {
            F *= (a/i - s);
        }
    }
    else if (m > y)
    {
        for (i=y+1; i<=m; i++)
        {
            F /= (a/i - s);
        }
    }
    if (v > F) goto Step10;
    goto Step60;

    Step52:
    rho = (k/(nrq))*((k*(k/3.0 + 0.625) + 0.16666666666666666)/nrq + 0.5);
    t = -k*k/(2*nrq);
    A = log(v);
    if (A < (t - rho)) goto Step60;
    if (A > (t + rho)) goto Step10;

    x1 = y+1;
    f1 = m+1;
    z = n+1-m;
    w = n-y+1;
    x2 = x1*x1;
    f2 = f1*f1;
    z2 = z*z;
    w2 = w*w;
    if (A > (xm*log(f1/x1)
           + (n-m+0.5)*log(z/w)
           + (y-m)*log(w*r/(x1*q))
           + (13680.-(462.-(132.-(99.-140./f2)/f2)/f2)/f2)/f1/166320.
           + (13680.-(462.-(132.-(99.-140./z2)/z2)/z2)/z2)/z/166320.
           + (13680.-(462.-(132.-(99.-140./x2)/x2)/x2)/x2)/x1/166320.
           + (13680.-(462.-(132.-(99.-140./w2)/w2)/w2)/w2)/w/166320.))
    {
        goto Step10;
    }

  Step60:
    if (p > 0.5)
    {
        y = n - y;
    }

    return y;
}

static long vrk_binomial_inversion(vrk_state *state, long n, double p)
{
    double q, qn, np, px, U;
    long X, bound;

    q = 1.0 - p;
    qn = exp(n * log(q));
    np = n*p;
    bound = min(n, np + 10.0*sqrt(np*q + 1));

    X = 0;
    px = qn;
    U = vrk_double(state);
    while (U > px)
    {
        X++;
        if (X > bound)
        {
            X = 0;
            px = qn;
            U = vrk_double(state);
        } else
        {
            U -= px;
            px  = ((n-X+1) * p * px)/(X*q);
        }
    }
    return X;
}

#undef min

static long vrk_binomial(vrk_state *state, long n, double p)
{
    double q;

    if (p <= 0.5)
    {
        if (p*n <= 30.0)
        {
            return vrk_binomial_inversion(state, n, p);
        }
        else
        {
            return vrk_binomial_btpe(state, n, p);
        }
    }
    else
    {
        q = 1.0-p;
        if (q*n <= 30.0)
        {
            return n - vrk_binomial_inversion(state, n, q);
        }
        else
        {
            return n - vrk_binomial_btpe(state, n, q);
        }
    }

}

