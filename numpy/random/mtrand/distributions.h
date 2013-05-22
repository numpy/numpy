/* Copyright 2005 Robert Kern (robert.kern@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef _RK_DISTR_
#define _RK_DISTR_

#include "randomkit.h"

#ifdef __cplusplus
extern "C" {
#endif

/* References:
 * 
 * Devroye, Luc. _Non-Uniform Random Variate Generation_.
 *  Springer-Verlag, New York, 1986.
 *  http://cgm.cs.mcgill.ca/~luc/rnbookindex.html
 * 
 * Kachitvichyanukul, V. and Schmeiser, B. W. Binomial Random Variate
 *  Generation. Communications of the ACM, 31, 2 (February, 1988) 216.
 *
 * Hoermann, W. The Transformed Rejection Method for Generating Poisson Random
 *  Variables. Insurance: Mathematics and Economics, (to appear)
 *  http://citeseer.csail.mit.edu/151115.html
 *
 * Marsaglia, G. and Tsang, W. W. A Simple Method for Generating Gamma 
 * Variables. ACM Transactions on Mathematical Software, Vol. 26, No. 3,
 * September 2000, Pages 363–372.
 */
 
/* Normal distribution with mean=loc and standard deviation=scale. */
RK_EXTERN double rk_normal(rk_state *state, double loc, double scale);

/* Standard exponential distribution (mean=1) computed by inversion of the 
 * CDF. */
RK_EXTERN double rk_standard_exponential(rk_state *state);

/* Exponential distribution with mean=scale. */
RK_EXTERN double rk_exponential(rk_state *state, double scale);

/* Uniform distribution on interval [loc, loc+scale). */
RK_EXTERN double rk_uniform(rk_state *state, double loc, double scale);

/* Standard gamma distribution with shape parameter. 
 * When shape < 1, the algorithm given by (Devroye p. 304) is used.
 * When shape == 1, a Exponential variate is generated.
 * When shape > 1, the small and fast method of (Marsaglia and Tsang 2000) 
 * is used.
 */
RK_EXTERN double rk_standard_gamma(rk_state *state, double shape);

/* Gamma distribution with shape and scale. */
RK_EXTERN double rk_gamma(rk_state *state, double shape, double scale);

/* Beta distribution computed by combining two gamma variates (Devroye p. 432).
 */
RK_EXTERN double rk_beta(rk_state *state, double a, double b);

/* Chi^2 distribution computed by transforming a gamma variate (it being a
 * special case Gamma(df/2, 2)). */
RK_EXTERN double rk_chisquare(rk_state *state, double df);

/* Noncentral Chi^2 distribution computed by modifying a Chi^2 variate. */
RK_EXTERN double rk_noncentral_chisquare(rk_state *state, double df, double nonc);

/* F distribution computed by taking the ratio of two Chi^2 variates. */
RK_EXTERN double rk_f(rk_state *state, double dfnum, double dfden);

/* Noncentral F distribution computed by taking the ratio of a noncentral Chi^2
 * and a Chi^2 variate. */
RK_EXTERN double rk_noncentral_f(rk_state *state, double dfnum, double dfden, double nonc);

/* Binomial distribution with n Bernoulli trials with success probability p.
 * When n*p <= 30, the "Second waiting time method" given by (Devroye p. 525) is
 * used. Otherwise, the BTPE algorithm of (Kachitvichyanukul and Schmeiser 1988)
 * is used. */
RK_EXTERN long rk_binomial(rk_state *state, long n, double p);

/* Binomial distribution using BTPE. */
RK_EXTERN long rk_binomial_btpe(rk_state *state, long n, double p);

/* Binomial distribution using inversion and chop-down */
RK_EXTERN long rk_binomial_inversion(rk_state *state, long n, double p);

/* Negative binomial distribution computed by generating a Gamma(n, (1-p)/p)
 * variate Y and returning a Poisson(Y) variate (Devroye p. 543). */
RK_EXTERN long rk_negative_binomial(rk_state *state, double n, double p);

/* Poisson distribution with mean=lam.
 * When lam < 10, a basic algorithm using repeated multiplications of uniform
 * variates is used (Devroye p. 504).
 * When lam >= 10, algorithm PTRS from (Hoermann 1992) is used.
 */
RK_EXTERN long rk_poisson(rk_state *state, double lam);

/* Poisson distribution computed by repeated multiplication of uniform variates.
 */
RK_EXTERN long rk_poisson_mult(rk_state *state, double lam);

/* Poisson distribution computer by the PTRS algorithm. */
RK_EXTERN long rk_poisson_ptrs(rk_state *state, double lam);

/* Standard Cauchy distribution computed by dividing standard gaussians 
 * (Devroye p. 451). */
RK_EXTERN double rk_standard_cauchy(rk_state *state);

/* Standard t-distribution with df degrees of freedom (Devroye p. 445 as
 * corrected in the Errata). */
RK_EXTERN double rk_standard_t(rk_state *state, double df);

/* von Mises circular distribution with center mu and shape kappa on [-pi,pi]
 * (Devroye p. 476 as corrected in the Errata). */
RK_EXTERN double rk_vonmises(rk_state *state, double mu, double kappa);

/* Pareto distribution via inversion (Devroye p. 262) */
RK_EXTERN double rk_pareto(rk_state *state, double a);

/* Weibull distribution via inversion (Devroye p. 262) */
RK_EXTERN double rk_weibull(rk_state *state, double a);

/* Power distribution via inversion (Devroye p. 262) */
RK_EXTERN double rk_power(rk_state *state, double a);

/* Laplace distribution */
RK_EXTERN double rk_laplace(rk_state *state, double loc, double scale);

/* Gumbel distribution */
RK_EXTERN double rk_gumbel(rk_state *state, double loc, double scale);

/* Logistic distribution */
RK_EXTERN double rk_logistic(rk_state *state, double loc, double scale);

/* Log-normal distribution */
RK_EXTERN double rk_lognormal(rk_state *state, double mean, double sigma);

/* Rayleigh distribution */
RK_EXTERN double rk_rayleigh(rk_state *state, double mode);

/* Wald distribution */
RK_EXTERN double rk_wald(rk_state *state, double mean, double scale);

/* Zipf distribution */
RK_EXTERN long rk_zipf(rk_state *state, double a);

/* Geometric distribution */
RK_EXTERN long rk_geometric(rk_state *state, double p);
RK_EXTERN long rk_geometric_search(rk_state *state, double p);
RK_EXTERN long rk_geometric_inversion(rk_state *state, double p);

/* Hypergeometric distribution */
RK_EXTERN long rk_hypergeometric(rk_state *state, long good, long bad, long sample);
RK_EXTERN long rk_hypergeometric_hyp(rk_state *state, long good, long bad, long sample);
RK_EXTERN long rk_hypergeometric_hrua(rk_state *state, long good, long bad, long sample);

/* Triangular distribution */
RK_EXTERN double rk_triangular(rk_state *state, double left, double mode, double right);

/* Logarithmic series distribution */
RK_EXTERN long rk_logseries(rk_state *state, double p);

#ifdef __cplusplus
}
#endif

#endif /* _RK_DISTR_ */
