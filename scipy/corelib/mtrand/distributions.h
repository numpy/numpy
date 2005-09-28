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
 *  Variables, Insurance: Mathematics and Economics, (to appear)
 *  http://citeseer.csail.mit.edu/151115.html
 */

/* Normal distribution with mean=loc and standard deviation=scale. */
extern double rk_normal(rk_state *state, double loc, double scale);

/* Standard exponential distribution (mean=1) computed by inversion of the 
 * CDF. */
extern double rk_standard_exponential(rk_state *state);

/* Exponential distribution with mean=scale. */
extern double rk_exponential(rk_state *state, double scale);

/* Uniform distribution on interval [loc, loc+scale). */
extern double rk_uniform(rk_state *state, double loc, double scale);

/* Standard gamma distribution with shape parameter. 
 * When shape < 1, the algorithm given by (Devroye p. 304) is used.
 * When shape == 1, a Exponential variate is generated.
 * When shape > 1, Cheng's GB rejection algorithm from Devroye p. 413 with the
 * corrections given in the Errata is used.
 */
extern double rk_standard_gamma(rk_state *state, double shape);

/* Gamma distribution with shape and scale. */
extern double rk_gamma(rk_state *state, double shape, double scale);

/* Beta distribution computed by combining two gamma variates (Devroye p. 432).
 */
extern double rk_beta(rk_state *state, double a, double b);

/* Chi^2 distribution computed by transforming a gamma variate (it being a
 * special case Gamma(df/2, 2)). */
extern double rk_chisquare(rk_state *state, double df);

/* Noncentral Chi^2 distribution computed by modifying a Chi^2 variate. */
extern double rk_noncentral_chisquare(rk_state *state, double df, double nonc);

/* F distribution computed by taking the ratio of two Chi^2 variates. */
extern double rk_f(rk_state *state, double dfnum, double dfden);

/* Noncentral F distribution computed by taking the ratio of a noncentral Chi^2
 * and a Chi^2 variate. */
extern double rk_noncentral_f(rk_state *state, double dfnum, double dfden, double nonc);

/* Binomial distribution with n Bernoulli trials with success probability p.
 * When n*p <= 30, the "Second waiting time method" given by (Devroye p. 525) is
 * used. Otherwise, the BTPE algorithm of (Kachitvichyanukul and Schmeiser 1988)
 * is used. */
extern long rk_binomial(rk_state *state, long n, double p);

/* Binomial distribution using BTPE. */
extern long rk_binomial_btpe(rk_state *state, long n, double p);

/* Binomial distribution using the waiting time algorithm. */
extern long rk_binomial_waiting(rk_state *state, long n, double p);

/* Negative binomial distribution computed by generating a Gamma(n, (1-p)/p)
 * variate Y and returning a Poisson(Y) variate (Devroye p. 543). */
extern long rk_negative_binomial(rk_state *state, long n, double p);

/* Poisson distribution with mean=lam.
 * When lam < 10, a basic algorithm using repeated multiplications of uniform
 * variates is used (Devroye p. 504).
 * When lam >= 10, algorithm PTRS from (Hoermann 1992) is used.
 */
extern long rk_poisson(rk_state *state, double lam);

/* Poisson distribution computed by repeated multiplication of uniform variates.
 */
extern long rk_poisson_mult(rk_state *state, double lam);

/* Poisson distribution computer by the PTRS algorithm. */
extern long rk_poisson_ptrs(rk_state *state, double lam);

/* Standard Cauchy distribution computed by dividing standard gaussians 
 * (Devroye p. 451). */
extern double rk_standard_cauchy(rk_state *state);

/* Standard t-distribution with df degrees of freedom (Devroye p. 445 as
 * corrected in the Errata). */
extern double rk_standard_t(rk_state *state, double df);

/* von Mises circular distribution with center mu and shape kappa on [-pi,pi]
 * (Devroye p. 476 as corrected in the Errata). */
extern double rk_vonmises(rk_state *state, double mu, double kappa);

/* Pareto distribution via inversion (Devroye p. 262) */
extern double rk_pareto(rk_state *state, double a);

/* Weibull distribution via inversion (Devroye p. 262) */
extern double rk_weibull(rk_state *state, double a);

/* Power distribution via inversion (Devroye p. 262) */
extern double rk_power(rk_state *state, double a);

#ifdef __cplusplus
}
#endif


#endif /* _RK_DISTR_ */
