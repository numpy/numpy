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

/* The implementations of rk_hypergeometric_hyp(), rk_hypergeometric_hrua(),
 * and rk_triangular() were adapted from Ivan Frohne's rv.py which has this
 * license:
 *
 *            Copyright 1998 by Ivan Frohne; Wasilla, Alaska, U.S.A.
 *                            All Rights Reserved
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation for any purpose, free of charge, is granted subject to the
 * following conditions:
 *   The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the software.
 *
 *   THE SOFTWARE AND DOCUMENTATION IS PROVIDED WITHOUT WARRANTY OF ANY KIND,
 *   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO MERCHANTABILITY, FITNESS
 *   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHOR
 *   OR COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM OR DAMAGES IN A CONTRACT
 *   ACTION, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 *   SOFTWARE OR ITS DOCUMENTATION.
 */

#include "distributions.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>

#ifndef min
#define min(x,y) ((x<y)?x:y)
#define max(x,y) ((x>y)?x:y)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif

/*
 * log-gamma function to support some of these distributions. The
 * algorithm comes from SPECFUN by Shanjie Zhang and Jianming Jin and their
 * book "Computation of Special Functions", 1996, John Wiley & Sons, Inc.
 */
static double loggam(double x)
{
    double x0, x2, xp, gl, gl0;
    long k, n;

    static double a[10] = {8.333333333333333e-02,-2.777777777777778e-03,
         7.936507936507937e-04,-5.952380952380952e-04,
         8.417508417508418e-04,-1.917526917526918e-03,
         6.410256410256410e-03,-2.955065359477124e-02,
         1.796443723688307e-01,-1.39243221690590e+00};
    x0 = x;
    n = 0;
    if ((x == 1.0) || (x == 2.0))
    {
        return 0.0;
    }
    else if (x <= 7.0)
    {
        n = (long)(7 - x);
        x0 = x + n;
    }
    x2 = 1.0/(x0*x0);
    xp = 2*M_PI;
    gl0 = a[9];
    for (k=8; k>=0; k--)
    {
        gl0 *= x2;
        gl0 += a[k];
    }
    gl = gl0/x0 + 0.5*log(xp) + (x0-0.5)*log(x0) - x0;
    if (x <= 7.0)
    {
        for (k=1; k<=n; k++)
        {
            gl -= log(x0-1.0);
            x0 -= 1.0;
        }
    }
    return gl;
}

double rk_normal(rk_state *state, double loc, double scale)
{
    return loc + scale*rk_gauss(state);
}

double rk_standard_exponential(rk_state *state)
{
    /* We use -log(1-U) since U is [0, 1) */
    return -log(1.0 - rk_double(state));
}

double rk_exponential(rk_state *state, double scale)
{
    return scale * rk_standard_exponential(state);
}

double rk_uniform(rk_state *state, double loc, double scale)
{
    return loc + scale*rk_double(state);
}

double rk_standard_gamma(rk_state *state, double shape)
{
    double b, c;
    double U, V, X, Y;

    if (shape == 1.0)
    {
        return rk_standard_exponential(state);
    }
    else if (shape < 1.0)
    {
        for (;;)
        {
            U = rk_double(state);
            V = rk_standard_exponential(state);
            if (U <= 1.0 - shape)
            {
                X = pow(U, 1./shape);
                if (X <= V)
                {
                    return X;
                }
            }
            else
            {
                Y = -log((1-U)/shape);
                X = pow(1.0 - shape + shape*Y, 1./shape);
                if (X <= (V + Y))
                {
                    return X;
                }
            }
        }
    }
    else
    {
        b = shape - 1./3.;
        c = 1./sqrt(9*b);
        for (;;)
        {
            do
            {
                X = rk_gauss(state);
                V = 1.0 + c*X;
            } while (V <= 0.0);

            V = V*V*V;
            U = rk_double(state);
            if (U < 1.0 - 0.0331*(X*X)*(X*X)) return (b*V);
            if (log(U) < 0.5*X*X + b*(1. - V + log(V))) return (b*V);
        }
    }
}

double rk_gamma(rk_state *state, double shape, double scale)
{
    return scale * rk_standard_gamma(state, shape);
}

double rk_beta(rk_state *state, double a, double b)
{
    double Ga, Gb;

    if ((a <= 1.0) && (b <= 1.0))
    {
        double U, V, X, Y;
        /* Use Johnk's algorithm */

        while (1)
        {
            U = rk_double(state);
            V = rk_double(state);
            X = pow(U, 1.0/a);
            Y = pow(V, 1.0/b);

            /* Reject if both U and V are 0.0, which is approx 1 in 10^106 */
            if (((X + Y) <= 1.0) && ((U + V) > 0.0))
            {
                if (X + Y > 0)
                {
                    return X / (X + Y);
                }
                else
                {
                    double logX = log(U) / a;
                    double logY = log(V) / b;
                    double logM = logX > logY ? logX : logY;
                    logX -= logM;
                    logY -= logM;

                    return exp(logX - log(exp(logX) + exp(logY)));
                }
            }
        }
    }
    else
    {
        Ga = rk_standard_gamma(state, a);
        Gb = rk_standard_gamma(state, b);
        return Ga/(Ga + Gb);
    }
}

double rk_chisquare(rk_state *state, double df)
{
    return 2.0*rk_standard_gamma(state, df/2.0);
}

double rk_noncentral_chisquare(rk_state *state, double df, double nonc)
{
    if (nonc == 0){
        return rk_chisquare(state, df);
    }
    if(1 < df)
    {
        const double Chi2 = rk_chisquare(state, df - 1);
        const double N = rk_gauss(state) + sqrt(nonc);
        return Chi2 + N*N;
    }
    else
    {
        const long i = rk_poisson(state, nonc / 2.0);
        return rk_chisquare(state, df + 2 * i);
    }
}

double rk_f(rk_state *state, double dfnum, double dfden)
{
    return ((rk_chisquare(state, dfnum) * dfden) /
            (rk_chisquare(state, dfden) * dfnum));
}

double rk_noncentral_f(rk_state *state, double dfnum, double dfden, double nonc)
{
    double t = rk_noncentral_chisquare(state, dfnum, nonc) * dfden;
    return t / (rk_chisquare(state, dfden) * dfnum);
}

long rk_binomial_btpe(rk_state *state, long n, double p)
{
    double r,q,fm,p1,xm,xl,xr,c,laml,lamr,p2,p3,p4;
    double a,u,v,s,F,rho,t,A,nrq,x1,x2,f1,f2,z,z2,w,w2,x;
    long m,y,k,i;

    if (!(state->has_binomial) ||
         (state->nsave != n) ||
         (state->psave != p))
    {
        /* initialize */
        state->nsave = n;
        state->psave = p;
        state->has_binomial = 1;
        state->r = r = min(p, 1.0-p);
        state->q = q = 1.0 - r;
        state->fm = fm = n*r+r;
        state->m = m = (long)floor(state->fm);
        state->p1 = p1 = floor(2.195*sqrt(n*r*q)-4.6*q) + 0.5;
        state->xm = xm = m + 0.5;
        state->xl = xl = xm - p1;
        state->xr = xr = xm + p1;
        state->c = c = 0.134 + 20.5/(15.3 + m);
        a = (fm - xl)/(fm-xl*r);
        state->laml = laml = a*(1.0 + a/2.0);
        a = (xr - fm)/(xr*q);
        state->lamr = lamr = a*(1.0 + a/2.0);
        state->p2 = p2 = p1*(1.0 + 2.0*c);
        state->p3 = p3 = p2 + c/laml;
        state->p4 = p4 = p3 + c/lamr;
    }
    else
    {
        r = state->r;
        q = state->q;
        fm = state->fm;
        m = state->m;
        p1 = state->p1;
        xm = state->xm;
        xl = state->xl;
        xr = state->xr;
        c = state->c;
        laml = state->laml;
        lamr = state->lamr;
        p2 = state->p2;
        p3 = state->p3;
        p4 = state->p4;
    }

  /* sigh ... */
  Step10:
    nrq = n*r*q;
    u = rk_double(state)*p4;
    v = rk_double(state);
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
    /* Reject if v == 0.0 since cast of inf not well defined */
    if ((y < 0) || (v == 0.0)) goto Step10;
    v = v*(u-p2)*laml;
    goto Step50;

  Step40:
    y = (long)floor(xr - log(v)/lamr);
    /* Reject if v == 0.0 since cast of inf not well defined */
    if ((y > n) || (v == 0.0)) goto Step10;
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

long rk_binomial_inversion(rk_state *state, long n, double p)
{
    double q, qn, np, px, U;
    long X, bound;

    if (!(state->has_binomial) ||
         (state->nsave != n) ||
         (state->psave != p))
    {
        state->nsave = n;
        state->psave = p;
        state->has_binomial = 1;
        state->q = q = 1.0 - p;
        state->r = qn = exp(n * log(q));
        state->c = np = n*p;
        state->m = bound = min(n, np + 10.0*sqrt(np*q + 1));
    } else
    {
        q = state->q;
        qn = state->r;
        np = state->c;
        bound = state->m;
    }
    X = 0;
    px = qn;
    U = rk_double(state);
    while (U > px)
    {
        X++;
        if (X > bound)
        {
            X = 0;
            px = qn;
            U = rk_double(state);
        } else
        {
            U -= px;
            px  = ((n-X+1) * p * px)/(X*q);
        }
    }
    return X;
}

long rk_binomial(rk_state *state, long n, double p)
{
    double q;

    if (p <= 0.5)
    {
        if (p*n <= 30.0)
        {
            return rk_binomial_inversion(state, n, p);
        }
        else
        {
            return rk_binomial_btpe(state, n, p);
        }
    }
    else
    {
        q = 1.0-p;
        if (q*n <= 30.0)
        {
            return n - rk_binomial_inversion(state, n, q);
        }
        else
        {
            return n - rk_binomial_btpe(state, n, q);
        }
    }

}

long rk_negative_binomial(rk_state *state, double n, double p)
{
    double Y;

    Y = rk_gamma(state, n, (1-p)/p);
    return rk_poisson(state, Y);
}

long rk_poisson_mult(rk_state *state, double lam)
{
    long X;
    double prod, U, enlam;

    enlam = exp(-lam);
    X = 0;
    prod = 1.0;
    while (1)
    {
        U = rk_double(state);
        prod *= U;
        if (prod > enlam)
        {
            X += 1;
        }
        else
        {
            return X;
        }
    }
}

/*
 * The transformed rejection method for generating Poisson random variables
 * W. Hoermann
 * Insurance: Mathematics and Economics 12, 39-45 (1993)
 */
#define LS2PI 0.91893853320467267
#define TWELFTH 0.083333333333333333333333
long rk_poisson_ptrs(rk_state *state, double lam)
{
    long k;
    double U, V, slam, loglam, a, b, invalpha, vr, us;

    slam = sqrt(lam);
    loglam = log(lam);
    b = 0.931 + 2.53*slam;
    a = -0.059 + 0.02483*b;
    invalpha = 1.1239 + 1.1328/(b-3.4);
    vr = 0.9277 - 3.6224/(b-2);

    while (1)
    {
        U = rk_double(state) - 0.5;
        V = rk_double(state);
        us = 0.5 - fabs(U);
        k = (long)floor((2*a/us + b)*U + lam + 0.43);
        if ((us >= 0.07) && (V <= vr))
        {
            return k;
        }
        if ((k < 0) ||
            ((us < 0.013) && (V > us)))
        {
            continue;
        }
        if ((log(V) + log(invalpha) - log(a/(us*us)+b)) <=
            (-lam + k*loglam - loggam(k+1)))
        {
            return k;
        }


    }

}

long rk_poisson(rk_state *state, double lam)
{
    if (lam >= 10)
    {
        return rk_poisson_ptrs(state, lam);
    }
    else if (lam == 0)
    {
        return 0;
    }
    else
    {
        return rk_poisson_mult(state, lam);
    }
}

double rk_standard_cauchy(rk_state *state)
{
    return rk_gauss(state) / rk_gauss(state);
}

double rk_standard_t(rk_state *state, double df)
{
    double N, G, X;

    N = rk_gauss(state);
    G = rk_standard_gamma(state, df/2);
    X = sqrt(df/2)*N/sqrt(G);
    return X;
}

/* Uses the rejection algorithm compared against the wrapped Cauchy
   distribution suggested by Best and Fisher and documented in
   Chapter 9 of Luc's Non-Uniform Random Variate Generation.
   http://cg.scs.carleton.ca/~luc/rnbookindex.html
   (but corrected to match the algorithm in R and Python)
*/
double rk_vonmises(rk_state *state, double mu, double kappa)
{
    double s;
    double U, V, W, Y, Z;
    double result, mod;
    int neg;

    if (kappa < 1e-8)
    {
        return M_PI * (2*rk_double(state)-1);
    }
    else
    {
        /* with double precision rho is zero until 1.4e-8 */
        if (kappa < 1e-5) {
            /*
             * second order taylor expansion around kappa = 0
             * precise until relatively large kappas as second order is 0
             */
            s = (1./kappa + kappa);
        }
        else {
            double r = 1 + sqrt(1 + 4*kappa*kappa);
            double rho = (r - sqrt(2*r)) / (2*kappa);
            s = (1 + rho*rho)/(2*rho);
        }

        while (1)
        {
        U = rk_double(state);
            Z = cos(M_PI*U);
            W = (1 + s*Z)/(s + Z);
            Y = kappa * (s - W);
            V = rk_double(state);
            if ((Y*(2-Y) - V >= 0) || (log(Y/V)+1 - Y >= 0))
            {
                break;
            }
        }

        U = rk_double(state);

        result = acos(W);
        if (U < 0.5)
        {
        result = -result;
        }
        result += mu;
        neg = (result < 0);
        mod = fabs(result);
        mod = (fmod(mod+M_PI, 2*M_PI)-M_PI);
        if (neg)
        {
            mod *= -1;
        }

        return mod;
    }
}

double rk_pareto(rk_state *state, double a)
{
    return exp(rk_standard_exponential(state)/a) - 1;
}

double rk_weibull(rk_state *state, double a)
{
    if (a == 0.0) {
        return 0.0;
    }
    return pow(rk_standard_exponential(state), 1./a);
}

double rk_power(rk_state *state, double a)
{
    return pow(1 - exp(-rk_standard_exponential(state)), 1./a);
}

double rk_laplace(rk_state *state, double loc, double scale)
{
    double U;

    U = rk_double(state);
    if (U >= 0.5)
    {
        U = loc - scale * log(2.0 - U - U);

    } else if (U > 0.0)
    {
        U = loc + scale * log(U + U);
    } else
    {
        /* Reject if U == 0.0 */
        return rk_laplace(state, loc, scale);
    }
    return U;
}

double rk_gumbel(rk_state *state, double loc, double scale)
{
    double U;

    U = 1.0 - rk_double(state);
    if (U < 1.0)
        return loc - scale * log(-log(U));
    return rk_gumbel(state, loc, scale);
}

double rk_logistic(rk_state *state, double loc, double scale)
{
    double U;

    U = rk_double(state);
    if (U > 0.0)
        return loc + scale * log(U/(1.0 - U));
    return rk_logistic(state, loc, scale);
}

double rk_lognormal(rk_state *state, double mean, double sigma)
{
    return exp(rk_normal(state, mean, sigma));
}

double rk_rayleigh(rk_state *state, double mode)
{
    return mode*sqrt(-2.0 * log(1.0 - rk_double(state)));
}

double rk_wald(rk_state *state, double mean, double scale)
{
    double U, X, Y;
    double mu_2l;

    mu_2l = mean / (2*scale);
    Y = rk_gauss(state);
    Y = mean*Y*Y;
    X = mean + mu_2l*(Y - sqrt(4*scale*Y + Y*Y));
    U = rk_double(state);
    if (U <= mean/(mean+X))
    {
        return X;
    } else
    {
        return mean*mean/X;
    }
}

long rk_zipf(rk_state *state, double a)
{
    double am1, b;

    am1 = a - 1.0;
    b = pow(2.0, am1);
    while (1) {
        double T, U, V, X;

        U = 1.0 - rk_double(state);
        V = rk_double(state);
        X = floor(pow(U, -1.0/am1));
        /*
         * The real result may be above what can be represented in a signed
         * long. Since this is a straightforward rejection algorithm, we can
         * just reject this value. This function then models a Zipf
         * distribution truncated to sys.maxint.
         */
        if (X > LONG_MAX || X < 1.0) {
            continue;
        }

        T = pow(1.0 + 1.0/X, am1);
        if (V*X*(T - 1.0)/(b - 1.0) <= T/b) {
            return (long)X;
        }
    }
}

long rk_geometric_search(rk_state *state, double p)
{
    double U;
    long X;
    double sum, prod, q;

    X = 1;
    sum = prod = p;
    q = 1.0 - p;
    U = rk_double(state);
    while (U > sum)
    {
        prod *= q;
        sum += prod;
        X++;
    }
    return X;
}

long rk_geometric_inversion(rk_state *state, double p)
{
    return (long)ceil(log(1.0-rk_double(state))/log(1.0-p));
}

long rk_geometric(rk_state *state, double p)
{
    if (p >= 0.333333333333333333333333)
    {
        return rk_geometric_search(state, p);
    } else
    {
        return rk_geometric_inversion(state, p);
    }
}

long rk_hypergeometric_hyp(rk_state *state, long good, long bad, long sample)
{
    long d1, K, Z;
    double d2, U, Y;

    d1 = bad + good - sample;
    d2 = (double)min(bad, good);

    Y = d2;
    K = sample;
    while (Y > 0.0)
    {
        U = rk_double(state);
        Y -= (long)floor(U + Y/(d1 + K));
        K--;
        if (K == 0) break;
    }
    Z = (long)(d2 - Y);
    if (good > bad) Z = sample - Z;
    return Z;
}

/* D1 = 2*sqrt(2/e) */
/* D2 = 3 - 2*sqrt(3/e) */
#define D1 1.7155277699214135
#define D2 0.8989161620588988
long rk_hypergeometric_hrua(rk_state *state, long good, long bad, long sample)
{
    long mingoodbad, maxgoodbad, popsize, m, d9;
    double d4, d5, d6, d7, d8, d10, d11;
    long Z;
    double T, W, X, Y;

    mingoodbad = min(good, bad);
    popsize = good + bad;
    maxgoodbad = max(good, bad);
    m = min(sample, popsize - sample);
    d4 = ((double)mingoodbad) / popsize;
    d5 = 1.0 - d4;
    d6 = m*d4 + 0.5;
    d7 = sqrt((double)(popsize - m) * sample * d4 * d5 / (popsize - 1) + 0.5);
    d8 = D1*d7 + D2;
    d9 = (long)floor((double)(m + 1) * (mingoodbad + 1) / (popsize + 2));
    d10 = (loggam(d9+1) + loggam(mingoodbad-d9+1) + loggam(m-d9+1) +
           loggam(maxgoodbad-m+d9+1));
    d11 = min(min(m, mingoodbad)+1.0, floor(d6+16*d7));
    /* 16 for 16-decimal-digit precision in D1 and D2 */

    while (1)
    {
        X = rk_double(state);
        Y = rk_double(state);
        W = d6 + d8*(Y- 0.5)/X;

        /* fast rejection: */
        if ((W < 0.0) || (W >= d11)) continue;

        Z = (long)floor(W);
        T = d10 - (loggam(Z+1) + loggam(mingoodbad-Z+1) + loggam(m-Z+1) +
                   loggam(maxgoodbad-m+Z+1));

        /* fast acceptance: */
        if ((X*(4.0-X)-3.0) <= T) break;

        /* fast rejection: */
        if (X*(X-T) >= 1) continue;

        if (2.0*log(X) <= T) break;  /* acceptance */
    }

    /* this is a correction to HRUA* by Ivan Frohne in rv.py */
    if (good > bad) Z = m - Z;

    /* another fix from rv.py to allow sample to exceed popsize/2 */
    if (m < sample) Z = good - Z;

    return Z;
}
#undef D1
#undef D2

long rk_hypergeometric(rk_state *state, long good, long bad, long sample)
{
    if (sample > 10)
    {
        return rk_hypergeometric_hrua(state, good, bad, sample);
    } else
    {
        return rk_hypergeometric_hyp(state, good, bad, sample);
    }
}

double rk_triangular(rk_state *state, double left, double mode, double right)
{
    double base, leftbase, ratio, leftprod, rightprod;
    double U;

    base = right - left;
    leftbase = mode - left;
    ratio = leftbase / base;
    leftprod = leftbase*base;
    rightprod = (right - mode)*base;

    U = rk_double(state);
    if (U <= ratio)
    {
        return left + sqrt(U*leftprod);
    } else
    {
      return right - sqrt((1.0 - U) * rightprod);
    }
}

long rk_logseries(rk_state *state, double p)
{
    double q, r, U, V;
    long result;

    r = log(1.0 - p);

    while (1) {
        V = rk_double(state);
        if (V >= p) {
            return 1;
        }
        U = rk_double(state);
        q = 1.0 - exp(r*U);
        if (V <= q*q) {
            result = (long)floor(1 + log(V)/log(q));
            /* Reject if v == 0.0 since cast of inf not well defined */
            if ((result < 1) || (V == 0.0)) {
                continue;
            }
            else {
                return result;
            }
        }
        if (V >= q) {
            return 1;
        }
        return 2;
    }
}
