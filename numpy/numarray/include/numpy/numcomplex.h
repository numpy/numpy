/* See numarray.h for Complex32, Complex64:

typedef struct { Float32 r, i; } Complex32;
typedef struct { Float64 r, i; } Complex64;

*/
typedef struct { Float32 a, theta; } PolarComplex32;
typedef struct { Float64 a, theta; } PolarComplex64;

#define NUM_SQ(x)         ((x)*(x))

#define NUM_CABSSQ(p)     (NUM_SQ((p).r) + NUM_SQ((p).i))

#define NUM_CABS(p)       sqrt(NUM_CABSSQ(p))

#define NUM_C_TO_P(c, p)  (p).a = NUM_CABS(c);                                \
                          (p).theta = atan2((c).i, (c).r);

#define NUM_P_TO_C(p, c)  (c).r = (p).a*cos((p).theta);                       \
                          (c).i = (p).a*sin((p).theta);

#define NUM_CASS(p, q)    (q).r = (p).r, (q).i = (p).i

#define NUM_CADD(p, q, s) (s).r = (p).r + (q).r,                              \
                          (s).i = (p).i + (q).i

#define NUM_CSUB(p, q, s) (s).r = (p).r - (q).r,                              \
                          (s).i = (p).i - (q).i

#define NUM_CMUL(p, q, s)                                                     \
          { Float64 rp = (p).r;                                               \
            Float64 rq = (q).r;                                               \
                (s).r = rp*rq - (p).i*(q).i;                                  \
                (s).i = rp*(q).i + rq*(p).i;                                  \
          }

#define NUM_CDIV(p, q, s)                                                     \
          {                                                                   \
	    Float64 rp = (p).r;                                               \
            Float64 ip = (p).i;                                               \
            Float64 rq = (q).r;                                               \
  	    if ((q).i != 0) {                                                 \
                Float64 temp = NUM_CABSSQ(q);                                 \
                (s).r = (rp*rq+(p).i*(q).i)/temp;                             \
                (s).i = (rq*(p).i-(q).i*rp)/temp;                             \
            } else {                                                          \
 	        (s).r = rp/rq;                                                \
		(s).i = ip/rq;                                                \
            }                                                                 \
          }

#define NUM_CREM(p, q, s)                                                     \
          {  Complex64 r;                                                     \
             NUM_CDIV(p, q, r);                                               \
             r.r = floor(r.r);                                                \
             r.i = 0;                                                         \
             NUM_CMUL(r, q, r);                                               \
             NUM_CSUB(p, r, s);                                               \
          }

#define NUM_CMINUS(p, s)  (s).r = -(p).r; (s).i = -(p).i;
#define NUM_CNEG NUM_CMINUS

#define NUM_CEQ(p, q)  (((p).r == (q).r) && ((p).i == (q).i))
#define NUM_CNE(p, q)  (((p).r != (q).r) || ((p).i != (q).i))
#define NUM_CLT(p, q)  ((p).r < (q).r)
#define NUM_CGT(p, q)  ((p).r > (q).r)
#define NUM_CLE(p, q)  ((p).r <= (q).r)
#define NUM_CGE(p, q)  ((p).r >= (q).r)

/* e**z = e**x * (cos(y)+ i*sin(y)) where z = x + i*y 
   so e**z = e**x * cos(y) +  i * e**x * sin(y)
*/
#define NUM_CEXP(p, s)                                                        \
          { Float64 ex = exp((p).r);                                          \
            (s).r = ex * cos((p).i);                                          \
            (s).i = ex * sin((p).i);                                          \
          }

/* e**w = z;     w = u + i*v;     z = r * e**(i*theta);

e**u * e**(i*v) = r * e**(i*theta); 

log(z) = w;  log(z) = log(r) + i*theta;
 */
#define NUM_CLOG(p, s)                                                      \
          { PolarComplex64 temp;   NUM_C_TO_P(p, temp);                     \
            (s).r = num_log(temp.a);                                        \
            (s).i = temp.theta;                                             \
          }

#define NUM_LOG10_E  0.43429448190325182

#define NUM_CLOG10(p, s)                                                    \
          { NUM_CLOG(p, s);                                                 \
            (s).r *= NUM_LOG10_E;                                           \
            (s).i *= NUM_LOG10_E;                                           \
          }

/* s = p ** q  */
#define NUM_CPOW(p, q, s) { if (NUM_CABSSQ(p) == 0) {                        \
	                       if ((q).r == 0 && (q).i == 0) {               \
                                   (s).r = (s).i = 1;                        \
                               } else {                                      \
                                   (s).r = (s).i = 0;                        \
                               }                                             \
                            } else {                                         \
                               NUM_CLOG(p, s);                               \
                               NUM_CMUL(s, q, s);                            \
                               NUM_CEXP(s, s);                               \
                            }                                                \
                          }
  
#define NUM_CSQRT(p, s)   { Complex64 temp; temp.r = 0.5; temp.i=0;           \
                            NUM_CPOW(p, temp, s);                             \
                          }

#define NUM_CSQR(p, s)   { Complex64 temp; temp.r = 2.0; temp.i=0;            \
                            NUM_CPOW(p, temp, s);                             \
                          }

#define NUM_CSIN(p, s) { Float64 sp = sin((p).r);                             \
                         Float64 cp = cos((p).r);                             \
                         (s).r = cosh((p).i) * sp;                            \
                         (s).i = sinh((p).i) * cp;                            \
                       }

#define NUM_CCOS(p, s) { Float64 sp = sin((p).r);                             \
                         Float64 cp = cos((p).r);                             \
                         (s).r = cosh((p).i) * cp;                            \
                         (s).i = -sinh((p).i) * sp;                           \
                       }

#define NUM_CTAN(p, s) { Complex64 ss, cs;                                    \
                         NUM_CSIN(p, ss);                                     \
                         NUM_CCOS(p, cs);                                     \
                         NUM_CDIV(ss, cs, s);                                 \
                       }

#define NUM_CSINH(p, s) { Float64 sp = sin((p).i);                            \
                          Float64 cp = cos((p).i);                            \
                         (s).r = sinh((p).r) * cp;                            \
                         (s).i = cosh((p).r) * sp;                            \
                       }

#define NUM_CCOSH(p, s) { Float64 sp = sin((p).i);                            \
                          Float64 cp = cos((p).i);                            \
                         (s).r = cosh((p).r) * cp;                            \
                         (s).i = sinh((p).r) * sp;                            \
                       }

#define NUM_CTANH(p, s) { Complex64 ss, cs;                                   \
                         NUM_CSINH(p, ss);                                    \
                         NUM_CCOSH(p, cs);                                    \
                         NUM_CDIV(ss, cs, s);                                 \
                       }

#define NUM_CRPOW(p, v, s) { Complex64 cr; cr.r = v; cr.i = 0;                \
                             NUM_CPOW(p,cr,s);                                \
                           }

#define NUM_CRMUL(p, v, s) (s).r = (p).r * v;  (s).i = (p).i * v;

#define NUM_CIMUL(p, s)    { Float64 temp = (s).r;                            \
                             (s).r = -(p).i;  (s).i = temp;                   \
                           }

/* asin(z) = -i * log(i*z + (1 - z**2)**0.5) */
#define NUM_CASIN(p, s) { Complex64 p1;  NUM_CASS(p, p1);                     \
                         NUM_CIMUL(p, p1);                                    \
			 NUM_CMUL(p, p, s);                                   \
                         NUM_CNEG(s, s);                                      \
                         (s).r += 1;                                          \
                         NUM_CRPOW(s, 0.5, s);                                \
                         NUM_CADD(p1, s, s);                                  \
                         NUM_CLOG(s, s);                                      \
                         NUM_CIMUL(s, s);                                     \
                         NUM_CNEG(s, s);                                      \
                       }

/* acos(z) = -i * log(z + i*(1 - z**2)**0.5) */
#define NUM_CACOS(p, s) { Complex64 p1;  NUM_CASS(p, p1);                     \
 			 NUM_CMUL(p, p, s);                                   \
                         NUM_CNEG(s, s);                                      \
                         (s).r += 1;                                          \
                         NUM_CRPOW(s, 0.5, s);                                \
                         NUM_CIMUL(s, s);                                     \
                         NUM_CADD(p1, s, s);                                  \
                         NUM_CLOG(s, s);                                      \
                         NUM_CIMUL(s, s);                                     \
                         NUM_CNEG(s, s);                                      \
                       }

/* atan(z) = i/2 * log( (i+z) / (i - z) )  */
#define NUM_CATAN(p, s) { Complex64 p1, p2;                                   \
                         NUM_CASS(p, p1); NUM_CNEG(p, p2);                    \
                         p1.i += 1;                                           \
                         p2.i += 1;                                           \
                         NUM_CDIV(p1, p2, s);                                 \
                         NUM_CLOG(s, s);                                      \
                         NUM_CIMUL(s, s);                                     \
                         NUM_CRMUL(s, 0.5, s);                                \
                       }
                         
/* asinh(z) = log( z + (z**2 + 1)**0.5 )   */
#define NUM_CASINH(p, s) { Complex64 p1;   NUM_CASS(p, p1);                   \
                          NUM_CMUL(p, p, s);                                  \
                          (s).r += 1;                                         \
                          NUM_CRPOW(s, 0.5, s);                               \
                          NUM_CADD(p1, s, s);                                 \
                          NUM_CLOG(s, s);                                     \
                        }

/* acosh(z) = log( z + (z**2 - 1)**0.5 )   */
#define NUM_CACOSH(p, s) { Complex64 p1;   NUM_CASS(p, p1);                   \
                          NUM_CMUL(p, p, s);                                  \
                          (s).r -= 1;                                         \
                          NUM_CRPOW(s, 0.5, s);                               \
                          NUM_CADD(p1, s, s);                                 \
                          NUM_CLOG(s, s);                                     \
                        }

/* atanh(z) = 1/2 * log( (1+z)/(1-z) )   */
#define NUM_CATANH(p, s) { Complex64 p1, p2;                                  \
                          NUM_CASS(p, p1); NUM_CNEG(p, p2);                   \
                          p1.r += 1;                                          \
                          p2.r += 1;                                          \
                          NUM_CDIV(p1, p2, s);                                \
                          NUM_CLOG(s, s);                                     \
                          NUM_CRMUL(s, 0.5, s);                               \
                        }


#define NUM_CMIN(p, q) (NUM_CLE(p, q) ? p : q)
#define NUM_CMAX(p, q) (NUM_CGE(p, q) ? p : q)

#define NUM_CNZ(p)      (((p).r != 0) || ((p).i != 0))
#define NUM_CLAND(p, q) (NUM_CNZ(p) & NUM_CNZ(q))
#define NUM_CLOR(p, q)  (NUM_CNZ(p)  | NUM_CNZ(q))
#define NUM_CLXOR(p, q) (NUM_CNZ(p) ^ NUM_CNZ(q))
#define NUM_CLNOT(p)    (!NUM_CNZ(p))

#define NUM_CFLOOR(p, s) (s).r = floor((p).r); (s).i = floor((p).i);
#define NUM_CCEIL(p, s) (s).r = ceil((p).r); (s).i = ceil((p).i);

#define NUM_CFABS(p, s)  (s).r = fabs((p).r);  (s).i = fabs((p).i);
#define NUM_CROUND(p, s) (s).r = num_round((p).r); (s).i = num_round((p).i);
#define NUM_CHYPOT(p, q, s) { Complex64 t;                                    \
                              NUM_CSQR(p, s);  NUM_CSQR(q, t);                \
                              NUM_CADD(s, t, s);                              \
                              NUM_CSQRT(s, s);                                \
                            }
