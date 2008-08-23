
typedef struct {double real; double imag;} cdouble;
typedef struct {double real; double imag;} cfloat;

/* Add arrays of contiguous data */
void zadd(cdouble *a, cdouble *b, cdouble *c, long n)
{
	while (n--) {
		c->real = a->real + b->real;
		c->imag = a->imag + b->imag;
		a++; b++; c++; 
	}	
}

void cadd(cfloat *a, cfloat *b, cfloat *c, long n) 
{
	while (n--) {
		c->real = a->real + b->real;
		c->imag = a->imag + b->imag;
		a++; b++; c++; 
	}	
}

void dadd(double *a, double *b, double *c, long n) 
{
	while (n--) {
		*c++ = *a++ + *b++;
	}	
}

void sadd(float *a, float *b, float *c, long n) 
{
	while (n--) {
		*c++ = *a++ + *b++;
	}
}

/* Assumes b is contiguous and 
   a has strides that are multiples of sizeof(double)
*/
void dfilter2d(double *a, double *b, int *astrides, int *dims)
{
    int i, j, M, N, S0, S1;
    int r, c, rm1, rp1, cp1, cm1;
    
    M = dims[0]; N = dims[1];
    S0 = astrides[0]/sizeof(double); 
    S1=astrides[1]/sizeof(double);
    for (i=1; i<M-1; i++) {
	r = i*S0; rp1 = r+S0; rm1 = r-S0;
	for (j=1; j<N-1; j++) {
	    c = j*S1; cp1 = j+S1; cm1 = j-S1;
	    b[i*N+j] = a[r+c] +			\
		(a[rp1+c] + a[rm1+c] +		\
		 a[r+cp1] + a[r+cm1])*0.5 +	\
		(a[rp1+cp1] + a[rp1+cm1] +	\
		 a[rm1+cp1] + a[rm1+cp1])*0.25;
	}
    }
}
