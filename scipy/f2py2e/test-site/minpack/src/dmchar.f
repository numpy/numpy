      subroutine dmchar(ibeta,it,irnd,ngrd,machep,negep,iexp,minexp,
     1                   maxexp,eps,epsneg,xmin,xmax)
c
      integer i,ibeta,iexp,irnd,it,iz,j,k,machep,maxexp,minexp,
     1        mx,negep,ngrd
      double precision a,b,beta,betain,betam1,eps,epsneg,one,xmax,
     1                 xmin,y,z,zero
c
c     this subroutine is intended to determine the characteristics
c     of the floating-point arithmetic system that are specified
c     below.  the first three are determined according to an
c     algorithm due to m. malcolm, cacm 15 (1972), pp. 949-951,
c     incorporating some, but not all, of the improvements
c     suggested by m. gentleman and s. marovich, cacm 17 (1974),
c     pp. 276-277.
c
c
c       ibeta   - the radix of the floating-point representation
c       it      - the number of base ibeta digits in the floating-point
c                 significand
c       irnd    - 0 if floating-point addition chops,
c                 1 if floating-point addition rounds
c       ngrd    - the number of guard digits for multiplication.  it is
c                 0 if  irnd=1, or if  irnd=0  and only  it  base  ibeta
c                   digits participate in the post normalization shift
c                   of the floating-point significand in multiplication
c                 1 if  irnd=0  and more than  it  base  ibeta  digits
c                   participate in the post normalization shift of the
c                   floating-point significand in multiplication
c       machep  - the largest negative integer such that
c                 1.0+float(ibeta)**machep .ne. 1.0, except that
c                 machep is bounded below by  -(it+3)
c       negeps  - the largest negative integer such that
c                 1.0-float(ibeta)**negeps .ne. 1.0, except that
c                 negeps is bounded below by  -(it+3)
c       iexp    - the number of bits (decimal places if ibeta = 10)
c                 reserved for the representation of the exponent
c                 (including the bias or sign) of a floating-point
c                 number
c       minexp  - the largest in magnitude negative integer such that
c                 float(ibeta)**minexp is a positive floating-point
c                 number
c       maxexp  - the largest positive integer exponent for a finite
c                 floating-point number
c       eps     - the smallest positive floating-point number such
c                 that  1.0+eps .ne. 1.0. in particular, if either
c                 ibeta = 2  or  irnd = 0, eps = float(ibeta)**machep.
c                 otherwise,  eps = (float(ibeta)**machep)/2
c       epsneg  - a small positive floating-point number such that
c                 1.0-epsneg .ne. 1.0. in particular, if ibeta = 2
c                 or  irnd = 0, epsneg = float(ibeta)**negeps.
c                 otherwise,  epsneg = (ibeta**negeps)/2.  because
c                 negeps is bounded below by -(it+3), epsneg may not
c                 be the smallest number which can alter 1.0 by
c                 subtraction.
c       xmin    - the smallest non-vanishing floating-point power of the
c                 radix.  in particular,  xmin = float(ibeta)**minexp
c       xmax    - the largest finite floating-point number.  in
c                 particular   xmax = (1.0-epsneg)*float(ibeta)**maxexp
c                 note - on some machines  xmax  will be only the
c                 second, or perhaps third, largest number, being
c                 too small by 1 or 2 units in the last digit of
c                 the significand.
c
c     latest revision - october 22, 1979
c
c     author - w. j. cody
c              argonne national laboratory
c
c-----------------------------------------------------------------
      one = dble(float(1))
      zero = 0.0d0
c-----------------------------------------------------------------
c     determine ibeta,beta ala malcolm
c-----------------------------------------------------------------
      a = one
   10 a = a + a
         if (((a+one)-a)-one .eq. zero) go to 10
      b = one
   20 b = b + b
         if ((a+b)-a .eq. zero) go to 20
      ibeta = int(sngl((a + b) - a))
      beta = dble(float(ibeta))
c-----------------------------------------------------------------
c     determine it, irnd
c-----------------------------------------------------------------
      it = 0
      b = one
  100 it = it + 1
         b = b * beta
         if (((b+one)-b)-one .eq. zero) go to 100
      irnd = 0
      betam1 = beta - one
      if ((a+betam1)-a .ne. zero) irnd = 1
c-----------------------------------------------------------------
c     determine negep, epsneg
c-----------------------------------------------------------------
      negep = it + 3
      betain = one / beta
      a = one
c
      do 200 i = 1, negep
         a = a * betain
  200 continue
c
      b = a
  210 if ((one-a)-one .ne. zero) go to 220
         a = a * beta
         negep = negep - 1
      go to 210
  220 negep = -negep
      epsneg = a
      if ((ibeta .eq. 2) .or. (irnd .eq. 0)) go to 300
      a = (a*(one+a)) / (one+one)
      if ((one-a)-one .ne. zero) epsneg = a
c-----------------------------------------------------------------
c     determine machep, eps
c-----------------------------------------------------------------
  300 machep = -it - 3
      a = b
  310 if((one+a)-one .ne. zero) go to 320
         a = a * beta
         machep = machep + 1
      go to 310
  320 eps = a
      if ((ibeta .eq. 2) .or. (irnd .eq. 0)) go to 350
      a = (a*(one+a)) / (one+one)
      if ((one+a)-one .ne. zero) eps = a
c-----------------------------------------------------------------
c     determine ngrd
c-----------------------------------------------------------------
  350 ngrd = 0
      if ((irnd .eq. 0) .and. ((one+eps)*one-one) .ne. zero) ngrd = 1
c-----------------------------------------------------------------
c     determine iexp, minexp, xmin
c
c     loop to determine largest i and k = 2**i such that
c         (1/beta) ** (2**(i))
c     does not underflow
c     exit from loop is signaled by an underflow.
c-----------------------------------------------------------------
      i = 0
      k = 1
      z = betain
  400 y = z
         z = y * y
c-----------------------------------------------------------------
c        check for underflow here
c-----------------------------------------------------------------
         a = z * one
         if ((a+a .eq. zero) .or. (dabs(z) .ge. y)) go to 410
         i = i + 1
         k = k + k
      go to 400
  410 if (ibeta .eq. 10) go to 420
      iexp = i + 1
      mx = k + k
      go to 450
c-----------------------------------------------------------------
c     for decimal machines only
c-----------------------------------------------------------------
  420 iexp = 2
      iz = ibeta
  430 if (k .lt. iz) go to 440
         iz = iz * ibeta
         iexp = iexp + 1
      go to 430
  440 mx = iz + iz - 1
c-----------------------------------------------------------------
c     loop to determine minexp, xmin
c     exit from loop is signaled by an underflow.
c-----------------------------------------------------------------
  450 xmin = y
         y = y * betain
c-----------------------------------------------------------------
c        check for underflow here
c-----------------------------------------------------------------
         a = y * one
         if (((a+a) .eq. zero) .or. (dabs(y) .ge. xmin)) go to 460
         k = k + 1
      go to 450
  460 minexp = -k
c-----------------------------------------------------------------
c     determine maxexp, xmax
c-----------------------------------------------------------------
      if ((mx .gt. k+k-3) .or. (ibeta .eq. 10)) go to 500
      mx = mx + mx
      iexp = iexp + 1
  500 maxexp = mx + minexp
c-----------------------------------------------------------------
c     adjust for machines with implicit leading
c     bit in binary significand and machines with
c     radix point at extreme right of significand
c-----------------------------------------------------------------
      i = maxexp + minexp
      if ((ibeta .eq. 2) .and. (i .eq. 0)) maxexp = maxexp - 1
      if (i .gt. 20) maxexp = maxexp - 1
      if (a .ne. y) maxexp = maxexp - 2
      xmax = one - epsneg
      if (xmax*one .ne. xmax) xmax = one - beta * epsneg
      xmax = xmax / (beta * beta * beta * xmin)
      i = maxexp + minexp + 3
      if (i .le. 0) go to 520
c
      do 510 j = 1, i
          if (ibeta .eq. 2) xmax = xmax + xmax
          if (ibeta .ne. 2) xmax = xmax * beta
  510 continue
c
  520 return
c     ---------- last card of dmchar ----------
      end
