C     Sorts an array arr(1:N) into ascending numerical order 
C      using the QuickSort algorithm.  On output arr is replaced with its
C      sorted rearrangement.
      SUBROUTINE DQSORT(N,ARR)
CF2PY INTENT(IN,OUT,COPY), ARR
CF2PY INTENT(HIDE), DEPEND(ARR), N=len(ARR)
      INTEGER N,M,NSTACK
      REAL*8 ARR(N)
      PARAMETER (M=7, NSTACK=100)
      INTEGER I, IR, J, JSTACK, K, L, ISTACK(NSTACK)
      REAL*8 A, TEMP

      JSTACK = 0
      L = 1
      IR = N
 1    IF(IR-L.LT.M)THEN
         DO  J=L+1,IR
            A = ARR(J)
            DO I = J-1,L,-1
               IF (ARR(I).LE.A) GOTO 2
               ARR(I+1)=ARR(I)
            ENDDO
            I = L-1
 2          ARR(I+1) = A
         ENDDO
         
         IF(JSTACK.EQ.0)RETURN
         IR=ISTACK(JSTACK)
         L=ISTACK(JSTACK-1)
         JSTACK = JSTACK - 2
         
      ELSE
         K = (L+IR)/2
         TEMP = ARR(K)
         ARR(K) = ARR(L+1)
         ARR(L+1) = TEMP
         IF(ARR(L).GT.ARR(IR))THEN
            TEMP = ARR(L)
            ARR(L) = ARR(IR)
            ARR(IR) = TEMP
         ENDIF
         IF(ARR(L+1).GT.ARR(IR))THEN
            TEMP=ARR(L+1)
            ARR(L+1)=ARR(IR)
            ARR(IR)=TEMP
         ENDIF
         IF(ARR(L).GT.ARR(L+1))THEN
            TEMP=ARR(L)
            ARR(L) = ARR(L+1)
            ARR(L+1) = TEMP
         ENDIF

         I=L+1
         J=IR
         A=ARR(L+1)
 3       CONTINUE
             I=I+1
         IF(ARR(I).LT.A)GOTO 3
 4       CONTINUE
             J=J-1
         IF(ARR(J).GT.A)GOTO 4
         IF(J.LT.I)GOTO 5
         TEMP = ARR(I)
         ARR(I) = ARR(J)
         ARR(J) = TEMP
         GOTO 3
 5       ARR(L+1) = ARR(J)
         ARR(J) = A
         JSTACK = JSTACK + 2
         IF(JSTACK.GT.NSTACK)RETURN
         IF(IR-I+1.GE.J-1)THEN
            ISTACK(JSTACK)=IR
            ISTACK(JSTACK-1)=I
            IR=J-1
         ELSE
            ISTACK(JSTACK)=J-1
            ISTACK(JSTACK-1)=L
            L=I
         ENDIF
      ENDIF
      GOTO 1
      END

C     Finds repeated elements of ARR and their occurrence incidence
C     reporting the result in REPLIST and REPNUM respectively.
C     NLIST is the number of repeated elements found.
C     Algorithm first sorts the list and then walks down it
C       counting repeats as they are found.
      SUBROUTINE DFREPS(ARR,N,REPLIST,REPNUM,NLIST)
CF2PY INTENT(IN), ARR
CF2PY INTENT(OUT), REPLIST
CF2PY INTENT(OUT), REPNUM
CF2PY INTENT(OUT), NLIST
CF2PY INTENT(HIDE), DEPEND(ARR), N=len(ARR)
      REAL*8 REPLIST(N), ARR(N)
      REAL*8 LASTVAL
      INTEGER REPNUM(N)
      INTEGER HOWMANY, REPEAT, IND, NLIST, NNUM

      CALL DQSORT(N,ARR)
      LASTVAL = ARR(1)
      HOWMANY = 0
      IND = 2
      NNUM = 1
      NLIST = 1
      REPEAT = 0
      DO WHILE(IND.LE.N)
         IF(ARR(IND).NE.LASTVAL)THEN
            IF (REPEAT.EQ.1)THEN
               REPNUM(NNUM)=HOWMANY+1
               NNUM=NNUM+1 
               REPEAT=0
               HOWMANY=0
            ENDIF
         ELSE
            HOWMANY=HOWMANY+1
            REPEAT=1
            IF(HOWMANY.EQ.1)THEN
               REPLIST(NLIST)=ARR(IND)
               NLIST=NLIST+1
            ENDIF
         ENDIF
         LASTVAL=ARR(IND)
         IND=IND+1
      ENDDO
      IF(REPEAT.EQ.1)THEN
         REPNUM(NNUM)=HOWMANY+1
      ENDIF
      NLIST = NLIST - 1
      END



         
         
      
