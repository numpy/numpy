      SUBROUTINE CABASE(HELP,ARRAY,X,Y,ARRAYB,A,B,ERROR)
 
C**** Cancels basis functions.
 
      INTEGER HELP,NBT,X,Y,ARRAY(X,Y),A,B,I,J
      REAL ARRAYB(A,B)
      CHARACTER ERROR*100

      IF(HELP.EQ.1) THEN
        WRITE(6,*) 'Hello C'
        WRITE(6,*) 'Hello Python'
        WRITE(6,*) 'Array X Size = ',X
        WRITE(6,*) 'Array Y Size = ',Y

        DO J=1,Y
          DO I=1,X
            WRITE(6,*) 'cabase array of',I,',',J,' = ',ARRAY(I,J)
          ENDDO
        ENDDO

        WRITE(6,*) ARRAY
        WRITE(6,*) ARRAYB
        WRITE(6,*) ERROR
          
      ELSE
        NBT=0
        HELP = 23
        WRITE(6,*) 'NBT=',NBT
      ENDIF
 
      RETURN
      END

C end of file
