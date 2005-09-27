      SUBROUTINE CABASE(HELP,ARRAY,X,ERROR)
 
C**** Cancels basis functions.
 
      INTEGER HELP,NBT,X,ARRAY(X),I,J
      CHARACTER ERROR*(*)

      IF(HELP.EQ.1) THEN
        WRITE(6,*) 'Hello C'
        WRITE(6,*) 'Hello Python'
        WRITE(6,*) 'Array X Size = ',X

        ARRAY(1) = 55
C        DO J=1,Y
          DO I=1,X
            WRITE(6,*) 'cabase array of',I,' = ',ARRAY(I)
          ENDDO
C        ENDDO

        WRITE(6,*) 'ARRAY = ',ARRAY
        WRITE(6,*) 'ERROR= ',ERROR
          
      ELSE
        NBT=0
        HELP = 23
        WRITE(6,*) 'ARRAY(1) = ',ARRAY(1)
        WRITE(6,*) 'ERROR= ',ERROR
      ENDIF
 
      RETURN
      END

C end of file
